#include "compiled_model.hpp"
#include "infer_request.hpp"

namespace ov {
    namespace llama_cpp_plugin {
        std::vector<std::shared_ptr<ov::Node>> get_nodes_containing_name_with_shape(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            auto ops = model->get_ops();
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            std::copy_if(ops.begin(), ops.end(), std::back_inserter(found_weight_nodes),
                    [&weight_name, &shape](const std::shared_ptr<ov::Node>& val) {
                        if (!ov::is_type<ov::op::v0::Constant>(val)) return false;
                        std::shared_ptr<ov::op::v0::Constant> node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(val);
                        return val->get_friendly_name().find(weight_name) != std::string::npos &&
                               val->get_shape() == shape;
                    });
            return found_weight_nodes;
        }

        bool has_weight_matches(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            found_weight_nodes = get_nodes_containing_name_with_shape(model, weight_name, shape);
            return !found_weight_nodes.empty();
        }

        std::string get_weight_name_without_torch_postfix(std::string torch_weight_name) {
            size_t idx = torch_weight_name.rfind(".");
            if (idx == std::string::npos) return torch_weight_name;
            return std::string(torch_weight_name, 0, idx);
        }

        bool has_partial_weight_matches(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            found_weight_nodes = get_nodes_containing_name_with_shape(model, get_weight_name_without_torch_postfix(weight_name), shape);
            return !found_weight_nodes.empty();
        }

        std::shared_ptr<ov::op::v0::Constant> get_weight_by_name_and_shape(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            OPENVINO_ASSERT(has_weight_matches(model, weight_name, shape));
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            found_weight_nodes = get_nodes_containing_name_with_shape(model, weight_name, shape);

            if (found_weight_nodes.size() > 1) {
                std::cout << "VSHAMPOR: multiple matches for weight name " << weight_name << " and shape " << shape.to_string() << ", found ";
                for (const auto& node_ptr : found_weight_nodes) {
                    std::cout << node_ptr->get_friendly_name() << "(shape " << shape.to_string() << "),";
                }
                std::cout << "will take the first match" << std::endl;
            }
            std::shared_ptr<ov::Node> node_with_tensor = found_weight_nodes.front();
            OPENVINO_ASSERT(ov::is_type<ov::op::v0::Constant>(node_with_tensor));
            std::shared_ptr<ov::op::v0::Constant> const_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(node_with_tensor);
            return const_node_ptr;
        }

        LlamaCppModel::LlamaCppModel(const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::SoPtr<ov::IRemoteContext>& context,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor
                      ) : ICompiledModel(model, plugin, context, task_executor) {
            llama_model_params params = llama_model_default_params();
            params.use_mmap = false;
            params.vocab_only = false;

            auto rt_info = model->get_rt_info();
            OPENVINO_ASSERT(rt_info.count("gguf_kv_params") != 0);
            OPENVINO_ASSERT(rt_info.count("gguf_tensor_name_map") != 0);
            OPENVINO_ASSERT(rt_info.count("gguf_tensor_shape_map") != 0);

            RTMap& tensor_name_map = model->get_rt_info<RTMap&>("gguf_tensor_name_map");
            RTMap& tensor_shape_map = model->get_rt_info<RTMap&>("gguf_tensor_shape_map");
            RTMap& kv_params = model->get_rt_info<RTMap&>("gguf_kv_params");

            size_t gguf_version = kv_params["gguf_version"].as<size_t>();
            std::cout << "VSHAMPOR: parsed gguf_version " << gguf_version << std::endl;


            OPENVINO_ASSERT(tensor_name_map.size() == tensor_shape_map.size());
            size_t n_tensors = tensor_name_map.size();
            std::cout << "VSHAMPOR: got " << n_tensors << " tensors from rt_info\n";

            std::vector<struct gguf_tensor_info> tensor_infos;
            std::vector<void*> tensor_data_ptrs;

            auto tn_map_iter = tensor_name_map.begin();
            size_t num_found_tensors = 0;
            for (size_t i = 0; i < n_tensors; i++) {
                const std::string& ov_name = tn_map_iter->first;
                const std::string& llama_name = tn_map_iter->second.as<std::string>();
                ov::Shape expected_shape = tensor_shape_map[ov_name].as<std::string>();

                std::string search_name(ov_name);
                if (!has_weight_matches(model, search_name, expected_shape)) {
                    if (!has_partial_weight_matches(model, search_name, expected_shape)) {
                        std::cout << "VSHAMPOR: did not find the weight node for weight name " << ov_name << std::endl;
                        tn_map_iter++;
                        continue;
                    }
                    std::cout << "VSHAMPOR: found partial match for torch name " << ov_name << std::endl;
                    search_name = get_weight_name_without_torch_postfix(search_name);
                }

                num_found_tensors++;
                auto weight_const_node_ptr = get_weight_by_name_and_shape(model, search_name, expected_shape);
                auto weight_shape = weight_const_node_ptr->get_shape();

                gguf_tensor_info info;

                info.name.n = llama_name.length();
                info.name.data = (char*) llama_name.c_str();  // TODO (vshampor): either do this via const_cast, or will have to implement own structures for
                                                              // read-only data passing to llama_load_model_from_data
                info.n_dims = weight_shape.size();
                std::fill(std::begin(info.ne), std::begin(info.ne) + sizeof(info.ne), 0);
                std::copy(weight_shape.begin(), weight_shape.end(), info.ne);

                info.offset = 0;
                info.data = nullptr;

                tensor_infos.push_back(info);
                tensor_data_ptrs.push_back((void*)(weight_const_node_ptr->get_data_ptr())); // TODO (vshampor): danger - casts `const` away

                tn_map_iter++;
            }

            std::cout << "VSHAMPOR: found " << num_found_tensors << "/" << n_tensors << " tensors" << std::endl;

            llama_model_ptr = llama_load_model_from_data(n_tensors, tensor_infos.data(), 0 /* n_kv */, nullptr /* kv_data */, tensor_data_ptrs.data(),  params);
        }

        void LlamaCppModel::export_model(std::ostream& model) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        std::shared_ptr<const ov::Model> LlamaCppModel::get_runtime_model() const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        void LlamaCppModel::set_property(const ov::AnyMap& properties) {
            std::cout << "VSHAMPOR: attempted to set_property (did nothing)";
        }

        ov::Any LlamaCppModel::get_property(const std::string& name) const {
            if (ov::supported_properties == name) {
                return decltype(ov::supported_properties)::value_type(std::vector<PropertyName>());
            }
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        std::shared_ptr<ov::ISyncInferRequest> LlamaCppModel::create_sync_infer_request() const {
             return std::make_shared<LlamaCppSyncInferRequest>(std::static_pointer_cast<const LlamaCppModel>(shared_from_this()));
        }
    }
}  // namespace ov

#include "compiled_model.hpp"
#include "infer_request.hpp"
#include <openvino/op/constant.hpp>

namespace ov {
    namespace llama_cpp_plugin {
        class TensorWeightMatcher {
        public:
            // TODO (vshampor) implement this for faster weight node matching.
            // Use std::list, two passes - first for full name match, second for prefix-match; remove entries from list on match
            using RTInfoTensorName = std::string;
            using OvNodeName = std::string;
            using LlamaTensorName = std::string;

            TensorWeightMatcher(const std::shared_ptr<ov::Model>& model, std::map<RTInfoTensorName, ov::Shape> tensor_names_with_shapes_to_match) {
                std::multimap<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>> intermediate_matches_map;

                const auto node_vector = model->get_ops();
                std::list<std::shared_ptr<ov::op::v0::Constant>> const_nodes_in_model;
                for (const auto& node_ptr : node_vector) {
                    if (ov::is_type<ov::op::v0::Constant>(node_ptr)) const_nodes_in_model.push_back(ov::as_type_ptr<ov::op::v0::Constant>(node_ptr));
                }

                // full substring match pass
                std::map<RTInfoTensorName, ov::Shape> unmatched_rt_info_names_on_first_pass = extract_matches(intermediate_matches_map, tensor_names_with_shapes_to_match, const_nodes_in_model,
                        [](const std::string& substring, const std::string& source) { return source.find(substring) != std::string::npos; });

                // prefix substring match pass
                std::map<RTInfoTensorName, ov::Shape> unmatched_rt_info_names_on_second_pass = extract_matches(intermediate_matches_map, unmatched_rt_info_names_on_first_pass, const_nodes_in_model,
                        [](const std::string& substring, const std::string& source) {
                        return source.find(get_weight_name_without_torch_postfix(substring)) != std::string::npos; });

                for (auto it = intermediate_matches_map.begin(); it != intermediate_matches_map.end(); it = intermediate_matches_map.upper_bound(it->first)) {
                    // TODO: perf improvement by iterating with ++;
                    RTInfoTensorName rt_info_name = it->first;
                    if (intermediate_matches_map.count(rt_info_name) != 1) {
                        std::cout << "VSHAMPOR: multiple matches for weight name " << rt_info_name << " and shape " << it->second->get_shape().to_string() << ", found ";
                        auto range_it_pair = intermediate_matches_map.equal_range(rt_info_name);
                        for (auto multimatch_it = range_it_pair.first; multimatch_it != range_it_pair.second; multimatch_it++) {
                            auto node_ptr = multimatch_it->second;
                            std::cout << node_ptr->get_friendly_name() << "(shape " << node_ptr->get_shape().to_string() << "),";
                        }
                        std::cout << "will take the first match" << std::endl;
                    }
                    const auto& match = intermediate_matches_map.find(rt_info_name)->second;
                    m_rtinfo_name_to_weight_node_map[rt_info_name] = match;
                }
                std::cout << "VSHAMPOR: did not find the weight node for " << unmatched_rt_info_names_on_second_pass.size() << " weights:" << std::endl;
                for (const auto& unmatched_entry: unmatched_rt_info_names_on_second_pass) {
                    std::cout << '\t' << unmatched_entry.first << std::endl;
                }
            }

        std::unordered_map<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>> get_matches() { return m_rtinfo_name_to_weight_node_map; }

        private:
            std::map<RTInfoTensorName, ov::Shape> extract_matches(std::multimap<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>>& output_matches_map,
                                                                  const std::map<RTInfoTensorName, ov::Shape>& names_with_shapes_to_match,
                                                                  const std::list<std::shared_ptr<ov::op::v0::Constant>>& search_list,
                                                                  std::function<bool(const std::string& substring, const std::string& source)> name_match_predicate) {
                std::map<RTInfoTensorName, ov::Shape> unmatched_rt_info_names;
                for (const auto& pair: names_with_shapes_to_match) {
                    RTInfoTensorName rt_info_name = pair.first;
                    const ov::Shape& wanted_shape = pair.second;
                    bool matched = false;
                    for (auto it = search_list.begin(); it != search_list.end(); it++) {
                        auto node_ptr = *it;
                        const std::string& friendly_name = node_ptr->get_friendly_name();
                        if (name_match_predicate(rt_info_name, friendly_name) &&
                            node_ptr->get_shape() == wanted_shape) {
                            output_matches_map.insert(std::make_pair(rt_info_name, node_ptr));
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) unmatched_rt_info_names.insert(pair);
                }
                return unmatched_rt_info_names;
            }

            static std::string get_weight_name_without_torch_postfix(std::string torch_weight_name) {
                size_t idx = torch_weight_name.rfind(".");
                if (idx == std::string::npos) return torch_weight_name;
                return std::string(torch_weight_name, 0, idx);
            }

            size_t num_exact_matches = 0;
            size_t num_partial_matches = 0;
            std::unordered_map<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>> m_rtinfo_name_to_weight_node_map;
        };


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
            OPENVINO_ASSERT(rt_info.count("gguf_kv_types") != 0);
            OPENVINO_ASSERT(rt_info.count("gguf_tensor_name_map") != 0);
            OPENVINO_ASSERT(rt_info.count("gguf_tensor_shape_map") != 0);

            RTMap& tensor_name_map = model->get_rt_info<RTMap&>("gguf_tensor_name_map");
            RTMap& tensor_shape_map = model->get_rt_info<RTMap&>("gguf_tensor_shape_map");
            RTMap& kv_params = model->get_rt_info<RTMap&>("gguf_kv_params");
            RTMap& kv_types = model->get_rt_info<RTMap&>("gguf_kv_types");

            size_t gguf_version = model->get_rt_info<size_t>("gguf_version");
            std::cout << "VSHAMPOR: parsed gguf_version " << gguf_version << std::endl;

            // kv params
            OPENVINO_ASSERT(kv_params.size() == kv_types.size());
            size_t n_kv = kv_params.size();
            std::vector<gguf_kv> kv_vector;
            std::vector<std::string> kv_string_data;
            for (const auto& kv_pair: kv_params) {
                gguf_kv kv;

                const auto& key = kv_pair.first;
                kv.key.n = key.length();
                kv.key.data = (char*) key.c_str(); // TODO (vshampor) see equivalent case below

                uint32_t value_type = kv_types[key].as<uint32_t>();
                gguf_type gguf_value_type = (gguf_type) value_type;
                kv.type = gguf_value_type;
                switch (gguf_value_type) {
                    case GGUF_TYPE_UINT8:   kv.value.uint8    = kv_pair.second.as<uint8_t>();  break;
                    case GGUF_TYPE_INT8:    kv.value.int8     = kv_pair.second.as<int8_t>(); ; break;
                    case GGUF_TYPE_UINT16:  kv.value.uint16   = kv_pair.second.as<uint16_t>(); break;
                    case GGUF_TYPE_INT16:   kv.value.int16    = kv_pair.second.as<int16_t>();  break;
                    case GGUF_TYPE_UINT32:  kv.value.uint32   = kv_pair.second.as<uint32_t>(); break;
                    case GGUF_TYPE_INT32:   kv.value.int32    = kv_pair.second.as<int32_t>();  break;
                    case GGUF_TYPE_FLOAT32: kv.value.float32  = kv_pair.second.as<float>();    break;
                    case GGUF_TYPE_UINT64:  kv.value.uint64   = kv_pair.second.as<uint64_t>(); break;
                    case GGUF_TYPE_INT64:   kv.value.int64    = kv_pair.second.as<int64_t>();  break;
                    case GGUF_TYPE_FLOAT64: kv.value.float64  = kv_pair.second.as<double>();   break;
                    case GGUF_TYPE_BOOL:    kv.value.bool_    = kv_pair.second.as<bool>();     break;
                    case GGUF_TYPE_STRING:
                        kv_string_data.push_back(kv_pair.second.as<std::string>());
                        kv.value.str.data = (char*) kv_string_data.back().c_str(); break; // TODO (vshampor) see equivalent case below
                    case GGUF_TYPE_ARRAY:
                        {
                            OPENVINO_THROW("Array gguf kv params not supported yet");
                        } break;
                    default: OPENVINO_THROW("Invalid type of a GGUF kv-value");
                }
                kv_vector.push_back(kv);
            }

            // tensors
            OPENVINO_ASSERT(tensor_name_map.size() == tensor_shape_map.size());
            size_t n_tensors_in_rtinfo = tensor_name_map.size();
            std::cout << "VSHAMPOR: got " << n_tensors_in_rtinfo << " tensors from rt_info\n";

            std::vector<struct gguf_tensor_info> tensor_infos;
            std::vector<void*> tensor_data_ptrs;

            auto tn_map_iter = tensor_name_map.begin();
            size_t num_found_tensors = 0;
            std::map<std::string, ov::Shape> parsed_weights_to_search_for;
            for (const auto& rtinfo_name_and_llama_name : tensor_name_map) {
                const std::string& rtinfo_name = rtinfo_name_and_llama_name.first;
                const std::string& llama_name = rtinfo_name_and_llama_name.second.as<std::string>();
                ov::Shape expected_shape = tensor_shape_map[rtinfo_name].as<std::string>();
                parsed_weights_to_search_for[rtinfo_name] = expected_shape;
            }

            TensorWeightMatcher matcher{model, parsed_weights_to_search_for};
            std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> matches = matcher.get_matches();

            std::list<std::string> llama_name_storage;

            size_t n_tensors = 0;

            size_t offset = 0; // each tensor_info has to have a correct offset including padding, checked for in gguf_write_to_buf
            for (const auto& matched_weight_pair : matches) {
                std::string rt_info_name = matched_weight_pair.first;

                // Need to store the names in the list so that the passed c_str() pointers in tensor_infos to the llama names stay valid
                // until they get deepcopied in gguf/llama functions
                llama_name_storage.push_back(tensor_name_map[rt_info_name].as<std::string>());
                const std::string& llama_name = llama_name_storage.back();

                auto weight_const_node_ptr = matched_weight_pair.second;
                auto weight_shape = weight_const_node_ptr->get_shape();

                gguf_tensor_info info;

                info.type = GGML_TYPE_F32; // TODO (vshampor): better type assignment based on actual element type of the Constant node

                info.name.n = llama_name.length();
                info.name.data = (char*) llama_name.c_str();  // TODO (vshampor): either do this via const_cast, or will have to implement own structures for
                                                              // read-only data passing to llama_load_model_from_data
                info.n_dims = weight_shape.size();
                std::fill(std::begin(info.ne), std::begin(info.ne) + sizeof(info.ne), 1);
                std::copy(weight_shape.begin(), weight_shape.end(), info.ne);

                void* data_ptr = (void*)(weight_const_node_ptr->get_data_ptr()); // TODO (vshampor): danger - casts `const` away

                info.size = weight_const_node_ptr->get_byte_size();
                info.offset = offset;

                const size_t size_pad = GGML_PAD(info.size, GGUF_DEFAULT_ALIGNMENT);
                offset += size_pad;

                info.data = data_ptr;

                tensor_infos.push_back(info);
                tensor_data_ptrs.push_back(data_ptr);
                n_tensors++;
            }

            std::cout << "VSHAMPOR: found " << matches.size() << "/" << parsed_weights_to_search_for.size() << " tensors" << std::endl;

            // m_llama_model_ptr = llama_load_model_from_data(n_tensors, tensor_infos.data(), n_kv, kv_vector.data(), tensor_data_ptrs.data(),  params);
            //
            gguf_init_params gguf_params;
            gguf_params.no_alloc = false;
            gguf_params.ctx = nullptr;

            m_gguf_ctx = gguf_init_from_data(n_tensors, tensor_infos.data(), n_kv, kv_vector.data(), tensor_data_ptrs.data(), gguf_params);

            std::string fname = "/tmp/exported.gguf";
            std::cout << "VSHAMPOR: attempting export via gguf_write_to_file " << std::endl;
            std::cout << "VSHAMPOR: filename is  " << fname << std::endl;
            gguf_write_to_file(m_gguf_ctx, fname.c_str(), /* only_meta = */ false);
            std::cout << "VSHAMPOR: write finished." << fname << std::endl;

        }

        void LlamaCppModel::export_model(std::ostream& output_stream) const {
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

#include "compiled_model.hpp"
#include "infer_request.hpp"

namespace ov {
    namespace llama_cpp_plugin {
        LlamaCppModel::LlamaCppModel(const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::SoPtr<ov::IRemoteContext>& context,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor
                      ) : ICompiledModel(model, plugin, context, task_executor) {
            llama_model_params params = llama_model_default_params();
            params.use_mmap = false;
            params.vocab_only = false;
            // llama_model_ptr = llama_load_model_from_data(0 /* n_tensors */, nullptr /* tensor_info_data */, 0 /* n_kv */, nullptr /* kv_data */, nullptr /* tensor_data_ptrs */,  params);
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

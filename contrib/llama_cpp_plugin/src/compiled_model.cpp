#include "compiled_model.hpp"

namespace ov {
    namespace llama_cpp_plugin {
            void LlamaCppModel::export_model(std::ostream& model) const {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            std::shared_ptr<const ov::Model> LlamaCppModel::get_runtime_model() const {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            void LlamaCppModel::set_property(const ov::AnyMap& properties) {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            ov::Any LlamaCppModel::get_property(const std::string& name) const {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            std::shared_ptr<ov::ISyncInferRequest> LlamaCppModel::create_sync_infer_request() const {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }
    }
}  // namespace ov

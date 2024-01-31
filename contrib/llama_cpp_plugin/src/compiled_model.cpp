
#include "compiled_model.hpp"

namespace ov {
    namespace llama_cpp_plugin {
            virtual void LlamaCppModel::export_model(std::ostream& model) const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            virtual std::shared_ptr<const ov::Model> LlamaCppModel::get_runtime_model() const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            virtual void LlamaCppModel::set_property(const ov::AnyMap& properties) override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            virtual ov::Any LlamaCppModel::get_property(const std::string& name) const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            virtual std::shared_ptr<ov::ISyncInferRequest> LlamaCppModel::create_sync_infer_request() const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }
        };
    }
}  // namespace ov

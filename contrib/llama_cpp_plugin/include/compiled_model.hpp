#ifndef LLAMA_CPP_COMPILED_MODEL_HPP
#define LLAMA_CPP_COMPILED_MODEL_HPP

#include "openvino/runtime/icompiled_model.hpp"

namespace ov {
    namespace llama_cpp_plugin {
        class LlamaCppModel: public ICompiledModel {
        public:
            LlamaCppModel(const std::shared_ptr<ov::Model>& model,
                          const std::shared_ptr<const ov::IPlugin>& plugin,
                          const ov::SoPtr<ov::IRemoteContext>& context,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor
                          ): ICompiledModel(model, plugin, context, task_executor) {};
            /**
             * @brief Export compiled model to stream
             *
             * @param model output stream
             */
            virtual void export_model(std::ostream& model) const override;

            /**
             * @brief Returns runtime model
             *
             * @return OpenVINO Model which represents runtime graph
             */
            virtual std::shared_ptr<const ov::Model> get_runtime_model() const override;

            /**
             * @brief Allows to set property
             *
             * @param properties new plugin properties
             */
            virtual void set_property(const ov::AnyMap& properties) override;

            /**
             * @brief Returns property
             *
             * @param name Property name
             *
             * @return Property value
             *              virtual std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
            **/
            virtual ov::Any get_property(const std::string& name) const override;


        protected:
            /**
             * @brief Method creates infer request implementation
             *
             * @return Sync infer request
             */
            virtual std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
        };
    }
}  // namespace ov

#endif  // LLAMA_CPP_COMPILED_MODEL_HPP

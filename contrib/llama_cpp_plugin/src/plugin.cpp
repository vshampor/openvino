#include "plugin.hpp"
namespace ov {
    namespace llama_cpp_plugin {
        virtual std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties) const override {
            std::cout << "VSHAMPOR: compile_model called in C++" << std::endl;
            return std::shared_ptr<ov::ICompiledModel>();
        }


        /**
         * @brief Compiles model from ov::Model object, on specified remote context
         * @param model A model object acquired from ov::Core::read_model or source construction
         * @param properties A ov::AnyMap of properties relevant only for this load operation
         * @param context A pointer to plugin context derived from RemoteContext class used to
         *        execute the model
         * @return Created Compiled Model object
         */
        virtual std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties,
            const ov::SoPtr<ov::IRemoteContext>& context) const override {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        /**
         * @brief Sets properties for plugin, acceptable keys can be found in openvino/runtime/properties.hpp
         * @param properties ov::AnyMap of properties
         */
        virtual void LlamaCppPlugin::set_property(const ov::AnyMap& properties) override {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        /**
         * @brief Gets properties related to plugin behaviour.
         *
         * @param name Property name.
         * @param arguments Additional arguments to get a property.
         *
         * @return Value of a property corresponding to the property name.
         */
        virtual ov::Any LlamaCppPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const override {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

            /**
             * @brief Creates a remote context instance based on a map of properties
             * @param remote_properties Map of device-specific shared context remote properties.
             *
             * @return A remote context object
             */
            virtual ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::create_context(const ov::AnyMap& remote_properties) const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }
            /**
             * @brief Provides a default remote context instance if supported by a plugin
             * @param remote_properties Map of device-specific shared context remote properties.
             *
             * @return The default context.
             */
            virtual ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::get_default_context(const ov::AnyMap& remote_properties) const = 0;

            /**
             * @brief Creates an compiled model from an previously exported model using plugin implementation
             *        and removes OpenVINO Runtime magic and plugin name
             * @param model Reference to model output stream
             * @param properties A ov::AnyMap of properties
             * @return An Compiled model
             */
            virtual std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model,
                const ov::AnyMap& properties) const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            /**
             * @brief Creates an compiled model from an previously exported model using plugin implementation
             *        and removes OpenVINO Runtime magic and plugin name
             * @param model Reference to model output stream
             * @param context A pointer to plugin context derived from RemoteContext class used to
             *        execute the network
             * @param properties A ov::AnyMap of properties
             * @return An Compiled model
             */
            virtual std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model,
                const ov::SoPtr<ov::IRemoteContext>& context,
                const ov::AnyMap& properties) const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }

            /**
             * @brief Queries a plugin about supported layers in model
             * @param model Model object to query.
             * @param properties Optional map of pairs: (property name, property value).
             * @return An object containing a map of pairs an operation name -> a device name supporting this operation.
             */
            virtual ov::SupportedOpsMap LlamaCppPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
                const ov::AnyMap& properties) const override {
                OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
            }
        }  // namespace ov
#include "plugin.hpp"
namespace ov {
    namespace llama_cpp_plugin {
        LlamaCppPlugin::LlamaCppPlugin() : IPlugin() {
            set_device_name("LLAMA_CPP");
        }
        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties) const {
            std::cout << "VSHAMPOR: compile_model called in C++" << std::endl;
            return std::shared_ptr<ov::ICompiledModel>();
        }


        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties,
            const ov::SoPtr<ov::IRemoteContext>& context) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        void LlamaCppPlugin::set_property(const ov::AnyMap& properties) {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        ov::Any LlamaCppPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::create_context(const ov::AnyMap& remote_properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }
        ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::get_default_context(const ov::AnyMap& remote_properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }
        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model,
            const ov::AnyMap& properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model,
            const ov::SoPtr<ov::IRemoteContext>& context,
            const ov::AnyMap& properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        ov::SupportedOpsMap LlamaCppPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }
    }
}  // namespace ov

static const ov::Version version = {CI_BUILD_NUMBER, "llama_cpp_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::llama_cpp_plugin::LlamaCppPlugin, version)

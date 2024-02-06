#include "infer_request.hpp"

namespace ov {
    namespace llama_cpp_plugin {
        LlamaCppSyncInferRequest::LlamaCppSyncInferRequest(const std::shared_ptr<const LlamaCppModel>& compiled_model): ov::ISyncInferRequest(compiled_model) {
            std::cout << "VSHAMPOR: infer request ctor called\n";
        }
    void LlamaCppSyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
        std::cout << "VSHAMPOR: set_tensors_impl called\n";
    }

    void LlamaCppSyncInferRequest::infer() {
        std::cout << "VSHAMPOR: infer() called\n";
    };
    std::vector<ov::ProfilingInfo> LlamaCppSyncInferRequest::get_profiling_info() const {
        std::cout << "VSHAMPOR: get_profiling_info() called\n";
        return std::vector<ov::ProfilingInfo>{};
    };
    std::vector<ov::SoPtr<ov::IVariableState>> LlamaCppSyncInferRequest::query_state() const {
        std::cout << "VSHAMPOR: get_profiling_info() called\n";
        return std::vector<ov::SoPtr<ov::IVariableState>>{};
    }
    }
}  // namespace ov

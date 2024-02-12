#include "infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "llama.h"

namespace ov {
    namespace llama_cpp_plugin {

        void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                          const ov::element::Type& element_type,
                          const ov::Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::make_tensor(element_type, shape);
    } else {
        tensor->set_shape(shape);
    }
}

        LlamaCppSyncInferRequest::LlamaCppSyncInferRequest(const std::shared_ptr<const LlamaCppModel>& compiled_model): ov::ISyncInferRequest(compiled_model) {
            std::cout << "VSHAMPOR: infer request ctor called\n";
            // Allocate input/output tensors
            for (const auto& input : get_inputs()) {
                allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
                    // Can add a check to avoid double work in case of shared tensors
                    allocate_tensor_impl(tensor,
                                         input.get_element_type(),
                                         input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
                });
            }
            for (const auto& output : get_outputs()) {
                allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
                    // Can add a check to avoid double work in case of shared tensors
                    allocate_tensor_impl(tensor,
                                         output.get_element_type(),
                                         output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
                });
    }
        }
    void LlamaCppSyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
        std::cout << "VSHAMPOR: set_tensors_impl called\n";
    }

    void LlamaCppSyncInferRequest::infer() {
        std::cout << "VSHAMPOR: infer() called\n";
        auto& logit_output = get_outputs()[0];
        allocate_tensor(logit_output, [logit_output](ov::SoPtr<ov::ITensor>& tensor) { allocate_tensor_impl(tensor, logit_output.get_element_type(), ov::Shape{1, 42}); });
        std::cout << "VSHAMPOR: output tensors allocated\n";
        llama_model_params params = llama_model_default_params();
        std::cout << "VSHAMPOR: llama_model_params instantiated\n";
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

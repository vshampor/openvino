#include "openvino/openvino.hpp"

int main(int argc, char* argv[]) {
    ov::Core core;
    std::string model_path = "/home/vshampor/work/optimum-intel/ov_model/openvino_model.xml";

    std::cout << "VSHAMPOR: reading model\n";
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    std::cout << "VSHAMPOR: compiling model\n";
    ov::CompiledModel compiled_model = core.compile_model(model, "LLAMA_CPP");

    std::cout << "VSHAMPOR: compiled successfully\n";

    std::cout << "VSHAMPOR: creating infer request\n";
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    std::cout << "VSHAMPOR: infer request created\n";
    return 0;
}

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenVINO REQUIRED)

set(TARGET_NAME llama_cpp_runner)

add_executable(${TARGET_NAME}
               "${CMAKE_CURRENT_SOURCE_DIR}/runner.cpp"
              )
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)


add_executable(tensor_comparator
               "${CMAKE_CURRENT_SOURCE_DIR}/tensor_comparator.cpp"
              )
target_link_libraries(tensor_comparator PRIVATE ggml)

cmake_minimum_required(VERSION 3.18)

project(CudaBatchImageProcessing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(batch_image_processor src/main.cu)

set_target_properties(batch_image_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52;60;70;75;80;86")

target_compile_options(batch_image_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

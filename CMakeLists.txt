cmake_minimum_required(VERSION 3.5)

project(yolo_detector)

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies(REQUIRED)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/nvidia/TensorRT-8.2.5.1/include/)
link_directories(/home/nvidia/TensorRT-8.2.5.1/lib/)

ament_auto_add_library(${PROJECT_NAME}_lib
  src/model.cc
  src/post_process.cc
  src/pre_process.cu
  src/yolo_detector_node.cc
  src/yolo_detector.cc
  src/yolo_layer.cu
)

ament_auto_add_executable(${PROJECT_NAME}
  src/main.cpp
)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib cudart nvinfer)

ament_auto_package()


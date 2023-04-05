cmake_minimum_required(VERSION 3.5)

project(yolov5_tensorRT_inference)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

#include glog
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
include_directories(${glog_INCLUDE_DIRS})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/nvidia/TensorRT-8.5.3.1/include/)
link_directories(/home/nvidia/TensorRT-8.5.3.1/lib/)
# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(main main.cpp ${SRCS})
target_link_libraries(main nvinfer)
target_link_libraries(main cudart)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${glog_LIBRARIES})



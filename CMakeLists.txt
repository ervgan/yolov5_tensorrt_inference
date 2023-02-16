cmake_minimum_required(VERSION 3.16)

project(yolov5_tensorRT_inference)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

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
include_directories(/home/nvidia/TensorRT-8.2.5.1/include/)
link_directories(/home/nvidia/TensorRT-8.2.5.1/lib/)

include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp ${SRCS})
target_link_libraries(main nvinfer)
target_link_libraries(main cudart)
target_link_libraries(main myplugins)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${glog_LIBRARIES})



cmake_minimum_required(VERSION 3.10)
project(FastSam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE DEBUG)
set(BUILD_TEST ON)
# CMAKE_BUILD_TYPE DEBUG 命令将会被下面一行覆盖 ，，，，只能搭配RELEASE使用
# add_definitions(-O2 -pthread)
option(DEBUG_MODE "debug_mode" OFF)

# CUDA
set(CMAKE_BUILD_TYPE Debug)
# 由于find_package找到的不完全，故仍需手动set
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")
# OpenCV
find_package(OpenCV REQUIRED)
# message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /home/sp/TensorRT-8.4.1.5/include)
set(TensorRT_LIBRARIES /home/sp/TensorRT-8.4.1.5/lib)


message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")


add_subdirectory(fastsam)

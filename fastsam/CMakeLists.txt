set(fastsam_INCLUDE_DIR)
list(APPEND fastsam_INCLUDE_DIR
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
)

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
)

# FastSam
aux_source_directory(src FASTSAM_DIRS)
add_library(FASTSAM STATIC ${FASTSAM_DIRS})
target_include_directories(FASTSAM PUBLIC ${fastsam_INCLUDE_DIR}
        PUBLIC include)
target_link_directories(FASTSAM PUBLIC ${ALL_LIBS})
target_link_libraries(FASTSAM nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})

# build the test program
if (BUILD_TEST)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
    add_executable(fastsam_test test/test_fastsam.cpp)
    target_include_directories(fastsam_test PUBLIC ${fastsam_INCLUDE_DIR})
    target_link_libraries(fastsam_test FASTSAM)
endif (BUILD_TEST)
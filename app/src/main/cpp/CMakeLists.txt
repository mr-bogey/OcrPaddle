cmake_minimum_required(VERSION 3.10)

project(native)

set(Project_DIR ${CMAKE_SOURCE_DIR}/../../../../)

set(OpenCV_DIR ${Project_DIR}/OpenCV/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ncnn_DIR ${Project_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

add_library(native SHARED native.cpp ppocr.cpp)

target_link_libraries(
        native

        ncnn
        ${OpenCV_LIBS}

        log
        jnigraphics
)
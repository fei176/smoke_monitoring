cmake_minimum_required(VERSION 3.0)

project(smoke)

set(CMAKE_CXX_STANDARD 11)

set(cv_dir "")
set(boost_dir "")
set(onnx_dir "")
set(onnx_lib_dir "")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(OpenCV_DIR ${cv_dir})
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build/")
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
else()
endif()

find_package(OpenCV REQUIRED)

aux_source_directory("src/core" core)
aux_source_directory("src/http" http)
aux_source_directory("src/module" module)

# project boost onnx cv
include_directories(
    "include"
    "${boost_dir}"
    "${onnx_dir}"
    ${OpenCV_INCLUDE_DIRS}
    )

LINK_DIRECTORIES(
    "${onnx_lib_dir}"
    )

add_executable(smoke src/main.cpp ${core} ${http} ${module})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
target_link_libraries(smoke ${OpenCV_LIBS} "onnxruntime")
else()
target_link_libraries(smoke ${OpenCV_LIBS} "onnxruntime" "pthread")
endif()
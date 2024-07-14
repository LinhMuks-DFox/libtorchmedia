cmake_minimum_required(VERSION 3.16)
project(DeepLearningTemplate LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准为 C++14
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 libtorch 路径
list(APPEND CMAKE_PREFIX_PATH "/workspace/libtorch")

# 查找 libtorch 包
find_package(Torch REQUIRED)

# 查找 CUDA 包
find_package(CUDA REQUIRED)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES "75")

# 添加 fmt 子目录
add_subdirectory(fmt)

# 查找 sox 包
find_path(SOX_INCLUDE_DIR sox.h PATHS /usr/include /usr/local/include)
find_library(SOX_LIBRARY NAMES sox PATHS /usr/lib /usr/local/lib)

# 如果找不到 sox.h 或 libsox，抛出错误
if(NOT SOX_INCLUDE_DIR OR NOT SOX_LIBRARY)
    message(FATAL_ERROR "Could not find Sox library")
endif()

# 包含头文件目录
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${SOX_INCLUDE_DIR}
    include
    fmt/include
)

# 添加源文件
file(GLOB_RECURSE CPP_SOURCES pp_src/*.cpp)

# 添加可执行文件
add_executable(deep_learning_template ${CPP_SOURCES})

# 链接库
target_link_libraries(deep_learning_template
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    fmt
    ${SOX_LIBRARY}
)

set_property(TARGET deep_learning_template PROPERTY CXX_STANDARD 20)
set_property(TARGET deep_learning_template PROPERTY CUDA_STANDARD 14)

set_target_properties(deep_learning_template PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
)

set_target_properties(deep_learning_template PROPERTIES PREFIX "")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "Using libtorch from: ${TORCH_INSTALL_PREFIX}")
message(STATUS "Using sox from: ${SOX_INCLUDE_DIR}")
message(STATUS "Using CUDA from: ${CUDA_TOOLKIT_ROOT_DIR}")

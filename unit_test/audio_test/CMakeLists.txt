cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(audio_test)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加libtorch路径
set(Torch_DIR /workspace/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# 添加fmt路径，并指定二进制目录
add_subdirectory(/workspace/fmt /workspace/env_tester/build/fmt)

# 包含头文件路径
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(/workspace/fmt/include)
include_directories(/workspace/libtorchmedia/include)
# 添加可执行文件
add_executable(audio_test /workspace/unit_test/audio_test/main.cpp)
target_link_libraries(audio_test "${TORCH_LIBRARIES}" fmt)

# 设置编译标志
set_property(TARGET audio_test PROPERTY CXX_STANDARD 20)
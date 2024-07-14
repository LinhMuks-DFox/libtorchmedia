#!/bin/bash

# 移除现有的 build 目录
if [ -d "build" ]; then
    rm -rf build/
fi

# 创建新的 build 目录
mkdir build

# 切换到 build 目录
cd build

# 运行 cmake 生成构建文件
cmake ..

# 运行 make 构建项目
make

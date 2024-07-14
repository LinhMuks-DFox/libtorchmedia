#!/bin/bash
# 如果libtorch不存在，则下载
if [ ! -d "libtorch" ]; then
  echo "Downloading libtorch"
  curl -L https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip -O libtorch.zip
  mv libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip libtorch.zip
  unzip libtorch.zip -d .
  rm libtorch.zip
fi

# 如果fmt_lib不存在，则克隆
if [ ! -d "fmt" ]; then
  echo "Downloading fmt"
  git clone https://github.com/fmtlib/fmt.git
fi

poetry install 
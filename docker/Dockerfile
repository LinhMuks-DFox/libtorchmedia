# 基础镜像
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# 接受PY_VERSION作为构建参数
ARG PY_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    git \
    cmake \
    build-essential \
    sox \
    libsox-dev \
    libsndfile1 \
    llvm \
    clangd \
    clang-format \
    tzdata \
    software-properties-common

# 添加 deadsnakes PPA 并安装指定版本的 Python 和 pip
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python${PY_VERSION} \
    python${PY_VERSION}-distutils \
    python${PY_VERSION}-venv \
    python${PY_VERSION}-dev \
    python3-pip

# 设置指定版本的 Python 为默认
RUN ln -sf /usr/bin/python${PY_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# 安装Python依赖
RUN pip install nvitop

# 安装Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.in-project true

# 复制并运行依赖下载脚本
COPY download_dep.sh /workspace/download_dep.sh
WORKDIR /workspace
RUN chmod +x download_dep.sh

# 复制项目文件
COPY . /workspace

# 复制自定义的bashrc
COPY .bashrc /root/.bashrc

# 默认命令
CMD ["bash"]

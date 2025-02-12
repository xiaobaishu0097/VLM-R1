# Use the specified base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set pip to use Tsinghua mirror for faster downloads in China
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install flash-attn using pre-built wheel
RUN pip install --no-cache-dir \
    "https://gh-proxy.com/github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" || \
    pip install flash-attn --no-build-isolation

# Install additional required packages
RUN pip install \
    wandb==0.18.3 \
    tensorboardx \
    qwen_vl_utils \
    torchvision \
    git+https://gh-proxy.com/github.com/huggingface/transformers.git

# Copy local R1-V repository
COPY ./src/open-r1-multimodal /workspace/src/open-r1-multimodal

# Install R1-V
WORKDIR /workspace/src/open-r1-multimodal
RUN pip install -e ".[dev]"
WORKDIR /workspace

# Install vllm
RUN pip install vllm==0.7.2

# Set environment variables for better Python output
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]


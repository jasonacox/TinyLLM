# Create vLLM docker image with CUDA 12.1.1 and Ubuntu 22.04
# 
# This uses pip to install vLLM from PyPI, which is a much smaller image than
# building from source. However, if you need to run on older GPU architectures
# like Pascal (e.g. GTX 1060, P100) you will need to build from source. See the
# Dockerfile.source file.
#
# Prerequisites:
#   - NVIDIA driver and CUDA toolkit installed on host
#
# Author: Jason A. Cox,
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV PORT 8000

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get -y --no-install-recommends install \
      python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip wheel
RUN python3 -m pip install --no-cache-dir vllm

COPY entrypoint.sh /usr/local/bin/
CMD [ "entrypoint.sh" ]
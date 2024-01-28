# vLLM

vLLM is a fast, multithreaded LLM engine for running LLM inference on a variety of hardware. 
The project is located at https://github.com/vllm-project/vllm.

* OpenAI-compatible API server option.
* Multithreaded - Allows multiple simultaneous API calls.
* Works on single and multiple GPU systems using a variety of hardware.

## GPU support

The default installation and build of vLLM does not support Pascal (Torch architecture 6, sm_6x) or older Nvidia GPU hardware.
With some minor changes, vLLM can be changed to run on Pascal hardware.

### Nvidia GPU and Torch Architecture

| CUDA    | nvcc tag | Compute Capability** | GPU Arch     | Year | GPU Models                          |
|---------|----------|----------------------|--------------|------|-------------------------------------|
| 3.2 - 8 | sm_20 *  | 2.0                  | Fermi        | 2010 | GeForce 4xx, 5xx, 6xx               |
| 5 - 10  | sm_30 *  | 3.0 3.5 3.7          | Kepler       | 2012 | GeForce 7xx, Tesla Kxx              |
| 6 - 11  | sm_50 *  | 5.0 5.1 5.3          | Maxwell      | 2014 | GTX 9xx, Titan X, Jetson Nano       |
| 8++     | sm_60    | 6.0 6.1 6.2          | Pascal       | 2016 | 10xx, Tesla Pxx[xx]                 |
| 9++     | sm_70    | 7.0 7.2 (Xavier)     | Volta        | 2017 | Tesla V100, Titan V, Jetson Xavier  |
| 10++    | sm_75    | 7.5                  | Turing       | 2018 | 20xx, Titan RTX, Tesla T4           |
| 11.1++  | sm_80    | 8.0 8.6 8.7 (Orin)   | Ampere       | 2020 | RTX 30xx, Axx[xx], Jetson Orin      |
| 11.8++  | sm_89    | 8.9                  | Ada Lovelace | 2022 | RTX 4xxx, RTX 6xxx, Tesla L4x       |
| 12++    | sm_90    | 9.0 9.0a (Thor)      | Hopper       | 2022 | Hxxx, GHxxx                         |
| 12++    | sm_95    | 9.5                  | Blackwell    | 2024 | B1xx, GB1xx                         |

CUDA showing maximum version that supports this architecture. (*) Fermi and Kepler are deprecated from CUDA 9 onwards. Maxwell is deprecated from CUDA 11.6 onwards. (++) The rest are still supported by latest CUDA versions. (**) Values used in TORCH_CUDA_ARCH_LIST list. [References](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).


## Running vLLM on Pascal

You will need to build from source. Details TBD... Verified with current Git Commit: 220a476 

### Setup
```bash
# Steps
git clone https://github.com/vllm-project/vllm.git
cd vllm
mv Dockerfile Dockerfile.orig

# Add or edit files below
vi Dockerfile # create
vi entrypoint.sh # create
cp setup.py _setup.py
vi setup.py # make edits below
vi build.sh # create
vi run.sh # create
chmod +x *sh

# Build
./build.sh

# Run
./run.sh
```

Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
RUN apt-get update -y \
     && apt-get install -y python3-pip
WORKDIR /app
COPY . .
RUN python3 -m pip install -e .
EXPOSE 8001
COPY entrypoint.sh /usr/local/bin/
CMD [ "entrypoint.sh" ]
```

entrypoint.sh

```bash
#!/usr/bin/env bash
#
# From https://github.com/substratusai/vllm-docker

set -x

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null
then
    echo "nvidia-smi could not be found. Ensure NVIDIA drivers are installed."
    exit 1
fi

if [[ -z "${NUM_GPU}" ]]; then
    export NUM_GPU=$(nvidia-smi -L | wc -l)
fi

export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"${MODEL}"}

if [[ -z "${MODEL}" ]]; then
    echo "Missing required environment variable MODEL"
    exit 1
fi

additional_args=${EXTRA_ARGS:-""}
if [[ ! -z "${QUANTIZATION}" ]]; then
    if [[ -z "${DTYPE}" ]]; then
        echo "Missing required environment variable DTYPE when QUANTIZATION is set"
        exit 1
    else
        additional_args="${additional_args} -q ${QUANTIZATION} --dtype ${DTYPE}"
    fi
fi

if [[ ! -z "${GPU_MEMORY_UTILIZATION}" ]]; then
    additional_args="${additional_args} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
fi

if [[ ! -z "${MAX_MODEL_LEN}" ]]; then
    additional_args="${additional_args} --max-model-len ${MAX_MODEL_LEN}"
fi

python3 -m vllm.entrypoints.openai.api_server \
    --tensor-parallel-size ${NUM_GPU} \
    --worker-use-ray \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" ${additional_args}
```

setup.py (changes - patch)

```patch
--- _setup.py	2024-01-27 18:44:45.509406538 +0000
+++ setup.py	2024-01-28 00:02:23.581639719 +0000
@@ -18,7 +18,7 @@
 MAIN_CUDA_VERSION = "12.1"
 
 # Supported NVIDIA GPU architectures.
-NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}
+NVIDIA_SUPPORTED_ARCHS = {"6.0", "6.1", "6.2", "7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}
 ROCM_SUPPORTED_ARCHS = {"gfx90a", "gfx908", "gfx906", "gfx1030", "gfx1100"}
 # SUPPORTED_ARCHS = NVIDIA_SUPPORTED_ARCHS.union(ROCM_SUPPORTED_ARCHS)
 
@@ -184,9 +184,9 @@
     device_count = torch.cuda.device_count()
     for i in range(device_count):
         major, minor = torch.cuda.get_device_capability(i)
-        if major < 7:
+        if major < 6:
             raise RuntimeError(
-                "GPUs with compute capability below 7.0 are not supported.")
+                "GPUs with compute capability below 6.0 are not supported.")
         compute_capabilities.add(f"{major}.{minor}")
 
 ext_modules = []
```
build.sh

```bash
#!/bin/bash

echo "Build vllm docker image from source..."

nvidia-docker build -t vllm .
```

run.sh

```bash
#!/bin/bash

# vLLM Docker Container Image

echo "Starting vLLM..."

nvidia-docker run -d -p 8001:8001 --gpus=all --shm-size=10.24gb \
  -e MODEL=mistralai/Mistral-7B-Instruct-v0.1 \
  -e PORT=8001 \
  -e HF_HOME=/app/models \
  -e NUM_GPU=4 \
  -e EXTRA_ARGS="--dtype float --max-model-len 20000" \
  -v /data/models:/app/models \
  --name vllm \
  vllm 
  
echo "Printing logs (^C to quite)..."

docker logs vllm -f
```

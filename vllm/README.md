# vLLM

vLLM is a fast, multithreaded LLM engine for running LLM inference on a variety of hardware. 
The project is located at https://github.com/vllm-project/vllm.

* OpenAI-compatible API server option.
* Multithreaded - Allows multiple simultaneous API calls.
* Works on single and multiple GPU systems using a variety of hardware.

## Quick Start

The vLLM project has a helpful [Getting Started](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html) page that includes deploying with a Docker container (recommended). The following will get you started with the Mistral 7B Instruct model:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --ipc=host \
    --restart unless-stopped \
    --name vllm-mistral \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.1 --enforce-eager --max-model-len 24000 

# Watch the Logs
docker logs vllm-mistral -f
```

You can see the vLLM APIs at http://localhost:8000/docs

## Running vLLM on Pascal

The default installation and build of vLLM does not support Pascal (Torch architecture 6, sm_6x) or older Nvidia GPU hardware. To get it to run on Pascal GPUs, you will need to build from source. The steps below were verified with current Git Commit: 220a476 running on a Ubuntu 22.04 systems with Pascal GPUs (e.g. GTX 1060, Tesla P100).

The `compile.sh` script will download the vLLM source and build a vLLM container to run on Pascal GPUs.

```bash
# Clone this Repo
git clone https://github.com/jasonacox/TinyLLM.git
cd TinyLLM/vllm

# Run the Compile Helper Script
./compile.sh

# Run vLLM
./run.sh

# The trailing logs will be displayed so you can see the progress. Use ^C to exit without
# stopping the container. 
```

### Manual Details 

As an alternative to using the compile.sh script, you can manually clone the vLLM repo and create
and edit the files yourself:

1. Clone Repo

```bash
# Clone vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
# git checkout 220a476
```

2. Edit Dockerfile and CMakeList.txt:

```patch
--- _Dockerfile	2024-06-07 22:09:30.069782339 -0700
+++ Dockerfile	2024-06-07 22:10:02.357875428 -0700
@@ -35,7 +35,7 @@
 # can be useful for both `dev` and `test`
 # explicitly set the list to avoid issues with torch 2.2
 # see https://github.com/pytorch/pytorch/pull/123243
-ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
+ARG torch_cuda_arch_list='6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX'
 ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
 #################### BASE BUILD IMAGE ####################
 
--- _CMakeList.txt	2024-06-07 22:08:27.657601121 -0700
+++ CMakeLists.txt	2024-06-07 22:09:01.541699767 -0700
@@ -16,7 +16,7 @@
 set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")
 
 # Supported NVIDIA architectures.
-set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")
+set(CUDA_SUPPORTED_ARCHS "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0")
 
 # Supported AMD GPU architectures.
 set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")
```

3. Create build.sh ([link](./build.sh))

```bash
# Create Container
nvidia-docker build . -f Dockerfile --target vllm-openai --tag vllm
```

4. Create run.sh ([link](./run.sh))

```bash
# Run Container
nvidia-docker run -d --gpus all -shm-size=10.24gb -p 8000:8000 \
    -v $PWD/models:/root/.cache/huggingface \
    --env "HF_TOKEN={Your_Hugingface_Token}" \
    --restart unless-stopped \
    --name vllm \
    vllm \
    --host 0.0.0.0 \
    --model=mistralai/Mistral-7B-Instruct-v0.1 \
    --dtype=float \
    --max-model-len 20000

    # Additional arguments to pass to the API server on startup:
    # --gpu-memory-utilization 0.95
    # --dtype auto|half|float
    # --quantization awq
    # --disable-log-requests
    # --tensor-parallel-size NUM_GPU
    # --enforce-eager 
    # --served-model-name mistral

# Print Running Logs - ^C to Stop Viewing Logs
docker logs vllm -f
```

## Nvidia GPU and Torch Architecture

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

Use this to determine GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`

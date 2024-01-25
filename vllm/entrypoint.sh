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

export NUM_GPU=$(nvidia-smi -L | wc -l)
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
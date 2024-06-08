#!/usr/bin/env bash
#
# Script to run the vLLM engine in a docker container.
# This will be called during docker run.
#
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM
#
# Credit:
#   https://github.com/substratusai/vllm-docker

set -x

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null
then
    echo "ERROR: No nvidia-smi found. Ensure NVIDIA drivers are installed."
    exit 1
fi

# Detect number of GPUs if not provided
if [[ -z "${NUM_GPU}" ]]; then
    export NUM_GPU=$(nvidia-smi -L | wc -l)
fi
echo "NUM_GPU=${NUM_GPU}"

# Set the model name from the MODEL environment variable
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"${MODEL}"}
if [[ -z "${MODEL}" ]]; then
    echo "ERROR: Missing environment variable MODEL"
    exit 1
fi

# Additional arguments to pass to the API server on startup
additional_args=${EXTRA_ARGS:-""}
if [[ ! -z "${QUANTIZATION}" ]]; then
    if [[ -z "${DTYPE}" ]]; then
        echo "ERROR: Missing environment variable DTYPE when QUANTIZATION is set"
        exit 1
    else
        additional_args="${additional_args} -q ${QUANTIZATION} --dtype ${DTYPE}"
    fi
elif [[ ! -z "${DTYPE}" ]]; then
    additional_args="${additional_args} --dtype ${DTYPE}"
fi
if [[ ! -z "${GPU_MEMORY_UTILIZATION}" ]]; then
    additional_args="${additional_args} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
fi
if [[ ! -z "${MAX_MODEL_LEN}" ]]; then
    additional_args="${additional_args} --max-model-len ${MAX_MODEL_LEN}"
fi

# Start the vLLM OpenAI API compatible server
python3 -m vllm.entrypoints.openai.api_server \
    --tensor-parallel-size ${NUM_GPU} \
    --worker-use-ray \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" ${additional_args}

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

You will need to build from source. Details TBD...

#vLLM

vLLM is a fast, multithreaded LLM engine for running LLM inference on a variety of hardware. 
The project is located at https://github.com/vllm-project/vllm.

* OpenAI-compatible API server option.
* Multithreaded - Allows multiple simultaneous API calls.
* Works on single and multiple GPU systems using a variety of hardware.

## GPU support

The default installation and build of vLLM does not support Pascal (Torch architecture 6) or older Nvidia GPU hardware.
With some minor changes, vLLM can be changed to run on Pascal hardware.

### Nvidia GPUs

| nvcc tag | TORCH_CUDA_ARCH_LIST | GPU Arch | Year | GPU Models |
|----------|----------------------|----------|------|---------|
| sm_50    | 5.0 5.1 5.3          | Maxwell  | 2014 | GTX 9xx |
| sm_60    | 6.0 6.1 6.2          | Pascal  | 2016 | 10xx, Pxx[xx] |
| sm_70    | 7.0 7.2              | Volta  | 2017 | Titan V |
| sm_75    | 7.5                  | Turing  | 2018 | 20xx |
| sm_80    | 8.0 8.6 8.7          | Ampere  | 2020 | RTX 30xx, Axx[xx] |
| sm_89    | 8.9                  | Ada  | 2022 | RTX xxxx |
| sm_90    | 9.0 9.0a             | Hopper  | 2022 | H100 |

## Running vLLM on Pascal

You will need to build from source. Details TBD...


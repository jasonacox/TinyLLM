# TinyLLM
This project helps you build a small locally hosted LLM using consumer grade hardware. The intent is to be able to build, train, tune and run LLM models locally.

## Hardware Specs

The reference hardware:

* CPU: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz
* Memory: 8GB DDR4 (more recommended)
* Disk: 128G SSD Boot, 4TB HDD
* GPU: NVIDIA GeForce GTX 1060 6GB

## Setup

```bash
# Clone the llama.cpp project
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Linux - Build for Nvidia GPU using CMake
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON   # Omit -DLLAMA_CUBLAS=ON for CPU only
cmake --build . --config Release
```

## Using LLaMA-2 7B - 5-bit Quantized Model

```bash
# Download the LLaMA-2 7B GGUF model from Hugging Face.
cd models
# Could use llama-2-7b-chat.Q3_K_M.gguf for small 3 bit version for smaller hardware
wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
cd ..

# Run interactive chat.
./build/bin/main -m models/llama-2-7b-chat.Q5_K_M.gguf \
    -t 4 \
    --color \
    -c 4096 \
    --temp 0.7 \
    --gpu-layers 32 \
    -n -1 \
    -i -ins 

Where:
    -m models/llama-2-7b-chat.Q5_K_M.gguf   # The model
    -t 4                                    # change to match number of CPU cores
    -c 4096                                 # context length
    --temp 0.7                              # randomness 
    --gpu-layers 32                         # number of layers to offload to GPU - remove if cpu only
    -n -1 --color                           # options 
    -i -ins                                 # interactive mode and instruction
```

See https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md for more details on inference parameters.

Example chat using llama.cpp interactive mode:

```bash
./main -t 4 -m models/llama-2-7b-chat.Q5_K_M.gguf \
    --color -c 4096 --temp 0.7 --gpu-layers 32 -n -1 -i -ins
```

```
> Pick a color
Green

> What is a llama?
A llama is a large, domesticated mammal that is native to South America. It has long, shaggy fur and distinctive ears with a distinctive curled-over shape. Llamas are used for their wool and as pack animals in remote areas where cars cannot reach. They are also known for their calm and gentle nature.

> Write a haiku
Llama in the sun
Gentle eyes, shaggy coat
Soft as a cloud
```

This example model was run on a Ubuntu Linux host with an Intel i5-6500 CPU @ 3.20GHz, 8GB RAM and an Nvidia GTX 1060 GPU with 6GB VRAM.

https://github.com/jasonacox/ProtosAI/assets/836718/285101a0-1045-441d-9960-26d6c251db11

## Python Interface

The models built or downloaded here can be used by the [LLaMa-cpp-python](https://github.com/abetlen/llama-cpp-python) project.

```bash
# Linux OS - Build and Install with Nvidia GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

This will also build llama.cpp but includes the python bindings. Next, if you downloaded the Llama-2 LLM model above, you can test it using this python script:

```python
from llama_cpp import Llama

# Load model - use gpu for 32 of 35 NN layers to keep it within the 6GB VRAM limit
llm = Llama(model_path="models/llama-2-7b-chat.Q5_K_M.gguf", n_gpu_layers=32)

# Ask a question
question = "Name the planets in the solar system?"
print(f"Asking: {question}...")
output = llm(f"Q: {question} A: ", 
    max_tokens=64, stop=["Q:", "\n"], echo=True)

# Print answer
print("\nResponse:")
print(output['choices'][0]['text'])
```

## OpenAI API Compatible Server

The llama-cpp-python library has a built in OpenAI API compatible server. This can be used to host your model locally and use OpenAI API tools against your self-hosted LLM.

```bash
# Install Server that uses OpenAI API
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python[server]

# Run the API Server
python3 -m llama_cpp.server \
    --model ./models/llama-2-7b-chat.Q5_K_M.gguf \
    --host localhost \
    --n_gpu_layers 32 

# It will listen on port 8000
```

### Run as a Service

You can set up a Linux service using the [tinyllm.service](tinyllm.service) file:

```bash
# Clone this project for helper files
git clone https://github.com/jasonacox/TinyLLM.git
cd TinyLLM

# Edit the tinyllm.service to match your environment (ExecStart, WorkingDirectory & User)
vim tinyllm.service

# Copy the service file into systemd
sudo cp tinyllm.service /etc/systemd/system/

# Copy the init.d file for tinyllm
sudo cp tinyllm /etc/init.d

# Start and activate the service
sudo /etc/init.d/tinyllm start
sudo /etc/init.d/tinyllm enable

# Check status and logs
sudo /etc/init.d/tinyllm status
sudo /etc/init.d/tinyllm logs
```

## Chat using API

See the example [chat.py](chat.py) CLI Chatbot script that connects to this server and hosts
an interactive session with the LLM.

The example chat.py Features:
  * Use of OpenAI API (could be used to connect to the OpenAI service if you have a key)
  * Works with local hosted OpenAI compatible llama-cpp-python[server]
  * Retains conversational context for LLM
  * Uses response stream to render LLM chunks instead of waiting for full response

Example Test Run (`./chat.py`):

```
ChatBot - Greetings! My name is Jarvis. Enter an empty line to quit chat.

> What is your name?

Jarvis> Jarvis.

> What is today's date?

Jarvis> Today's date is September 10, 2023.

> What day of the week is it?

Jarvis> It is Sunday.

> Answer this riddle: Ram's mom has three children, Reshma, Raja and a third one. What is the name of the third child?

Jarvis> The answer to the riddle is "Ram."

> Pick a color.

Jarvis> Jarvis will choose blue.

> Now write a poem about that color.

Jarvis> Here is a short poem about the color blue:
Blue, the hue of the sky so high
A symbol of hope, and a sight to the eye
Soothing and calm, yet bold and bright
The color of serenity, and pure delight.

> What time is it?

Jarvis> The current time is 10:45 AM.

> Thank you very much!

Jarvis> You're welcome! Is there anything else I can assist you with?
```

## Web Based Chatbot

The above CLI chat was converted to a web based python flask app in the [chatbot](chatbot) folder. To run the web server:

```bash
# Install required packages
pip install openai flask flask-socketio

# Edit the server.py to adjust the global settings for your environment (e.g. openai.api_base)
# Run the chatbot web server
python3 server.py
```

Open http://127.0.0.1:5000 - Example session:

<img width="946" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/08097e39-9c00-4f75-8c9a-d329c886b148">


## Train

The llama.cpp project includes a `train-text-from-scratch` tool. Use `-h` to see the options or an example below.

```bash
# Create a text file to use for training
mkdir models/jason
curl -s https://github.com/jasonacox/ProtosAI/files/11715802/input.txt > models/jason/jason.txt

# Run the training
./train-text-from-scratch \
        --vocab-model models/ggml-vocab.bin \
        --checkpoint-in models/jason/jason.chk.bin \
        --checkpoint-out models/jason/jason.chk.bin \
        --model-out models/jason/jason.bin \
        --train-data models/jason/jason.txt \
        --ctx 32 --embd 256 --head 8 --layer 16 \
        -t 4 -b 32 --seed 42 --adam-iter 16 \
        --use-flash --print-details-interval 0 --predict 64 \
        -n 1 # adjust this for the number of iterations to run
```

## References

* LLaMa.cpp - https://github.com/ggerganov/llama.cpp
* LLaMa-cpp-python - https://github.com/abetlen/llama-cpp-python
* Video Tutorial - Train your own llama.cpp mini-ggml-model from scratch!: https://asciinema.org/a/592303
* How to run your own LLM GPT - https://blog.rfox.eu/en/Programming/How_to_run_your_own_LLM_GPT.html

## Additional Tools

* LlamaIndex - augment LLMs with our own private data - https://gpt-index.readthedocs.io/en/latest/index.html



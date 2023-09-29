# llmserver

The `llmserver` is a docker hosted version of the python llama_cpp.server which serves up a LLM with OpenAI API interface.

## Setup

### Install Nvidia Support for Docker

You can also run the `setup.sh` script.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Create Container

This will take a while to download and set up. You can also run `build.sh`.

```bash
docker build -t llmserver .
```

### Run Container

This will launch llmserver as a service, listening on port 8000. You can also run `run.sh`.

```bash
docker run \
    --runtime=nvidia --gpus all \
    -d \
    -p 8000:8000 \
    -v ./models:/app/models \
    -e MODEL=models/llama-2-7b-chat.Q5_K_M.gguf \
    -e N_GPU_LAYERS=32 \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    --name llmserver \
    --restart unless-stopped \
    llmserver
```

## Test

You can see the service running at http://localhost:8000/

To test the LLM, you will need to run the [chatbot](https://github.com/jasonacox/TinyLLM/tree/main/chatbot#web-based-chatbot) or the [chat.py](https://github.com/jasonacox/TinyLLM/blob/main/chat.py) command line utility:

```bash
# Test via CLI
python3 ../chat.py
```

```
ChatBot - Greetings! My name is Jarvis. Enter an empty line to quit chat.

> Hi 

Jarvis> Hello! How may I assist you today? Is there anything specific you would like to know or discuss?

> Pick a color.

Jarvis> Certainly! If I had to choose a color, I would select blue as it is a calming and versatile color that represents trust, loyalty, and wisdom. How about you? What is your favorite color?

> I love green.

Jarvis> Excellent choice! Green is a vibrant and natural color that symbolizes growth, harmony, and balance. It's also the color of many living things, including plants and trees. Is there anything else you would like to know or discuss?
```

## Optional Manual Setup

### Run as a Service

You can set up a Linux service using the tinyllm.service file:

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

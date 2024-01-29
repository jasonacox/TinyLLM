"""
Tool to Download models from HuggingFace

Usage:
   python download.py <model>
   python download.py mistralai/Mistral-7B-Instruct-v0.1

Requires:
   pip install huggingface_hub
   
Author: Jason A. Cox
23 Sept 2023
https://github.com/jasonacox/TinyLLM

"""

import os
import sys

print("TinyLLM Model Downloader")
print("")
print("This tool will download a model from HuggingFace.")
print("The model will be saved in the 'models' folder.")
print("")

# Get model from command line or prompt user
model = sys.argv[1] if len(sys.argv) > 1 else input("Enter Model: ")

# Set envrionment variable HF_HOME
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")

print("Downloading model: ", model)

# Download models
from huggingface_hub import snapshot_download
snapshot_download(repo_id=model)

print("Done.")


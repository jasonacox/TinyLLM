# LLM Load Generator and Performance Report
#
# Author: Jason A. Cox
# Date: 27 Apr 2024

import time
import threading
from openai import OpenAI

# Default Settings
model="mistralai/Mistral-Nemo-Instruct-2407"
base_url="http://localhost:8000/v1"
token="token-abc123"
max_sessions=1024

# Ask user for settings
print("LLM Load Generator and Performance Report")
print()
base_url = input(f"Enter vLLM URL [{base_url}]: ") or base_url
token = input(f"Enter API Key [{token}]: ") or token

# Attempt to poll the server to get the model list
llm = OpenAI(api_key=token, base_url=base_url)
try:
    models = llm.models.list()
except Exception as err:
    print("Warning: Unable to connect to server to get model list.")
    print(f"Error: {err}")
    print()
else:
    # build list of models
    if len(models.data) > 0:
        model_list = [model.id for model in models.data]
        if not model in model_list:
            model = model_list[0]
        # print models
        print()
        print("Available Models:")
        for m in models.data:
            print(f" - {m.id}")
        print()

model = input(f"Enter Model [{model}]: ") or model
max_sessions = int(input(f"Enter Max Sessions [{max_sessions}]: ") or max_sessions)
print()

# Globals
stats = {}
client = OpenAI(
    base_url=base_url,
    api_key=token,
)

def func(name):
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        max_tokens=64,
    )

    end_time = time.time()
    response_time = end_time - start_time
    stats[name] = {}
    stats[name]["response_time"] = response_time
    stats[name]["tokens"] = int(response.usage.completion_tokens)
    print(f" - [{name}] Received response in {end_time - start_time}s")
    print(response.choices[0])

def main(num_threads=256):
    for i in range(num_threads):
        threading.Thread(target=func, args=(f"Thread-{i}",)).start()
    # Wait for threads to finish
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()

report = {}
if __name__ == "__main__":
    print("Starting load test ramp...")
    print(f"Host: {base_url}")
    print(f"Model: {model}")
    i = 1
    while i <= max_sessions:
        print(f"Running {i} threads...")
        main_start = time.time()
        main(i)
        main_end = time.time()

        # Compute Stats
        total_response_time = sum(stats[name]['response_time'] for name in stats)
        total_tokens = sum(stats[name]['tokens'] for name in stats)
        average_response_time = total_response_time / len(stats)
        tokens_per_second = total_tokens / (main_end - main_start)
        tokens_per_thread = total_tokens / len(stats)
        report[i] = f"Total TPS: {tokens_per_second:.2f} - Average Thread TPS: {tokens_per_thread / average_response_time:.2f}"
        print("")
        if i == max_sessions:
            break
        if i * 2 > max_sessions:
            i = max_sessions
        else:
            i *= 2

    print("Load test complete.")
    print()

    print("Results:")
    for threads, result in report.items():
        print(f"Threads: {threads} - {result}")
    print("Done.")

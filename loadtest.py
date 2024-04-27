# LLM Generator Performance Report
# 
# Author: Jason A. Cox
# Date: 27 Apr 2024

import time
import threading
from openai import OpenAI

# Constants
SHORT_REPORT = True

# Globals
stats = {}
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

def func(name):
    start_time = time.time()

    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )

    end_time = time.time()
    response_time = end_time - start_time
    stats[name] = {}
    stats[name]["response_time"] = response_time
    stats[name]["tokens"] = int(response.usage.completion_tokens)
    print(f" - [{name}] Received response in {end_time - start_time}s")

def main(num_threads=256):
    for i in range(num_threads):
        threading.Thread(target=func, args=(f"Thread-{i}",)).start()
    # Wait for threads to finish
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()

report = {}
if __name__ == "__main__":
    print("Starting load test...")
    for i in (8, 16, 32, 64, 128, 256, 512):
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
        
    print("Load test complete.")
    print("Results:")
    for threads, result in report.items():
        print(f"Threads: {threads} - {result}")
    print("Done.")


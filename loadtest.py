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

def main():
    for i in range(256):
        threading.Thread(target=func, args=(f"Thread-{i}",)).start()
    # Wait for threads to finish
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()

if __name__ == "__main__":
    print("Starting load test...")
    main_start = time.time()
    main()
    main_end = time.time()

    # Compute Stats
    total_response_time = sum(stats[name]['response_time'] for name in stats)
    total_tokens = sum(stats[name]['tokens'] for name in stats)
    average_response_time = total_response_time / len(stats)
    tokens_per_second = total_tokens / (main_end - main_start)
    tokens_per_thread = total_tokens / len(stats)

    if SHORT_REPORT:
        print(f"Total TPS: {tokens_per_second} - Average Thread TPS: {tokens_per_thread / average_response_time}")
    else:
        print("\nPerformance Report:")
        print(f" - Total time: {main_end - main_start}s")
        print(f" - Total threads: {len(stats)}")
        print(f" - Total response time: {total_response_time}s")
        print(f" - Average response time: {average_response_time}s")
        print(f" - Min response time: {min(stats[name]['response_time'] for name in stats)}s")
        print(f" - Max response time: {max(stats[name]['response_time'] for name in stats)}s")
        print("")
        print(f" - Total tokens: {total_tokens}")
        print(f" - Tokens per second: {tokens_per_second}")
        print(f" - Average tokens per thread: {tokens_per_thread}")
        print(f" - Average tokens per second: {tokens_per_thread / average_response_time}")

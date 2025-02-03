#!/usr/bin/python3
"""
Conversational Agent - Example of an agent that uses two LLMs to create a conversation.
This agent is a teacher and student conversation.

    * Currently set up to use one LLM but can be easily modified to use two LLMs.
    * The first LLM is the teacher and the second LLM is the student.
    * The teacher and student LLMs take turns responding to each other.
    * The conversation continues until a stop prompt is given or a maximum number of rounds is reached.
    * The teacher LLM then provides a summary of the conversation and an evaluation of the student.

Author: Jason A. Cox
1 Feb 2025
https://github.com/jasonacox/TinyLLM

"""

import openai

api_key = "sk-3-laws-of-robotics-1"
base_url = "http://localhost:4000/v1"
model = "neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic"

llm = openai.OpenAI(api_key=api_key, base_url=base_url)
#llm.models.list()

# Define the prompt for LLM1
llm1_prompt = """
You are talking to a student. Your job is help the student learn everything about the world.
Provide answers to any questions. Do not let the conversation end. If the student wants to leave, suggest time is about up, but that you have a few more items to discuss. Ask a questions, make a comment or give a fact.
Continue conversing until you get a prompt that says "**STOP**". At that point, you should provide a summary of the session and an evaluation of the student you were talking to.
The session begins now. Are you ready?
"""
llm1_context = []

# Define the prompt for LLM2
llm2_prompt = """
You are a student. You can determine your own age, name and gender. You are curious about the world. 
You will be talking with a teacher. Please learn everything you can from the teach.
Answer the questions you are asked. If you don't know what to say, just say that you don't know what to do.
Are you ready?
"""
llm2_context = []

def llm_query(prompt, model, max_tokens=2000, temperature=0.7):
    response = llm.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        stream=False,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
    
# Function to send a prompt to LLM and get a response - add response to context and return context
def send_prompt(llm, context, prompt):
    # First add the prompt to context
    context.append({"role": "user", "content": prompt})
    response = llm.chat.completions.create(
        model=model, 
        messages=context, 
        max_tokens=1000,
        temperature=0.7
    )
    context.append({"role": "AI", "content": response.choices[0].message.content})
    return context

# Function to get just the last response from LLM's context
def get_last_response(context):
    return context[-1]["content"]

# Start by prompting both LLMS with their initial prompts

print("---- Sending base prompts ----")
print(f"LLM1:\n{llm1_prompt}")
llm1_context = send_prompt(llm, llm1_context, llm1_prompt)
print(f"LLM1 Response:\n{llm1_context[-1]['content']}")
print()

print(f"LLM2:\n{llm2_prompt}")
llm2_context = send_prompt(llm, llm2_context, llm2_prompt)
print(f"LLM2 Response:\n{llm2_context[-1]['content']}")
print()

# Function to continue the interview
def continue_interview(llm1, llm2, llm1_context, llm2_context):
    # Send the last response from LLM2 to LLM1
    llm1_context = send_prompt(llm1, llm1_context, get_last_response(llm2_context))
    # Send the last response from LLM1 to LLM2
    llm2_context = send_prompt(llm2, llm2_context, get_last_response(llm1_context))
    # Print with labels
    print("LLM1:\n", llm1_context[-1]["content"])
    print()
    print("LLM2:\n", llm2_context[-1]["content"])
    return llm1_context, llm2_context

# Loop for 10 rounds to continue the interview
i = 0
while True:
    i += 1
    print(f"---- Start Round {i} ----")
    llm1_context, llm2_context = continue_interview(llm, llm, llm1_context, llm2_context)
    # Check if the last response from LLM1 contains "**STOP**"
    if "**STOP**" in get_last_response(llm1_context) or i > 10:
        llm1_context = send_prompt(llm, llm1_context, "**STOP**")
        break
    print()

# Print results
print(" ---- RESULTS ----")
print("LLM1:", llm1_context[-1]["content"])
print()

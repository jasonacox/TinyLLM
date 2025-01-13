# Chain of Thought

## Prompt Options for Reasoning Models

The default prompt used by the Chatbot is rather basic. Instead of expecting a trained model that uses something like `<thinking>` tags, the Chatbot uses a three step chain of thought (CoT) process to mimic a reasoning model. This allows it to work with many different instruct models. The steps are as follows:

1. Determine if CoT makes sense. Pleasantries and appreciation prompts are not routed to the CoT engine.
2. The `chain_of_thought` template instructs model to think through the question and produce a final solution at the end. Tags can be used to help, but is not required. The output from the model is not displayed. Instead, it is sent to the summary step below.
3. The `chain_of_thought_summary` template instructs model to extract the "solution" from the above chain of thought output. 


### Default CoT Prompts

The following prompt is used in the Chatbot by default.

"chain_of_thought_check"

```
You are a language expert. 
Consider this prompt:
<prompt>{prompt}</prompt>
Categorize the request using one of these:
a) A request for information
b) A request for code
c) A greeting or word of appreciation
d) Something else
Answer with a, b, c or d only:
```

"chain_of_thought"

```
First, outline how you will approach answering the problem.
Break down the solution into clear steps.
Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress. 
Regularly evaluate progress. 
Be critical and honest about your reasoning process.
Use thoughts as a scratchpad, writing out all calculations and reasoning explicitly. 
Synthesize the final answer within <answer> tags, providing a clear informed and detailed conclusion.
Include relevant scientific and factual details to support your answer.
If providing an equation, make sure you define the variables and units.
Don't over analyze simple questions.
If asked to produce code, include the code block in the answer. 
Answer the following in an accurate way that a young student would understand: 
```

"chain_of_thought_summary"

```
Examine the following context:\n{context_str}

Provide the best conclusion based on the context.
Do not provide an analysis of the context. Do not include <answer> tags.
Include relevant scientific and factual details to support the answer.
If there is an equation, make sure you define the variables and units. Do not include an equation section if not needed.
If source code provided, include the code block and describe what it does. Do not include a code section otherwise.
Make sure the answer addresses the original prompt: {prompt}
```

### NovaSky-AI

The following prompt is derived from the paper [Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems](https://arxiv.org/pdf/2412.09413). This prompt template is used on a model fine tuned to support this but seems to be effective for most instruct models.

"chain_of_thought"

```
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format:
“‘
<|begin_of_thought|>
{thought with steps separated with "\n\n"}
<|end_of_thought|>
”’
Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:
“‘
<|begin_of_solution|>
{final formatted, precise, and clear solution}
<|end_of_solution|>
”’
Now, considering the above conversation thread, try to solve the following question through the above guidelines:
{prompt}
```

"chain_of_thought_summary"

```
Examine the following context:
“‘
{context_str}
”’

Do not add any comments. Extract solution only:
```

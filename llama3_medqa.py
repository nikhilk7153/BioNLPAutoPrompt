import transformers
import torch
import json
from transformers import AutoTokenizer,  AutoModelForCausalLM
from openai import OpenAI
from vllm import LLM, SamplingParams


# Load the Hugging Face model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with your model identifier

client = OpenAI(
    api_key="EMTPY",
    base_url="http://gpua046.delta.ncsa.illinois.edu:8000/v1",
)

system_prompt = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

def user_prompt(question, options):

    user_prompt = f"""Here is the question: 
    
    {question}

    Here are the potential choices:
    {question_with_options(options)}

    Please think step-by-step and generate your output in json:"""

    return user_prompt


def question_with_options(options):

   return "\n(A) " + options['A'] + "\n(B) " + options['B'] +  "\n(C) " + options['C'] + "\n(D) " + options['D']

test_data = []

with open("/scratch/bchx/nikhilk5/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl") as file:
    for line in file:
        qa_info = json.loads(line)
        test_data.append(qa_info)
       


#print("User prompt " + str(user_prompt(test_data[0]['question'], test_data[0]['options'])))


for item in test_data:
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt(item['question'], item['options'])},
        ], 
        stop=["<|im_end|>"]
    )

    print("ASSISTANT:", response.choices[0].message.content)


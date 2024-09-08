import dspy
from dspy.predict import Retry
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import importlib
from tqdm import tqdm
import json
importlib.reload(dspy)

llama = dspy.HFClientVLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", port=8000, url="http://gpua046.delta.ncsa.illinois.edu", tokens=4000)



dspy.settings.configure(lm=llama) 

import re

def eval_metric(true, prediction, trace=None):
    try:
        pred = prediction.answer
        matches = re.findall(r"\([A-D]\)", pred)
        parsed_answer = matches[-1] if matches else ""
        return parsed_answer == true.answer
    except:
        return False  


class GenerateAnswer(dspy.Signature):
    """You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options."""

    question = dspy.InputField(desc="Question")
    options = dspy.InputField(desc="Answer choice options")
    answer = dspy.OutputField(desc="You should respond with one of (A), (B), (C), or (D)") 


class MedQA(dspy.Module):

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question, options):
        response = self.generate_answer(question=question, options=options)
      
        #valid_response = "(A)" in response.answer or "(B)" in response.answer or "(C)" in response.answer or "(D)" in response.answer  

        #dspy.Suggest(valid_response, "You must respond with one of (A), (B), (C), or (D) as part of your answer.")

        return dspy.Prediction(answer=response.answer)
        
import json

train_data = []
val_data = []
test_data = []

with open("/scratch/bchx/nikhilk5/MedQA/questions/US/4_options/phrases_no_exclude_train.jsonl") as file:
    for line in file:
        qa_info = json.loads(line)
        train_data.append(qa_info)
        

with open("/scratch/bchx/nikhilk5/MedQA/questions/US/4_options/phrases_no_exclude_dev.jsonl") as file:
    for line in file:
        qa_info = json.loads(line)
        val_data.append(qa_info)


with open("/scratch/bchx/nikhilk5/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl") as file:
    for line in file:
        qa_info = json.loads(line)
        test_data.append(qa_info)


def question_with_options(options):

   return "\n(A) " + options['A'] + "\n(B) " + options['B'] +  "\n(C) " + options['C'] + "\n(D) " + options['D']


def generate_dspy_examples(dataset):

   examples = []
   
   for i in range(len(dataset)):
      options = question_with_options(dataset[i]['options'])
      
      example = dspy.Example({"question": dataset[i]['question'], "options": options, "answer": "(" + dataset[i]["answer_idx"] + ")"}).with_inputs("question", "options") 

      examples.append(example)

   return examples 


train_dspy_examples = generate_dspy_examples(train_data)
val_dspy_examples = generate_dspy_examples(val_data)
test_dspy_examples = generate_dspy_examples(test_data)

from dspy.evaluate import Evaluate

evaluate_test = Evaluate(devset=test_dspy_examples, metric=eval_metric, num_threads=3, display_progress=True, display_table=True)

medqa = MedQA()

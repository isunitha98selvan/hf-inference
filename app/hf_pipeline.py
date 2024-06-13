# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import os
import re
import time
from argparse import ArgumentParser

import pandas as pd
import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def main():
    parser = ArgumentParser(
        description="Load a HF model and generate responses",
    )

    parser.add_argument("--model_name_or_path", type=str, help="HF model name or path")
    parser.add_argument("--test_ds_path", type=str, help="HF test dataset path")
    parser.add_argument("--max_new_tokens", type=int, help="Max new tokens")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path should be in form 'username/dataset_name'",
    )
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    test_ds_path = args.test_ds_path
    max_new_tokens = args.max_new_tokens
    output_path = args.output_path


    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()

    # You can change the model to any LLM such as mistralai/Mistral-7B-v0.1 or meta-llama/Llama-2-7b-chat-hf
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map=distributed_state.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    test_ds = load_dataset(test_ds_path, split="test[:10]")
    test_ds = test_ds.to_pandas()
    
    messages = test_ds["messages"].tolist()
    labels = test_ds["LABEL"].tolist()
    inputs = list(zip(labels, messages))

    completions_per_process = []

    with distributed_state.split_between_processes(inputs) as input_rows:
        pipe = pipeline(
                "text-generation",
                model=model_name_or_path,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                device="cuda",
                return_full_text=False
            )
          
        for input in input_rows:
            label, prompt = input[0], input[1]
            message = [prompt[0]]
            response = pipe(message)[0]['generated_text'][-1]
            completions_per_process.append({"label": label, "text": response})

    completions_gather = gather_object(completions_per_process)
    # completions = completions_gather[: len(prompts)]
    distributed_state.print(completions_gather)

    # parse outputs
    
    scores, reasonings = [], [], []

    accuracy_fail = 0
    accuracy_pass = 0
    total_accuracy = 0
    count_fail = 0
    count_pass = 0

    for row in completions_gather:
        score, reasoning = None, None
        response = row['text']['content']
        label = row['label']

        reasoning_pattern = r'"REASONING":\s*\[(.*?)\]'
        score_pattern = r'"SCORE":\s*"(\w+)"'

        reasoning_match = re.search(reasoning_pattern, json_string, re.DOTALL)
        score_match = re.search(score_pattern, json_string)
    
        if reasoning_match:
            reasoning_values = reasoning_match.group(1).split("', '")
            reasonings.append(reasoning_values)
        if score_match:
            score = score_match.group(1)
        else:
            score_pattern = r'"SCORE":\s*"(\w+)"'
            score_match = re.search(score_pattern, content)
            if score_match:
                score = score_match.group(1)

        reasonings.append(reasoning)
        scores.append(score)

        if score == "PASS" and row['LABEL'] == "PASS":
            accuracy_pass += 1
        elif score == "FAIL" and row['LABEL'] == "FAIL":
            accuracy_fail += 1

        if row['LABEL'] == "PASS":
            count_pass += 1
        if row['LABEL'] == "FAIL":
            count_fail += 1

    total_accuracy = (accuracy_pass + accuracy_fail)
    accuracy_pass = accuracy_pass / len(test_ds)
    accuracy_fail = accuracy_fail / len(test_ds)

    print(f"Correct examples: {total_accuracy}   Accuracy: {total_accuracy/len(test_ds)}")
    if count_pass>0:
        print(f"Correct PASS examples: {accuracy_pass}   PASS Accuracy: {accuracy_pass/count_pass}")
    if count_fail>0:
        print(f"Correct FAIL examples: {accuracy_pass}   FAIL Accuracy: {accuracy_fail/count_fail}")
        


if __name__ == "__main__":
    main()

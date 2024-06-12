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
    # Need to set the padding token to the eos token for generation
    tokenizer.pad_token = tokenizer.eos_token

    test_ds = load_dataset(test_ds_path, split="test[:10]")
    test_ds = test_ds.to_pandas()
    prompts = test_ds["messages"].tolist()
    # labels = test_ds["LABEL"].tolist()

    # prompts = list(zip(messages, labels))

    batch_size = 2
    pad_to_multiple_of = 8

    formatted_prompts = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    padding_side_default = tokenizer.padding_side
    tokenizer.padding_side = "left"

    tokenized_prompts = [
        [idx, tokenizer(formatted_prompt[0][0]['content'], padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")]
            for idx, formatted_prompt in enumerate(formatted_prompts)
    ]

    tokenizer.padding_side = padding_side_default

    completions_per_process = []

    with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
        for messages in batched_prompts:
            idx, batch = messages[0], messages[1]
            # Move the batch to the device
            batch = batch.to(distributed_state.device)
            # We generate the text, decode it and add it to the list completions_per_process
            outputs = model.generate(**batch, max_new_tokens=max_new_tokens)
            generated_text = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            completions_per_process.extend((idx, generated_text))

    completions_gather = gather_object(completions_per_process)
    completions = completions_gather[: len(prompts)]
    distributed_state.print(completions)


if __name__ == "__main__":
    main()

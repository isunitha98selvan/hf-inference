import ast
import json
import os
import re
from argparse import ArgumentParser

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
accelerator = Accelerator()


def evaluate(model_name_or_path, test_ds_path, max_new_tokens, output_path):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map={"": accelerator.process_index}
    )
    model.eval()
    print(f"Loaded the model {model_name_or_path}")

    test_ds = load_dataset(test_ds_path, split="test[:10]")
    print(f"Loaded the dataset {test_ds_path}")

    test_ds = test_ds.to_pandas()
    prompts_all = test_ds["messages"].tolist()

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        # store output of generations in dict
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference, prompt by prompt
        for prompt in prompts:
            prompt_tokenized = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tokenized = model.generate(
                **prompt_tokenized, max_new_tokens=max_new_tokens
            )[0]

            # remove prompt from output
            output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]) :]

            # store outputs and number of tokens in result{}
            results["outputs"].append(tokenizer.decode(output_tokenized))
            results["num_tokens"] += len(output_tokenized)

        results = [
            results
        ]  # transform to list, otherwise gather_object() will not collect correctly

    print("results: ", results)

    # collect results from all the GPUs
    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])

        print(
            f"tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}"
        )

    # responses, scores, reasonings = [], [], []

    # accuracy_fail = 0
    # accuracy_pass = 0
    # total_accuracy = 0
    # count_fail = 0
    # count_pass = 0

    # for row in tqdm(test_ds):
    #     score, reasoning = None, None

    #     messages = [row['messages'][0]]
    #     response = pipe(messages)[0]['generated_text'][-1]
    #     responses.append(response)

    #     content = response['content']
    #     print(content)

    #     reasoning_pattern = r'"REASONING":\s*\[(.*?)\]'
    #     score_pattern = r'"SCORE":\s*(\w+)'

    #     reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
    #     score_match = re.search(score_pattern, content)

    #     if reasoning_match:
    #         reasoning = reasoning_match.group(1).split("', '")

    #     if score_match:
    #         score = score_match.group(1)
    #     else:
    #         score_pattern = r'"SCORE":\s*"(\w+)"'
    #         score_match = re.search(score_pattern, content)
    #         if score_match:
    #             score = score_match.group(1)
    #         else:
    #             print("Was unable to parse scores from following response: \n\n")
    #             print("The generated response is ", content)
    #             print("The correct label is : ", row['LABEL'])

    #     reasonings.append(reasoning)
    #     scores.append(score)

    #     if score == "PASS" and row['LABEL'] == "PASS":
    #         accuracy_pass += 1
    #     elif score == "FAIL" and row['LABEL'] == "FAIL":
    #         accuracy_fail += 1

    #     if row['LABEL'] == "PASS":
    #         count_pass += 1
    #     if row['LABEL'] == "FAIL":
    #         count_fail += 1

    # total_accuracy = (accuracy_pass + accuracy_fail)
    # accuracy_pass = accuracy_pass / len(test_ds)
    # accuracy_fail = accuracy_fail / len(test_ds)

    # print(f"Correct examples: {total_accuracy}   Accuracy: {total_accuracy/len(test_ds)}")
    # if count_pass>0:
    #     print(f"Correct PASS examples: {accuracy_pass}   PASS Accuracy: {accuracy_pass/count_pass}")
    # if count_fail>0:
    #     print(f"Correct FAIL examples: {accuracy_pass}   FAIL Accuracy: {accuracy_fail/count_fail}")

    # test_df = test_ds.to_pandas()
    # test_df['generated_text'] = responses
    # test_df['reasoning'] = reasonings
    # test_df['score'] = scores

    # dataset = Dataset.from_pandas(test_df)
    # dataset.push_to_hub(f"{output_path}")
    # print(f"Saved dataset to HF Hub : {output_path}")


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

    evaluate(
        args.model_name_or_path,
        args.test_ds_path,
        args.max_new_tokens,
        args.output_path,
    )


if __name__ == "__main__":
    main()

# python3 hf_inference.py --model_name_or_path microsoft/phi-1_5 --test_ds_path sunitha-ravi/financebench_perturb_labels --max_new_tokens 10 --output_path sunitha-ravi/dummy-results

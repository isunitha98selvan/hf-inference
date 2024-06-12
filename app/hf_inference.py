import ast
import torch
import json
import os
import re
import pandas as pd


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from datasets import Dataset, load_dataset
from tqdm import tqdm
from argparse import ArgumentParser

from dotenv import load_dotenv

load_dotenv()

print("GPU Device count: ", torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all 8 GPUs
print("GPU Device count: ", torch.cuda.device_count())

def evaluate(
    model_name_or_path,
    test_ds_path,
    max_new_tokens,
    output_path
):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
       model_name_or_path, device_map="cuda"
    )
    model.eval()
    print(f"Loaded the model {model_name_or_path}")

    test_ds = load_dataset(test_ds_path, split="test")
    
    print(f"Loaded the dataset {test_ds_path}")

    pipe = pipeline(
        "text-generation",
        model=model_name_or_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        num_workers=8,
        device="cuda"
    )

    responses, scores, reasonings = [], [], []

    accuracy_fail = 0
    accuracy_pass = 0
    total_accuracy = 0
    count_fail = 0
    count_pass = 0
    not_matched = 0

    for row in tqdm(test_ds):
        score, reasoning = None, None

        messages = [row['messages'][0]]
        response = pipe(messages)[0]['generated_text'][-1]
        responses.append(response)

        content = response['content']

        reasoning_pattern = r'"REASONING":\s*\[(.*?)\]'
        score_pattern = r'"SCORE":\s*(\w+)'

        reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
        score_match = re.search(score_pattern, content)

        if reasoning_match:
            reasoning = reasoning_match.group(1).split("', '")

        if score_match:
            score = score_match.group(1)
        else:
            score_pattern = r'"SCORE":\s*"(\w+)"'
            score_match = re.search(score_pattern, content)
            if score_match:
                score = score_match.group(1)
            else:
                print("Was unable to parse scores from following response: \n\n")
                print("The generated response is ", content)
                print("The correct label is : ", row['LABEL'])
                not_matched+=1
    
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
    
    print(f"Correct examples: {total_accuracy}   Accuracy: {total_accuracy/len(test_ds)}")
    if count_pass>0:
        print(f"Correct PASS examples: {accuracy_pass}   PASS Accuracy: {accuracy_pass/count_pass}")
    if count_fail>0:
        print(f"Correct FAIL examples: {accuracy_pass}   FAIL Accuracy: {accuracy_fail/count_fail}")

    print(f"Was unable to parse content for {not_matched} rows")
    
    test_df = test_ds.to_pandas()
    test_df['generated_text'] = responses
    test_df['reasoning'] = reasonings
    test_df['score'] = scores

    dataset = Dataset.from_pandas(test_df)
    dataset.push_to_hub(f"{output_path}")
    print(f"Saved dataset to HF Hub : {output_path}")


def main():
    parser = ArgumentParser(
        description='Load a HF model and generate responses',
    )

    parser.add_argument("--model_name_or_path", type=str, help="HF model name or path")
    parser.add_argument("--test_ds_path", type=str, help="HF test dataset path")
    parser.add_argument("--max_new_tokens", type=int, help="Max new tokens")
    parser.add_argument("--output_path", type=str, help="Output path should be in form 'username/dataset_name'")
    args = parser.parse_args()

    evaluate(args.model_name_or_path, args.test_ds_path, args.max_new_tokens, args.output_path)

if __name__ == "__main__":
    main()

# python3 hf_inference.py --model_name_or_path microsoft/phi-1_5 --test_ds_path sunitha-ravi/financebench_perturb_labels --max_new_tokens 10 --output_path sunitha-ravi/dummy-results

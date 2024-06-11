import ast
import torch
import json
import os
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from datasets import Dataset, load_dataset
from tqdm import tqdm
from argparse import ArgumentParser

from dotenv import load_dotenv

load_dotenv()

def evaluate(
    model_name_or_path,
    test_ds_path,
    max_new_tokens,
    output_path
):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
       model_name_or_path, device_map="auto"
    )

    print("Loaded the model!")

    dataset = load_dataset("sunitha-ravi/drop-answer-perturb")
    test_ds = dataset['test']['messages']
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model_name_or_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,

    )

    responses = []
    for row in tqdm(test_ds):
        messages = [row[0]]
        response = pipe(messages)[0]['generated_text'][-1]
        responses.append(response)
        print(response)

    test_df = pd.DataFrame({"message": test_ds})
    test_df['generated_text'] = responses
    dataset = Dataset.from_pandas(test_df)
    dataset.push_to_hub(f"sunitha-ravi/{output_path}")
    print("Saved dataset to HF!")


def main():
    parser = ArgumentParser(
        description='Load a HF model and generate responses',
    )

    parser.add_argument("--model_name_or_path", type=str, help="HF model name or path")
    parser.add_argument("--test_ds_path", type=str, help="HF test dataset path")
    parser.add_argument("--max_new_tokens", type=int, help="Max new tokens")
    parser.add_argument("--output_path", type=str, help="Output path")
    args = parser.parse_args()

    evaluate(args.model_name_or_path, args.test_ds_path, args.max_new_tokens, args.output_path)

if __name__ == "__main__":
    main()

# python3 hf_inference.py --model_name_or_path microsoft/phi-1_5 --test_ds_path sunitha-ravi/financebench-answer-perturb --max_new_tokens 10 --output_path output.json
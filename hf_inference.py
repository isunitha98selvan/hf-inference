import ast
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm


def evaluate(
    model_name_or_path,
    test_ds_path,
    max_new_tokens
):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
       model_name_or_path, device_map="auto"
    )

    test_dataset = load_dataset(test_ds_path, split="test")

    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model_name_or_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,

    )

    responses = []
    for row in test_ds:
        messages = [row[0]]
        response = pipe(messages)[0]['generated_text'][-1]
        responses.append(response)
        print(response)

def main():
    parser = ArgumentParser(
        description='Load a HF model and generate responses',
    )

    parser.add_argument("--model_name_or_path", type=str, help="HF model name or path")
    parser.add_argument("--test_ds_path", type=str, help="HF test dataset path")
    parser.add_argument("--max_new_tokens", type=int, help="Max new tokens")
    args = parser.parse_args()

    evaluate(args.model_name_or_path, args.test_ds_path, args.max_new_tokens)

if __name__ == "__main__":
    main()
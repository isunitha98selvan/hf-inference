from openai import OpenAI
from argparse import ArgumentParser


parser = ArgumentParser(
        description="Load a HF model and generate responses",
    )

parser.add_argument("--model_name_or_path", type=str, help="HF model name or path")
parser.add_argument(
    "--api_key",
    type=str,
    help="OpenAI api key",
)

args = parser.parse_args()

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=args.api_key,
)

completion = client.chat.completions.create(
  model=args.model_name_or_path,
  messages=[
    {"role": "user", "content": "Hello! Please generate some text!"}
  ]
)

print(completion.choices[0].message)

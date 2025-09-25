import argparse
from pathlib import Path

from langextract import io, providers
from pipeline import extract_policy

input_path = Path("../inputs")
output_path = Path("../outputs/extract")


def main(args):
    policy_file = input_path / f"{args.file}.txt"
    output_name = f"{args.file}.jsonl"
    policy = policy_file.read_text(encoding="utf-8")

    providers.load_builtins_once()

    result = extract_policy(policy)

    io.save_annotated_documents(
        [result], output_name=output_name, output_dir=output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="policy1")
    args = parser.parse_args()
    main(args)

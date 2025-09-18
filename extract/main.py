from pathlib import Path

from langextract import io, providers
from pipeline import extract_policy

if __name__ == "__main__":
    file = Path("../inputs/policy1.txt")
    policy = file.read_text(encoding="utf-8")

    providers.load_builtins_once()

    result = extract_policy(policy)

    io.save_annotated_documents(
        [result], output_name="extraction_results.jsonl", output_dir="../outputs"
    )

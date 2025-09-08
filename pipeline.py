# pipeline.py
from langextract import data, extract

from config import (
    MAX_WORKERS,
    PROMPT_DESCRIPTION,
    model_configs,
)
from examples import examples


def extract_policy(text: str):
    try:
        # 调用 extract，保留 debug=True 查看原始输出
        annotation = extract(
            text_or_documents=text,
            prompt_description=PROMPT_DESCRIPTION,
            examples=examples,  # 你之前定义的 example
            config=model_configs["qwen-turbo"],
            max_workers=MAX_WORKERS,
            format_type=data.FormatType.JSON,
            debug=True,
        )
        print("=== Raw output from lx.extract ===")
        print(annotation)
        return annotation
    except Exception as e:
        print("=== Exception occurred ===")
        print(e)
        return None

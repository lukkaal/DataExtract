# pipeline.py
import langextract as lx
from config import MODEL_ID, MAX_WORKERS, DEBUG, REQUIRED_KEYS, PROMPT_DESCRIPTION, model_config
from examples import examples
from normalizers import normalize_date, parse_amount, normalize_contact
import re
from typing import Any, Dict

def extract_policy(text: str):
    try:
        # 调用 extract，保留 debug=True 查看原始输出
        annotation = lx.extract(
            text_or_documents=text,
            prompt_description=PROMPT_DESCRIPTION,
            examples=examples,  # 你之前定义的 example
            model_id=MODEL_ID,
            config=model_config,
            max_workers=MAX_WORKERS,
            debug=True,
        )
        print("=== Raw output from lx.extract ===")
        print(annotation)
        return annotation
    except Exception as e:
        print("=== Exception occurred ===")
        print(e)
        return None
import os
from openai import OpenAI
from prompt import POLICY_PARSE_PROMPT  # 引入规则Prompt
import json

# 初始化 Qwen API 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def main():
    # 从当前文件夹读取政策原文
    input_file = "policy_input.txt"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到 {input_file} 文件，请在当前目录下创建该文件并填入政策原文。")

    with open(input_file, "r", encoding="utf-8") as f:
        policy_text = f.read()

    # 调用 Qwen 模型进行解析
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": POLICY_PARSE_PROMPT},
            {"role": "user", "content": policy_text},
        ],
        extra_body={"enable_search": True}
    )

    # 模型返回结果
    result = completion.choices[0].message.content.strip()

    # 转换为 JSON
    try:
        parsed_result = json.loads(result)
        if isinstance(parsed_result, str):
            parsed_result = json.loads(parsed_result)
    except json.JSONDecodeError:
        print("⚠️ 模型输出不是严格 JSON，已原样保存。")
        parsed_result = {"raw_output": result}

    # 保存为规整的 JSON 文件
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(parsed_result, f, ensure_ascii=False, indent=2)

    print("解析结果已保存到 output.json")

if __name__ == "__main__":
    main()

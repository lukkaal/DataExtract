import json
import langextract as lx

# ---------- 初始化 ----------
lx.providers.load_builtins_once()

# 查看可用提供商
providers = lx.providers.router.list_providers()
print("可用提供商:", providers)

# ---------- 配置模型 ----------
model_config = lx.factory.ModelConfig(
    model_id="gpt-oss",               # 与 vLLM 部署的模型名称一致
    provider="openai",                 # 强制使用 OpenAI 兼容 provider
    provider_kwargs={
        "base_url": "http://192.168.31.127:19090/v1",  # vLLM API 地址
        "api_key": "gpustack_d402860477878812_9ec494a501497d25b565987754f4db8c",
        "model_id": "gpt-oss"
    }
)

# ---------- 定义示例，用于模型对齐 ----------
examples = [
    lx.data.ExampleData(
        text="王小明，25岁，是一名软件开发者，隶属于A公司，职称工程师。",
        extractions=[
            lx.data.Extraction(
                extraction_class="person",
                extraction_text="王小明",
                attributes={
                    "age": "25",
                    "occupation": "软件开发者",
                    "company": "A公司",
                    "title": "工程师"
                }
            )
        ]
    ),
    lx.data.ExampleData(
        text="李华，30岁，担任项目经理，工作于B科技。",
        extractions=[
            lx.data.Extraction(
                extraction_class="person",
                extraction_text="李华",
                attributes={
                    "age": "30",
                    "occupation": "项目经理",
                    "company": "B科技",
                    "title": None
                }
            )
        ]
    )
]

# ---------- 定义实际输入文本 ----------
input_text = """
张华，今年32岁，是一位经验丰富的工程师，供职于C科技公司，职称高级工程师。
李明，28岁，是一位充满活力的设计师，供职于D设计院，职称设计师。
王刚，35岁，担任产品经理，工作于E互联网公司，职称经理。
赵丽，29岁，是数据分析师，隶属于F数据科技，职称分析师。
陈晨，40岁，是公司 CTO，供职于G科技集团。
"""

# ---------- 执行信息提取 ----------
try:
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description="从文本中提取人物姓名、年龄、职业、单位和职称信息",
        examples=examples,
        config=model_config,
        fence_output=True,
        use_schema_constraints=False
    )

    # ---------- 打印结果 ----------
    print("提取结果：")
    print(f"文档ID: {result.document_id}")
    print(f"提取数量: {len(result.extractions)}")
    print("-" * 50)

    extracted_data = []
    for i, extraction in enumerate(result.extractions, 1):
        print(f"提取 {i}:")
        print(f"  类别: {extraction.extraction_class}")
        print(f"  文本: {extraction.extraction_text}")
        print(f"  位置: {extraction.char_interval}")
        print(f"  属性: {extraction.attributes}")
        print(f"  状态: {extraction.alignment_status}")
        print("-" * 50)
        extracted_data.append({
            "class": extraction.extraction_class,
            "text": extraction.extraction_text,
            "interval": extraction.char_interval,
            "attributes": extraction.attributes,
            "status": extraction.alignment_status
        })

    # ---------- 保存为 JSON ----------
    # output_file = "extracted_result.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    # print(f"提取结果已保存到 {output_file}")

except Exception as e:
    print(f"错误: {e}")
    print("请检查:")
    print("1. vLLM 服务是否正在运行")
    print("2. 模型名称是否正确")
    print("3. API 地址和密钥是否有效")
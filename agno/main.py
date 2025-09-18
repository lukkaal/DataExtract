import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from schema import (
    ApplicationProcessSchema,
    ContactSchema,
    EligibilitySchema,
    FundingSchema,
    PolicyInfoSchema,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from agno.agent import Agent
from agno.models.openai import OpenAILike

load_dotenv()


model = OpenAILike(
    id="qwen-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)


agent = Agent(
    model=model,
    description="你是一个专业的政策文件信息提取助手。",
    use_json_mode=True,
    # debug_mode=True,
)


@retry(
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(ValueError),
)
def call_agent(content: str, output_schema: type[BaseModel]) -> BaseModel:
    agent.output_schema = output_schema
    output = agent.run(content).content

    if not isinstance(output, output_schema):
        raise ValueError("The output doesn't match output schema")

    return output


def main():
    policy_file = Path("../inputs/policy1.txt")
    result_file = Path("../outputs/agno-result.json")
    policy_content = policy_file.read_text(encoding="utf-8")

    output_schema_list = [
        PolicyInfoSchema,
        ApplicationProcessSchema,
        EligibilitySchema,
        FundingSchema,
        ContactSchema,
    ]

    policy_infos = {}

    for output_schema in output_schema_list:
        agent.output_schema = output_schema
        output = call_agent(policy_content, output_schema)
        print(output)
        policy_infos.update(output.model_dump())

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(policy_infos, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

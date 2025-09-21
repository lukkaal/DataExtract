import os

from pydantic import BaseModel

from agno.agent import Agent
from agno.models.openai import OpenAIChat, OpenAILike

model_configs = {
    "gpt-oss": OpenAIChat(
        id="gpt-oss",
        base_url="http://192.168.31.127:19090/v1",  # vLLM API 地址
        api_key=os.getenv("VLLM_API_KEY"),
    ),
    "qwen-turbo": OpenAILike(
        id="qwen-turbo",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    ),
    "qwen-plus": OpenAILike(
        id="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    ),
    "qwen-max": OpenAILike(
        id="qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    ),
}


def call_agent(
    agent: Agent,
    input: str,
    session_id: str | None = None,
    output_schema: type[BaseModel] | None = None,
):
    if output_schema:
        agent.output_schema = output_schema

    output = agent.run(input, session_id=session_id).content

    return output

# config.py
import os

from dotenv import load_dotenv
from langextract import factory

load_dotenv()

MAX_WORKERS = 4
DEBUG = False

# 注册模型
model_configs = {
    "gpt-oss": factory.ModelConfig(
        model_id="gpt-oss",
        provider="openai",
        provider_kwargs={
            "base_url": "http://192.168.31.127:19090/v1",  # vLLM API 地址
            "api_key": os.getenv("VLLM_API_KEY"),
        },
    ),
    "qwen-turbo": factory.ModelConfig(
        model_id="qwen-turbo",
        provider="openai",
        provider_kwargs={
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
        },
    ),
    "qwen-plus": factory.ModelConfig(
        model_id="qwen-plus",
        provider="openai",
        provider_kwargs={
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
        },
    ),
    "qwen-max": factory.ModelConfig(
        model_id="qwen-max",
        provider="openai",
        provider_kwargs={
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
        },
    ),
}

# 统一字段定义
REQUIRED_KEYS = [
    "policy_title",
    "issuing_department",
    "publish_date",
    "effective_date",
    "legal_basis",
    "applicability_scope",
    "eligibility_criteria",
    "application_deadline",
    "submission_method",
    "materials_required",
    "funding_amount_total",
    "funding_limit_per_project",
    "funding_source",
    "disbursement_schedule",
    "evaluation_process",
    "contact_info",
    "source_url",
    # 次要
    "project_duration",
    "number_of_awards",
    "restrictions_and_conditions",
    "attachments",
    "pilot_flag",
    "core_mechanism",
    "background_and_objectives",
    "required_team_composition",
    "special_requirements",
    "normalization_notes",
    # 低优先
    "confidence",
    "key_clauses",
    "impact_summary",
    "self_funding_ratio",
    "enforcement_and_penalties",
    "templates_and_forms",
    # metadata
    "metadata",
]

# Prompt 描述
# PROMPT_DESCRIPTION = """
# 任务：从输入的政策/公告/通知文本中抽取结构化字段，返回单个 JSON 对象。必须包含下列字段（即使未找到也要保留键并赋 null）...
# （省略，内容保持和你原始脚本里一致）
# """
PROMPT_DESCRIPTION = """
任务：从输入的政策/公告/通知文本中抽取结构化字段，返回单个 JSON 对象。
请严格输出 JSON，使用双引号，不要换行或包含注释。每个字段必须在 JSON 中闭合。
字段结构及示例可参考提供的 examples。必须保留所有字段，即使未找到也要保留并赋 null。
请把原始被抽取文本片段保存在 元数据.原始文本片段 中以便人工回溯。

归一化规则：
- 日期统一为 YYYY-MM-DD（含时间则使用完整 ISO 8601）。
- 金额统一为整数元并设置 币种 字段（CNY），同时保留 原始文本 原始字符串。
- 多值字段用数组保持原文顺序。

输出要求：
- 仅返回 JSON 对象，不要包含任何额外说明文本或代码块。
- JSON 字段名称必须与 examples 保持一致（中文字段）。
"""

# 同义词映射
SYNONYMS = {
    "发文单位": "issuing_department",
    "发文部门": "issuing_department",
    "发文日期": "publish_date",
    "发布日期": "publish_date",
    "实施日期": "effective_date",
    "生效日期": "effective_date",
    "申报方式": "submission_method",
    "申报材料": "materials_required",
    "附件": "attachments",
    "资助经费": "funding_amount_total",
}

# policy_langextract_pipeline.py
# 要求: pip install langextract
# 运行前: 配置 langextract provider/model（例如环境变量或 lx.init(...)）
import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

import langextract as lx  # 假设已正确安装并配置 provider

# ---------- 配置 ----------
MODEL_ID = None  # 若需指定，填入你的 model id，例如 "gpt-4o" 或 langextract provider id
MAX_WORKERS = 4
DEBUG = False

# ---------- 目标字段（确保输出都存在） ----------
REQUIRED_KEYS = [
    "policy_title", "issuing_department", "publish_date", "effective_date",
    "legal_basis", "applicability_scope", "eligibility_criteria",
    "application_deadline", "submission_method", "materials_required",
    "funding_amount_total", "funding_limit_per_project", "funding_source",
    "disbursement_schedule", "evaluation_process", "contact_info", "source_url",
    # 次要/可选字段
    "project_duration", "number_of_awards", "restrictions_and_conditions",
    "attachments", "pilot_flag", "core_mechanism", "background_and_objectives",
    "required_team_composition", "special_requirements", "normalization_notes",
    # 低优先
    "confidence", "key_clauses", "impact_summary", "self_funding_ratio",
    "enforcement_and_penalties", "templates_and_forms",
    # metadata 用于保存 raw spans
    "metadata"
]

# ---------- Prompt 描述（中文，遵循你给的 schema） ----------
prompt_description = """
任务：从输入的政策/公告/通知文本中抽取结构化字段，返回单个 JSON 对象。必须包含下列字段（即使未找到也要保留键并赋 null）：
policy_title, issuing_department (数组或字符串), publish_date (YYYY-MM-DD),
effective_date (YYYY-MM-DD 或 "YYYY-MM-DD to YYYY-MM-DD"),
legal_basis, applicability_scope, eligibility_criteria, application_deadline (ISO 8601 当包含时分),
submission_method, materials_required (数组), funding_amount_total ({amount,currency,raw_text}),
funding_limit_per_project ({amount,currency,raw_text}), funding_source, disbursement_schedule,
evaluation_process, contact_info (数组: {department, phone[], email[], note}), source_url,
以及若干次要字段 (project_duration, number_of_awards, ...)。请把原始被抽取文本片段保存在 metadata.raw_text_spans 中以便人工回溯。
归一化规则：
- 日期统一为 YYYY-MM-DD（含时间则使用完整 ISO 8601）。
- 金额统一为整数元并设置 currency 字段（CNY），同时保留 raw_text 原始字符串。
- 多值字段用数组保持原文顺序。
输出仅为 JSON 对象，不要输出任何额外说明文本或代码块。
"""

# ---------- Few-shot 示例（示例要尽量覆盖常见变体） ----------
examples = [
    lx.data.ExampleData(
        text=(
            "关于支持人工智能试点示范项目的通知\n"
            "财政部 科技部 联合发布\n"
            "文件编号：财教〔2024〕88号\n"
            "发布日期：2024年12月31日\n"
            "实施日期：2025年01月01\n"
            "资助总额：合计人民币5000万元\n"
            "单项目资助上限：50万元\n"
            "适用范围：全国高校与科研机构\n"
            "申报方式：通过国家科研管理平台（https://apply.example.gov.cn）在线申报\n"
            "申报材料：立项申请书；单位证明；成果说明；预算表\n"
            "联系人：张三，电话 010-12345678，邮箱 zhangsan@moe.cn\n"
        ),
        extraction=lx.data.Extraction(
            fields={
                "policy_title": "关于支持人工智能试点示范项目的通知",
                "issuing_department": ["财政部", "科技部"],
                "publish_date": "2024-12-31",
                "effective_date": "2025-01-01",
                "legal_basis": "财教〔2024〕88号",
                "applicability_scope": "全国高校与科研机构",
                "eligibility_criteria": None,
                "application_deadline": None,
                "submission_method": "国家科研管理平台 https://apply.example.gov.cn 在线申报",
                "materials_required": ["立项申请书", "单位证明", "成果说明", "预算表"],
                "funding_amount_total": {"amount": 50000000, "currency": "CNY", "raw_text": "合计人民币5000万元"},
                "funding_limit_per_project": {"amount": 500000, "currency": "CNY", "raw_text": "50万元"},
                "funding_source": None,
                "disbursement_schedule": None,
                "evaluation_process": None,
                "contact_info": [{"department": None, "phone": ["010-12345678"], "email": ["zhangsan@moe.cn"], "note": "联系人: 张三"}],
                "source_url": None,
                "metadata": {"raw_text_spans": {}}
            }
        )
    ),
    lx.data.ExampleData(
        text=(
            "关于“高水平科研团队建设”项目的实施细则（试点）\n"
            "广州市人民政府印发\n"
            "发文号：穗府规〔2024〕5号\n"
            "发布日期：2024年06月01\n"
            "实施期限：2024-06-01 至 2025-05-31\n"
            "资助总额：2亿元人民币\n"
            "拟立项数量：每年度不超过20项\n"
            "申报方式：线下提交纸质材料至市科技局窗口\n"
            "联系人：科技局，电话：020-87654321；邮箱：kjj@gz.gov.cn\n"
        ),
        extraction=lx.data.Extraction(
            fields={
                "policy_title": "关于“高水平科研团队建设”项目的实施细则（试点）",
                "issuing_department": ["广州市人民政府"],
                "publish_date": "2024-06-01",
                "effective_date": "2024-06-01 to 2025-05-31",
                "legal_basis": "穗府规〔2024〕5号",
                "applicability_scope": "广州市范围内符合条件的团队",
                "eligibility_criteria": None,
                "application_deadline": None,
                "submission_method": "线下提交纸质材料至市科技局窗口",
                "materials_required": None,
                "funding_amount_total": {"amount": 200000000, "currency": "CNY", "raw_text": "2亿元人民币"},
                "funding_limit_per_project": None,
                "funding_source": None,
                "disbursement_schedule": None,
                "evaluation_process": None,
                "contact_info": [{"department": "市科技局", "phone": ["020-87654321"], "email": ["kjj@gz.gov.cn"], "note": ""}],
                "source_url": None,
                "number_of_awards": "每年度不超过20项",
                "pilot_flag": True,
                "metadata": {"raw_text_spans": {}}
            }
        )
    ),
    lx.data.ExampleData(
        text=(
            "国家自然科学基金委员会关于青年科学基金的公告\n"
            "发布日期：2023-03-15\n"
            "申报截止：2023年05月15日 17:00\n"
            "申报方式：在线提交至 https://nsfc.example.org\n"
            "联系人：李四 13800138000\n"
        ),
        extraction=lx.data.Extraction(
            fields={
                "policy_title": "国家自然科学基金委员会关于青年科学基金的公告",
                "issuing_department": ["国家自然科学基金委员会"],
                "publish_date": "2023-03-15",
                "effective_date": None,
                "legal_basis": None,
                "applicability_scope": None,
                "eligibility_criteria": None,
                "application_deadline": "2023-05-15T17:00:00+08:00",
                "submission_method": "在线提交 https://nsfc.example.org",
                "materials_required": None,
                "funding_amount_total": None,
                "funding_limit_per_project": None,
                "funding_source": None,
                "disbursement_schedule": None,
                "evaluation_process": None,
                "contact_info": [{"department": None, "phone": ["13800138000"], "email": [], "note": "联系人: 李四"}],
                "source_url": None,
                "metadata": {"raw_text_spans": {}}
            }
        )
    ),
    lx.data.ExampleData(
        text=(
            "科技创新专项支持（示例）\n"
            "发布单位：某省科技厅\n"
            "资金来源：省级财政专项\n"
            "自筹比例：30%\n"
            "附件：项目申请表（附件1），预算模板（附件2）\n"
        ),
        extraction=lx.data.Extraction(
            fields={
                "policy_title": "科技创新专项支持（示例）",
                "issuing_department": ["某省科技厅"],
                "publish_date": None,
                "effective_date": None,
                "legal_basis": None,
                "applicability_scope": None,
                "eligibility_criteria": None,
                "application_deadline": None,
                "submission_method": None,
                "materials_required": None,
                "funding_amount_total": None,
                "funding_limit_per_project": None,
                "funding_source": "省级财政专项",
                "disbursement_schedule": None,
                "evaluation_process": None,
                "contact_info": None,
                "source_url": None,
                "self_funding_ratio": "30%",
                "attachments": ["项目申请表（附件1）", "预算模板（附件2）"],
                "metadata": {"raw_text_spans": {}}
            }
        )
    )
]

# ---------- 辅助：归一化函数 ----------
def normalize_date(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    # 处理区间（常见“至/到/to/-”）
    s = s.replace("－", "-").replace("—", "-").replace("–", "-")
    for sep in [" 至 ", " 至", "到", "到 ", " to ", " - ", " -", " - "]:
        if sep in s and any(ch.isdigit() for ch in s):
            parts = re.split(r'\s*(?:至|到|to|-)\s*', s)
            if len(parts) >= 2:
                a = normalize_date(parts[0])
                b = normalize_date(parts[1])
                if a or b:
                    return f"{a or ''} to {b or ''}"
    # yyyy年mm月dd日 或 yyyy-mm-dd 或 yyyy/mm/dd
    m = re.search(r'(\d{4})[年\-\/\.](\d{1,2})[月\-\/\.](\d{1,2})', s)
    if m:
        y, mo, d = m.groups()
        try:
            return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
        except:
            pass
    # yyyy年mm月
    m2 = re.search(r'(\d{4})[年\-\/\.](\d{1,2})[月]?', s)
    if m2:
        y, mo = m2.groups()
        try:
            return datetime(int(y), int(mo), 1).strftime("%Y-%m-%d")
        except:
            pass
    # ISO 日期时间
    try:
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y-%m-%d")
    except:
        pass
    # 仅年份
    m3 = re.search(r'(\d{4})', s)
    if m3:
        return f"{m3.group(1)}-01-01"
    return None

def parse_amount(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.replace(",", "").strip()
    # 提取第一个数字（可能有小数）
    num_m = re.search(r'(\d+(\.\d+)?)', s)
    if not num_m:
        return {"amount": None, "currency": None, "raw_text": raw}
    num = float(num_m.group(1))
    # 检测单位
    if re.search(r'亿', s):
        value = int(num * 1e8)
    elif re.search(r'万', s):
        value = int(num * 1e4)
    elif re.search(r'元', s) or re.search(r'￥', s) or re.search(r'CNY', s, re.I):
        value = int(num)
    else:
        # 无单位，尝试假定为元
        value = int(num)
    return {"amount": int(value), "currency": "CNY", "raw_text": raw}

def extract_phones(text: str) -> List[str]:
    if not text:
        return []
    phones = set()
    # 常见手机号/座机/带区号
    for m in re.finditer(r'(?:(?:\+?86[\s\-]?)?)((?:1\d{10})|(?:0\d{2,3}\-?\d{7,8}))', text):
        phones.add(m.group(1))
    return list(phones)

def extract_emails(text: str) -> List[str]:
    if not text:
        return []
    return list(set(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)))

def normalize_contact(raw_text: Optional[str]) -> List[Dict[str, Any]]:
    if not raw_text:
        return []
    # 可能包含多条联系人信息，用换行或分号分割
    items = re.split(r'[；;\n\r]+', raw_text)
    result = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        phones = extract_phones(it)
        emails = extract_emails(it)
        # 尝试抽出部门或姓名（非常简单的启发式）
        dept = None
        note = it
        # 如果前面有"联系人："或"联系人"等
        m = re.search(r'^(?:联系人[:：]?)\s*([^,，;；]+)', it)
        if m:
            name = m.group(1).strip()
            note = it
            result.append({"department": None, "phone": phones or [], "email": emails or [], "note": note})
            continue
        # 否则若包含"局"、"科"、"处"等可能是部门
        m2 = re.search(r'([^\s,，;；]{2,40}?(局|厅|部门|科|处|委员会|中心|学院|办公室))', it)
        if m2:
            dept = m2.group(1)
        result.append({"department": dept, "phone": phones or [], "email": emails or [], "note": note})
    return result

# 同义词映射（用户给的常见同义词）
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
    "资助经费": "funding_amount_total"
}

# ---------- 主抽取函数 ----------
def extract_policy(text: str) -> Dict[str, Any]:
    # 调用 langextract
    annotation = lx.extract(
        text_or_documents=text,
        prompt_description=prompt_description,
        examples=examples,
        model_id=MODEL_ID,
        max_workers=MAX_WORKERS,
        debug=DEBUG,
    )
    # langextract 可能返回注释对象或列表；这里尝试兼容
    if isinstance(annotation, dict):
        extracted = annotation.get("extraction", annotation)
    else:
        try:
            # 若是 iterable
            extracted = next(iter(annotation))
            if isinstance(extracted, dict) and "extraction" in extracted:
                extracted = extracted["extraction"]
        except Exception:
            extracted = annotation

    # 将提取的字段取为 dict（若是 lx.data.Extraction 结构）
    if hasattr(extracted, "fields"):
        doc = extracted.fields
    elif isinstance(extracted, dict):
        doc = extracted
    else:
        # fallback：转换为 dict（保守处理）
        doc = dict(extracted)

    # 后处理与归一化
    out = {}
    for k in REQUIRED_KEYS:
        v = doc.get(k, None)
        # 处理常见字段
        if k in ("publish_date", "application_deadline"):
            out[k] = None
            if v:
                normalized = normalize_date(str(v))
                out[k] = normalized
        elif k == "effective_date":
            if v:
                # 允许区间
                if isinstance(v, str) and ("to" in v or "至" in v or "-" in v):
                    out[k] = normalize_date(str(v))
                else:
                    out[k] = normalize_date(str(v))
            else:
                out[k] = None
        elif k in ("funding_amount_total", "funding_limit_per_project"):
            if isinstance(v, dict) and v.get("amount") is not None:
                out[k] = v
            else:
                out[k] = parse_amount(v) if v else None
        elif k == "materials_required" or k == "attachments":
            if isinstance(v, list):
                out[k] = v
            elif isinstance(v, str):
                # 尝试按分隔符切分
                parts = re.split(r'[；;,\n]|、', v)
                out[k] = [p.strip() for p in parts if p.strip()]
            else:
                out[k] = None
        elif k == "issuing_department":
            if isinstance(v, list):
                out[k] = v
            elif isinstance(v, str):
                # 简单按逗号/、切分联合发文
                parts = re.split(r'[、,，；;]', v)
                out[k] = [p.strip() for p in parts if p.strip()]
            else:
                out[k] = None
        elif k == "contact_info":
            if isinstance(v, list):
                out[k] = v
            elif isinstance(v, str):
                out[k] = normalize_contact(v)
            else:
                out[k] = None
        elif k == "policy_title":
            if isinstance(v, str):
                out[k] = " ".join(v.split())  # 去多余空白
            else:
                out[k] = None
        elif k == "metadata":
            # metadata 原样保存，保证存在
            out[k] = v or {"raw_text_spans": {}}
        else:
            # 其他字段原样保留（若不存在则 null）
            out[k] = v if (v is not None and v != "") else None

    # 保证每个 key 必有（如果上面遗漏）
    for k in REQUIRED_KEYS:
        if k not in out:
            out[k] = None

    # 附加 raw_text_span: 尝试从 annotation 中取出原始片段（若 langextract 返回 grounding）
    try:
        # 不同版本返回结构不同，这里做保守尝试
        if isinstance(annotation, dict) and "annotations" in annotation:
            out["metadata"]["raw_text_spans"] = annotation.get("annotations")
    except Exception:
        pass

    return out

# ---------- 示例运行（替换为你的文本） ----------
if __name__ == "__main__":
    sample_text = """
    【示例公告】关于支持人工智能试点示范项目的通知
    发布部门：财政部、科技部
    发布日期：2024年12月31日
    实施日期：2025年1月1日
    文件编号：财教〔2024〕88号
    资金总额：合计人民币5000万元
    单项最高资助：50万元
    申报方式：通过国家科研管理平台(https://apply.example.gov.cn)在线申报
    申报材料：立项申请书；单位证明；成果说明；预算表
    联系人：张三，电话：010-12345678，邮箱：zhangsan@moe.cn
    原文链接：https://example.gov.cn/doc/2024_ai_notice.pdf
    """

    result = extract_policy(sample_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 保存为 JSONL 以便批量处理（示例）
    with open("policy_extraction_results.jsonl", "w", encoding="utf8") as fw:
        fw.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("保存：policy_extraction_results.jsonl")

# normalizers.py
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

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
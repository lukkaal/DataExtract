# Agno 方案文档

`agno/` 目录包含了政策信息提取方案的核心实现。该方案旨在使用大语言模型（LLMs）、智能体框架（[Agno](https://www.agno.com/)）以及验证机制自动从政策文档中提取结构化信息。

## 概述

该方案提供了一个从非结构化政策文档中提取结构化信息的框架。它采用模块化方法，不同的政策信息方面由专门的类提取，每个类都有自己的验证和规范化逻辑。

## 目录结构

```
agno/
├── main.py        # 程序入口点
├── extraction.py  # 政策提取模块
└── llm.py         # LLM 模型配置和 Agent 调用工具
```

## 核心组件

### 程序入口点 `main.py`

1. 从文本文件加载政策内容
2. 初始化不同政策信息类型的提取类
3. 运行每个类的提取过程
4. 组合结果并以 JSON 格式输出

### 信息提取模块`extraction.py`

#### 1. `Extraction` 基类

`Extraction` 类提供了一个提取结构化信息的框架，包括：

- `extract`: 核心提取方法，负责调用 Agent 从给定文本中提取结构化信息。该方法会将提取结果传递给验证方法，如果验证失败则根据错误信息重新调用 Agent 进行提取。

- `validate`: 验证方法，用于检验提取结果的准确性和格式正确性。不同的提取类会重写此方法以实现特定的验证逻辑，例如检查日期格式是否符合ISO8601标准或验证提取的文本是否与原文一致。

- `normalize`: 规范化方法，将验证通过的提取结果转换为统一的输出格式。此方法确保不同类别的提取信息能够以一致的结构进行输出。

#### 2. `Extraction` 派生类

```python
class BasicInformation(Extraction):
   """政策基本信息"""
   policy_title # 政策标题
   legal_basis # 政策文号
   issuing_department # 发文部门

class TemporalInformation(Extraction):
   """政策时间信息"""
   publish_date # 发布时间
   application_deadline # 截止时间
   effective_start_date # 实施开始时间
   effective_end_date # 实施结束时间

class EligibilityInformation(Extraction):
   """申请资格与条件信息"""
   applicability_scope # 适用范围
   eligibility_criteria # 申报资格

class ApplicationProcessInformation(Extraction):
   """申请流程信息"""
   submission_method # 申报方式
   materials_required # 申报材料
   evaluation_process # 评审方式

class FundingInformation(Extraction):
   """资金信息"""
   funding_amount_total # 资金总额
   funding_limit_per_project # 单项资助上限
   funding_source # 资金来源
   disbursement_schedule # 拨款方式

class ContactInformation(Extraction):
   """联系方式信息"""
   contact_methods # 联系方式（电话/邮件）
   contact_departments # 联系部门
   contact_notes # 联系说明
   contact_persons # 联系人员
```

### 模型配置/智能体调用模块 `llm.py`

- 不同 LLM 提供商(OpenAI/Qwen)的模型配置
- `call_agent` 函数用于执行带有指定输出格式的 Agent 运行

## 工作流程

1. 政策文档作为文本文件加载
2. 内容由专门的提取类处理
3. 每个类使用 LLM 提取结构化信息
4. 验证确保提取的信息与源文档匹配
5. 规范化标准化输出格式
6. 结果组合并保存为 JSON 文件

## 主要特性

- **验证**：每个提取类都实现验证以确保准确性
- **重试机制**：失败的提取会自动重试并提供错误反馈
- **模块化设计**：易于扩展新的提取类
- **结构化输出**：所有提取信息的一致 JSON 输出格式
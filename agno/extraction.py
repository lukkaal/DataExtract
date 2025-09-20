import os
import re

from dateutil.parser import isoparse
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from agno.agent import Agent
from agno.models.openai import OpenAILike


class ExtractionError(Exception):
    pass


model = OpenAILike(
    id="qwen-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)


@retry(
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(ValueError),
)
def call_agent(agent: Agent, input: str, output_schema: type[BaseModel] | None = None):
    if output_schema:
        agent.output_schema = output_schema
    else:
        output_schema = agent.output_schema

    output = agent.run(input).content

    if not isinstance(output, output_schema):
        raise ValueError("The output doesn't match output schema")

    return output


class Extraction:
    policy_content: str
    extract_result: dict
    agent: Agent
    output_schema = type[BaseModel]
    retry_message: str
    retry_count: int

    def __init__(
        self,
        policy_content: str,
        output_schema: type[BaseModel],
        description: str | None = None,
    ):
        self.policy_content = policy_content
        self.output_schema = output_schema
        self.extract_result = {}
        self.retry_message = ""
        self.retry_count = 0
        self.agent = Agent(
            model=model,
            description=description or "你是一个专业的政策文件信息提取助手",
            use_json_mode=True,
            add_history_to_context=True,
            output_schema=output_schema,
            # debug_mode=True,
        )

    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(ExtractionError),
        before_sleep=lambda rs: print(
            f"Extract Retry: {rs.attempt_number}: {rs.outcome.exception()}"
        ),
    )
    def extract(self) -> dict:
        # Call agent to extract structured information
        output = call_agent(
            agent=self.agent,
            input=(self.policy_content if not self.retry_count else self.retry_message),
        )
        # Validate the output that the agent generate
        try:
            output = self.validate(output)
            print(output)
        except ExtractionError as e:
            self.retry_count += 1
            self.retry_message = str(e)
            print(f"Validate policy information output error:{e}")
            raise e

        # Normalize the output
        output = self.normalize(output)

        return output

    def validate(self, output: BaseModel) -> BaseModel:
        return output

    def normalize(self, output: BaseModel) -> dict:
        return output.model_dump()


class BasicInformation(Extraction):
    class OutputSchema(BaseModel):
        policy_title: str | None = Field(
            description="政策标题：公告/文件的正式名称",
        )
        legal_basis: str | None = Field(
            description="政策文号：引用的上位文件或编号（如“xxx〔20xx〕x号”）",
        )
        issuing_department: list[str] | None = Field(
            description="发文部门：主发/联合发文单位",
        )

        def __str__(self):
            return (
                f"政策标题：{self.policy_title}\n"
                f"政策文号：{self.legal_basis}\n"
                f"发文部门：{self.issuing_department}\n"
            )

    def __init__(self, policy_content: str):
        super().__init__(
            policy_content=policy_content,
            output_schema=BasicInformation.OutputSchema,
        )

    def _is_text_in_input(self, text: str):
        """严格的字符串匹配"""
        clean_text = re.sub(r"\s+", "", text)
        clean_policy = re.sub(r"\s+", "", self.policy_content)
        return clean_text in clean_policy

    def validate(self, output: OutputSchema) -> OutputSchema:
        policy_title = output.policy_title
        legal_basis = output.legal_basis
        issuing_department = output.issuing_department

        retry_messages = []

        if policy_title and not self._is_text_in_input(policy_title):
            retry_messages.append(
                f"你提取的政策标题'{policy_title}'与原文不一致，"
                "请从原文中直接提取政策标题，不得进行任何修改"
            )

        if legal_basis and not self._is_text_in_input(legal_basis):
            retry_messages.append(
                f"你提取的政策文号'{legal_basis}'与原文不一致，"
                "请从原文中直接提政策文号，不得进行任何修改"
            )

        if issuing_department:
            for department in issuing_department:
                if not self._is_text_in_input(department):
                    retry_messages.append(
                        f"你提取的相关部门'{department}'与原文不一致或不存在，"
                        "请从原文中直接提取相关部门，不得进行任何修改"
                    )

        if retry_messages:
            raise ExtractionError("\n".join(retry_messages))

        return output


class TemporalInformation(Extraction):
    class OutputSchema(BaseModel):
        publish_date: str | None = Field(
            description="发布时间：文档发布时间",
        )
        application_deadline: str | None = Field(
            description="截止时间：申报/办理截止时间",
        )
        effective_start_date: str | None = Field(
            description="实施开始时间：政策开始生效/实施的时间",
        )
        effective_end_date: str | None = Field(
            description="实施结束时间：政策停止生效/实施时间",
        )

        def __str__(self):
            return (
                f"发文时间：{self.publish_date}\n"
                f"截止时间：{self.application_deadline}\n"
                f"实施时间：{self.effective_start_date}-{self.effective_end_date}\n"
            )

    def __init__(self, policy_content: str):
        super().__init__(
            policy_content=policy_content,
            output_schema=TemporalInformation.OutputSchema,
            description="从文件中提取相关时间信息,时间格式遵循ISO8601(20YY-MM-DDThh:mm:ssZ)",
        )

    def _check_data_format(self, date: str):
        try:
            isoparse(date)
            return True
        except ValueError:
            return False

    def validate(self, output: OutputSchema) -> OutputSchema:
        publish_date = output.publish_date
        application_deadline = output.application_deadline
        effective_start_date = output.effective_start_date
        effective_end_date = output.effective_end_date

        retry_messages = []

        if publish_date and not self._check_data_format(publish_date):
            retry_messages.append(
                f"`publish_date`'{publish_date}'不符合ISO8601格式要求"
            )

        if application_deadline and not self._check_data_format(application_deadline):
            retry_messages.append(
                f"`application_deadline`'{application_deadline}'不符合ISO8601格式要求"
            )

        if effective_start_date and not self._check_data_format(effective_start_date):
            retry_messages.append(
                f"`effective_start_date`'{effective_start_date}'不符合ISO8601格式要求"
            )

        if effective_end_date and not self._check_data_format(effective_end_date):
            retry_messages.append(
                f"`effective_end_date`'{effective_end_date}'不符合ISO8601格式要求"
            )

        if retry_messages:
            raise ExtractionError("\n".join(retry_messages))

        return output

    def normalize(self, output: OutputSchema) -> dict:
        return {
            "publish_date": output.publish_date,
            "application_deadline": output.application_deadline,
            "effective_date": output.effective_start_date
            + (f"/{output.effective_end_date}" if output.effective_end_date else ""),
        }


class EligibilityInformation(Extraction):
    class OutputSchema(BaseModel):
        applicability_scope: list[str] | None = Field(
            description="适用范围：适用单位/对象的简述",
        )
        eligibility_criteria: list[str] | None = Field(
            description="申报资格：可申报或受益的具体条件（职称、年龄、单位类型等）",
        )

        def __str__(self):
            return (
                f"适用范围：{self.applicability_scope}\n"
                f"申报资格：{self.eligibility_criteria}\n"
            )

    def __init__(self, policy_content: str):
        super().__init__(
            policy_content=policy_content,
            output_schema=EligibilityInformation.OutputSchema,
        )

    def validate(self, output: OutputSchema) -> OutputSchema:
        return output


class ApplicationProcessInformation(Extraction):
    class OutputSchema(BaseModel):
        submission_method: list[str] | None = Field(
            description="申报方式：申报/办理方式与平台，含平台名、URL、线下流程等",
        )
        materials_required: list[str] | None = Field(
            description="申报材料：必须上传或提交的材料",
        )
        evaluation_process: list[str] | None = Field(
            description="评审方式：评审立项方式，如网络评审、答辩、评分规则等",
        )

        def __str__(self):
            return (
                f"申报方式：{self.submission_method}\n"
                f"申报材料：{self.materials_required}\n"
                f"评审方式：{self.evaluation_process}\n"
            )

    def __init__(self, policy_content: str):
        super().__init__(
            policy_content=policy_content,
            output_schema=ApplicationProcessInformation.OutputSchema,
        )

    def validate(self, output: OutputSchema) -> OutputSchema:
        return output


class FundingInformation(Extraction):
    class OutputSchema(BaseModel):
        funding_amount_total: str | None = Field(
            description="资金总额：若披露，数值与币种分离（例如：100CNY）",
        )
        funding_limit_per_project: str | None = Field(
            description="单项资助上限：单项目最高资助金额（若适用）",
        )
        funding_source: str | None = Field(
            description="资金来源：资金来源描述",
        )
        disbursement_schedule: str | None = Field(
            description="拨款方式：资金如何拨付（一次性/分批/条件触发）",
        )

        def __str__(self):
            return (
                f"资金总额：{self.funding_amount_total}\n"
                f"单项资助上限：{self.funding_limit_per_project}\n"
                f"资金来源：{self.funding_source}\n"
                f"拨款方式：{self.disbursement_schedule}\n"
            )

    def __init__(self, policy_content: str):
        super().__init__(
            policy_content=policy_content,
            output_schema=FundingInformation.OutputSchema,
        )

    def validate(self, output: OutputSchema) -> OutputSchema:
        return output


class ContactInformation(Extraction):
    class OutputSchema(BaseModel):
        contact_methods: list[str] = Field(
            default=[],
            description="联系方式（电话/邮件）",
        )
        contact_departments: dict[str, str] = Field(
            default={},
            description="`contact_methods`中每个联系方式对应的部门, 如果没有填None",
        )
        contact_notes: dict[str, str | None] = Field(
            default={},
            description="`contact_methods`中每个联系方式对应的说明, 如果没有填None",
        )
        contact_persons: dict[str, str | None] = Field(
            default={},
            description="`contact_methods`中每个联系方式对应的负责人, 如果没有填None",
        )

        def __str__(self):
            contact_details = ""
            for contact in self.contact_methods:
                contact_details += (
                    f"{contact}: "
                    f"{self.contact_departments.get(contact, None)}, "
                    f"{self.contact_notes.get(contact, None)}, "
                    f"{self.contact_persons.get(contact, None)}\n"
                )

            return contact_details
            # return f"{self.contact_methods}"

    def __init__(self, policy_content: str):
        super().__init__(
            policy_content=policy_content,
            output_schema=ContactInformation.OutputSchema,
            # TODO: 拆解成两个步骤，第一步列出所有联系方式，第二步根据联系方式获取具体信息
            description=(
                "在`contact_methods`中列出文件中提及的所有联系方式(电话/邮件)\n"
                "在`contact_departments`中列出`contact_methods`中的这些联系方式对应的部门\n"
                "在`contact_persons`中列出`contact_methods`中的这些联系方式对应的负责人\n"
                "在`contact_notes`中列出`contact_methods`中的这些联系方式对应的说明\n"
            ),
        )

    def validate(self, output: OutputSchema) -> OutputSchema:
        contact_methods = output.contact_methods
        contact_departments = output.contact_departments
        contact_notes = output.contact_notes
        contact_persons = output.contact_persons

        retry_messages = []

        if contact_methods != contact_departments.keys():
            retry_messages.append(
                "`contact_departments`中列出的联系方式与`contact_methods`中列出的联系方式不匹配请重新整理"
            )
        if contact_methods != contact_notes.keys():
            retry_messages.append(
                "`contact_notes`中列出的联系方式与`contact_methods`中列出的联系方式不匹配请重新整理"
            )
        if contact_methods != contact_persons.keys():
            retry_messages.append(
                "`contact_persons`中列出的联系方式与`contact_methods`中列出的联系方式不匹配请重新整理"
            )

        return output

    def normalize(self, output: OutputSchema) -> dict:
        contacts = output.contact_methods
        departments = output.contact_departments
        notes = output.contact_notes
        persons = output.contact_persons

        return {
            "contact": {
                contact: {
                    "department": departments.get(contact),
                    "note": notes.get(contact),
                    "person": persons.get(contact),
                }
                for contact in contacts
            }
            if contacts
            else None
        }

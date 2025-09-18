from pydantic import BaseModel, Field


class PolicyInfoSchema(BaseModel):
    policy_title: str | None = Field(
        description="政策标题：公告/文件的正式名称，去除多余空白",
    )
    issuing_department: list[str] | None = Field(
        description="发布部门：主发/联合发文单位",
    )
    publish_date: str | None = Field(
        description="发布日期：文档发布时间，格式 YYYY-MM-DD",
    )
    effective_date: str | None = Field(
        description="实施日期：生效/实施日期或区间，ISO 日期或区间字符串",
    )
    legal_basis: str | None = Field(
        description="政策文号：引用的上位文件或编号（如“xxx〔20xx〕x号”）",
    )
    source_url: str | None = Field(
        description="原文链接：公告页面或文件下载链接",
    )

    def __str__(self):
        return (
            f"政策标题：{self.policy_title}\n"
            f"发文部门：{self.issuing_department}\n"
            f"发文时间：{self.publish_date}\n"
            f"实施日期：{self.effective_date}\n"
            f"政策依据：{self.legal_basis}\n"
            f"原文链接：{self.source_url}\n"
        )


class ApplicationProcessSchema(BaseModel):
    submission_method: str | None = Field(
        description="申报方式：申报/办理方式与平台，含平台名、URL、线下流程等",
    )
    materials_required: list[str] | None = Field(
        description="所需材料：必须上传或提交的材料（数组）",
    )
    evaluation_process: str | None = Field(
        description="评审资格：立项方式，如网络评审、答辩、评分规则等",
    )
    application_deadline: str | None = Field(
        description="截止时间：申报/办理截止时间，若有时分则用 ISO 8601",
    )

    def __str__(self):
        return (
            f"申报方式：{self.submission_method}\n"
            f"所需材料：{self.materials_required}\n"
            f"评审资格：{self.evaluation_process}\n"
            f"截止时间：{self.application_deadline}\n"
        )


class EligibilitySchema(BaseModel):
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


class FundingSchema(BaseModel):
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


class ContactSchema(BaseModel):
    contact_department: str | None = Field(
        description="联系部门",
    )
    contact_phone: str | None = Field(
        description="联系电话",
    )
    contact_email: str | None = Field(
        description="联系邮箱",
    )

    def __str__(self):
        return (
            f"联系部门：{self.contact_department}\n"
            f"联系电话：{self.contact_phone}\n"
            f"联系邮箱：{self.contact_email}\n"
        )

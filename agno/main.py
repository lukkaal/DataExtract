import json
from pathlib import Path

from dotenv import load_dotenv
from extraction import (
    ApplicationProcessInformation,
    BasicInformation,
    ContactInformation,
    EligibilityInformation,
    FundingInformation,
    TemporalInformation,
)

load_dotenv(dotenv_path="../.env")


def main():
    policy_file = Path("../inputs/policy1.txt")
    result_file = Path("../outputs/agno-result.json")
    policy_content = policy_file.read_text(encoding="utf-8")

    policy_infos = {}

    extraction_classes = [
        BasicInformation,
        TemporalInformation,
        EligibilityInformation,
        ApplicationProcessInformation,
        FundingInformation,
        ContactInformation,
    ]

    for extraction_class in extraction_classes:
        extraction = extraction_class(policy_content)
        output = extraction.extract()
        policy_infos.update(output)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(policy_infos, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

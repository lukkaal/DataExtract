## Setup
- 在终端中输入 `uv sync` 初始化python环境
- 在终端中输入 `source .venv/bin/activate` 进入虚拟环境
- 复制 `.env.template` 到 `.env`，并在`.env`中填写上相应的API-KEY
  >**Note**: 不要直接将API-KEY写到源代码中，会造成API-KEY泄漏!!!

## Extract
方案一：使用 `langextract` 从政策文件中提取信息
```bash
cd extract
python main.py
```
结果会被保存到 `outputs/` 中的 `extract-result.jsonl`
```bash
python visualize.py
```
`visualize.py` 会将结果 `extract-result.jsonl` 转换成 `visualization.html` 以可视化的形式展示


## Example
方案二：直接利用OpenAI SDK，通过设置Prompt的方式从政策文件中提取信息
```bash
cd example
python main.py
```
结果会被保存到 `outputs/` 中的  `example-result.json` 中

## Agno
方案三：在方案二的基础上利用`Agno`(一个Agent框架)按照信息类别分多次提取政策信息
```python
class BasicInformation(Extraction):
    """政策基础信息信息"""

class TemporalInformation(Extraction):
    """政策时间信息"""

class EligibilityInformation(Extraction):
    """申请资格与条件信息"""

class ApplicationProcessInformation(Extraction):
    """申请流程信息"""

class FundingInformation(Extraction):
    """资金信息"""

class ContactInformation(Extraction):
    """联系方式信息"""
```
```bash
cd agno
python main.py
```
结果会被保存到 `outputs/` 中的  `agno-result.json` 中

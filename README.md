## Setup
- 在终端中输入 `uv sync` 初始化python环境
- 在终端中输入 `source .venv/bin/activate` 进入虚拟环境
- 复制 `.env.template` 到 `.env`，并在`.env`中填写上相应的API-KEY
  >**Note**: 不要直接将API-KEY写到源代码中，会造成API-KEY泄漏!!!

## Extract
方案一：使用 `langextract` 从政策文件中提取信息，相关文档 [langextract.md](docs/langextract.md)
```bash
cd extract
# 使用`-f`/`--file`指定`inputs/`下的政策文件
python main.py --file=policy1
```
结果会被保存到 `outputs/extract/` 下
```bash
python visualize.py
```
`visualize.py` 会将结果转换成 `visualization.html` 以可视化的形式展示


## Example
方案二：直接利用OpenAI SDK，通过设置Prompt的方式从政策文件中提取信息，相关文档 [llm.md](docs/llm.md)
```bash
cd example
# 使用`-f`/`--file`指定`inputs/`下的政策文件
python main.py --file=policy1
```
结果会被保存到 `outputs/example` 下

## Agno
方案三：在方案二的基础上利用智能体框架 ([Agno](https://www.agno.com/)) 按照信息类别分多次提取政策信息，相关文档 [agno.md](docs/agno.md)。
```bash
cd agno
# 使用`-f`/`--file`指定`inputs/`下的政策文件
python main.py --file=policy1
```
结果会被保存到 `outputs/agno/` 下

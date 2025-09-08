import importlib
import sys

# 要检查的依赖包
packages = [
    "langextract",
    "litellm",
    "jinja2",   # 用于 HTML 可视化
    "pytest"    # 可选：单元测试
]

print("=== LangExtract 环境自检 ===\n")

for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "未知版本")
        print(f"✅ {pkg} 已安装 (版本: {version})")
    except ImportError:
        print(f"❌ {pkg} 未安装")

print("\n=== 功能测试 ===")

# 测试 LangExtract 数据结构
try:
    import langextract as lx
    ex = lx.data.Extraction(extraction_class="fruit", extraction_text="apple")
    print("✅ LangExtract 数据结构可用:", ex)
except Exception as e:
    print("❌ LangExtract 使用失败:", e)

# 测试 HTML 可视化（生成 test.html）
try:
    import langextract as lx
    res = lx.data.Result(text="I like banana.", extractions=[ex])
    res.to_html("test.html")
    print("✅ HTML 可视化正常，已生成 test.html")
except Exception as e:
    print("❌ HTML 可视化失败:", e)

print("\n=== 自检完成 ===")

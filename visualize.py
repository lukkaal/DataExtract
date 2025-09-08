from langextract import visualize

html_content = visualize("./outputs/extraction_results.jsonl")
with open("./outputs/visualization.html", "w") as f:
    if hasattr(html_content, "data"):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)

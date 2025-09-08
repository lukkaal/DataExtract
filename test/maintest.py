import requests
import json

base_url = "http://192.168.31.127:19090"
api_key = "gpustack_d402860477878812_9ec494a501497d25b565987754f4db8c"

prompt = "请用中文简单介绍张华和李明。"

url = f"{base_url}/v1/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "gpt-oss",
    "prompt": prompt,
    "max_tokens": 200,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    # OpenAI 格式通常在 choices[0].text
    print("模型返回：", result['choices'][0]['text'])
else:
    print(f"请求失败: {response.status_code} {response.text}")

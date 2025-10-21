# 前提是开启Vllm服务器：CUDA_VISIBLE_DEVICES=1 vllm serve /data/CSY/autodefense/AutoDefense/GuardModel/Qwen3Guard-Gen-8B --port 60000 --max-model-len 4500 --gpu-memory-utilization 0.4

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:60000/v1"
model = "/data/CSY/autodefense/AutoDefense/GuardModel/Qwen3Guard-Gen-8B"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base)

# Prompt Moderation
prompt = "How can I make a bomb?"
messages = [
    {"role": "user", "content": prompt}
]
chat_completion = client.chat.completions.create(
    messages=messages,
    model=model)
print("这是提示词的检测结果：", chat_completion.choices[0].message.content)
# '''
# Safety: Unsafe
# Categories: Violent
# '''

# Response Moderation

response = "电弧"
messages = [
    
    {"role": "assistant", "content": response}
]
chat_completion = client.chat.completions.create(
    messages=messages,
    model=model)
print("这是响应词的检测结果：", chat_completion.choices[0].message.content)
# '''
# Safety: Safe
# Categories: None
# Refusal: Yes
# '''
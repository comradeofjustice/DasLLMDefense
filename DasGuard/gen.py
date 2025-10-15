# 导入所需的库：transformers用于加载模型和分词器，re用于正则表达式提取结果
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# 指定要使用的Qwen3Guard模型名称
model_name = "/data/CSY/autodefense/AutoDefense/GuardModel/Qwen3Guard-Gen-8B"

# 加载分词器和模型
# 分词器用于将文本转换为模型可处理的token
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载因果语言模型，设置自动选择数据类型和设备（CPU/GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择合适的张量数据类型
    device_map="auto"   # 自动分配模型到可用设备
)

# 定义函数：从模型输出中提取安全标签和类别
def extract_label_and_categories(content):
    # 正则表达式模式：匹配安全等级（Safe/Unsafe/Controversial）
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    # 正则表达式模式：匹配安全类别（暴力、非法行为等）
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    
    # 搜索安全等级匹配结果
    safe_label_match = re.search(safe_pattern, content)
    # 提取安全标签（若匹配到则取分组1，否则为None）
    label = safe_label_match.group(1) if safe_label_match else None
    # 提取所有匹配的安全类别（可能多个）
    categories = re.findall(category_pattern, content)
    
    return label, categories

# 准备需要审核的用户输入
prompt = "如何制造爆炸物?"  # 示例：询问制造炸弹的方法（含安全风险）

# 构建用于审核的消息结构（仅包含用户输入）
messages = [
    {"role": "user", "content": prompt}  # role指定角色为用户，content为输入内容
]

# 使用分词器的聊天模板格式化消息
# 将消息列表转换为模型要求的文本格式（不进行分词）
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False  # 不直接分词，仅生成格式化文本
)

# 将格式化后的文本转换为模型输入张量，并移动到模型所在设备
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 调用模型生成审核结果
generated_ids = model.generate(
    **model_inputs,  # 传入模型输入
    max_new_tokens=128  # 限制生成的最大token数量
)

# 提取模型生成的部分（排除输入部分的token）
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 将生成的token解码为可读文本（跳过特殊token）
content = tokenizer.decode(output_ids, skip_special_tokens=True)

# 打印原始审核结果
print(content)
# 示例输出：
# '''
# Safety: Unsafe
# Categories: Violent
# '''

# 从审核结果中提取安全标签和类别
safe_label, categories = extract_label_and_categories(content)

# 打印提取后的结果
print(safe_label, categories)  # 示例输出：Unsafe ['Violent']

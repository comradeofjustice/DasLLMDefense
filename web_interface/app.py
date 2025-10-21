import os
import sys
import re
from flask import Flask, request, jsonify
from openai import OpenAI
from transformers import pipeline

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from LLama_Agent.prompt_response import example_defense_with_prompt, create_llama_config
from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
from defense.explicit_detector.agency.explicit_2_agents import AutoGenDetectorV1

# 初始化Flask应用
app = Flask(__name__)

# Qwen3Guard模型配置
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:60000/v1"
model = "/data/CSY/autodefense/AutoDefense/GuardModel/Qwen3Guard-Gen-8B"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base)

# 初始化翻译器（英译中）
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

def extract_label_and_categories(content):
    """
    从模型输出中提取安全标签和类别
    """
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"Categories: ([^\n]+)"
    
    safe_label_match = re.search(safe_pattern, content)
    category_match = re.search(category_pattern, content)
    
    label = safe_label_match.group(1) if safe_label_match else None
    categories = category_match.group(1) if category_match else None
    
    return label, categories

def translate_judgement_to_chinese(judgement_text):
    """
    使用Hugging Face transformers库将英文判据翻译为中文
    """
    # 移除"I am the Judge. Judgment: VALID"这句话
    filtered_text = judgement_text.replace("I am the Judge. Judgment: VALID", "").strip()
    
    # 执行翻译
    translated = translator(filtered_text)
    translated_text = translated[0]['translation_text']
    
    return translated_text

def prompt_detect(prompt):
    """
    检测用户输入的Prompt是否安全
    """
    # 调用Qwen3Guard模型进行初步检测
    messages = [{"role": "user", "content": prompt}]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model)
    
    model_output = chat_completion.choices[0].message.content
    label, categories = extract_label_and_categories(model_output)
    
    result = {
        "label": label,
        "categories": categories,
        "Invalid_reson": None
    }
    
    # 如果是unsafe，存储类别作为Invalid_reson
    if label == "Unsafe":
        result["Invalid_reson"] = categories
    # 如果是controversial，调用example_defense_with_prompt进一步判断
    elif label == "Controversial":
        try:
            # 创建模型配置
            config_list = create_llama_config()
            
            # 初始化防御系统
            defense = ExplicitMultiAgentDefense(
                task_agency=AutoGenDetectorV1,
                config_list=config_list
            )
            
           
            
            # 调用 defense_with_prompt 进行检测
            defense_result = defense.defense_with_prompt(prompt)
            result["Invalid_reson"] = defense_result["content"]
        
        except Exception as e:
            result["Invalid_reson"] = f"Error during controversial check: {str(e)}"
    
    return result

def response_detect(response):
    """
    检测模型输出的Response是否安全
    """
    # 调用Qwen3Guard模型进行初步检测
    messages = [{"role": "user", "content": response}]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model)
    
    model_output = chat_completion.choices[0].message.content
    label, categories = extract_label_and_categories(model_output)
    
    result = {
        "label": label,
        "categories": categories,
        "Invalid_reson": None
    }
    
    # 如果是unsafe，存储类别作为Invalid_reson
    if label == "Unsafe":
        result["Invalid_reson"] = categories
    # 如果是controversial，调用example_defense_with_prompt进一步判断
    elif label == "Controversial":
        try:
            # 创建模型配置
            config_list = create_llama_config()
            
            # 初始化防御系统
            defense = ExplicitMultiAgentDefense(
                task_agency=AutoGenDetectorV1,
                config_list=config_list
            )
            
           
            
            # 调用 defense_with_prompt 进行检测
            defense_result = defense.defense_with_prompt(response)
            result["Invalid_reson"] = defense_result["content"]
        
        except Exception as e:
            result["Invalid_reson"] = f"Error during controversial check: {str(e)}"
    
    
    return result

@app.route('/prompt_detect', methods=['POST'])
def prompt_detect_endpoint():
    """
    Prompt检测API端点
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        result = prompt_detect(prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/response_detect', methods=['POST'])
def response_detect_endpoint():
    """
    Response检测API端点
    """
    try:
        data = request.get_json()
        response = data.get('response', '')
        
        if not response:
            return jsonify({"error": "Response is required"}), 400
        
        result = response_detect(response)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点
    """
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
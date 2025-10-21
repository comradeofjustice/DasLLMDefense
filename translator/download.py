import os
from huggingface_hub import snapshot_download

def download_translator_model():
    """
    使用snapshot_download下载翻译模型到指定目录
    """
    # 设置环境变量使用国内镜像（可选，加速下载）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 模型名称
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    
    # 目标目录
    local_dir = "/data/CSY/autodefense/AutoDefense/translator"
    
    # 创建目标目录（如果不存在）
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"开始下载模型 {model_name}...")
    print(f"目标目录: {local_dir}")
    
    try:
        # 使用snapshot_download下载模型
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=True,          # 支持断点续传
            token=None,                    # 如果需要认证，可在此添加huggingface token
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # 可选：忽略不必要的文件
        )
        print("✅ 模型下载成功！")
        print(f"模型已保存到: {local_dir}")
        
        # 列出下载的文件
        downloaded_files = os.listdir(local_dir)
        print(f"下载的文件列表: {downloaded_files}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False
    
    return True

def test_model_loading():
    """
    测试模型是否能正常加载
    """
    try:
        from transformers import pipeline
        
        print("\n测试模型加载...")
        # 从本地目录加载模型
        translator = pipeline(
            "translation", 
            model="/data/CSY/autodefense/AutoDefense/translator"
        )
        
        # 测试翻译
        test_result = translator("Hello, how are you?", max_length=40)
        print(f"✅ 模型测试成功！")
        print(f"测试结果: {test_result}")
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")

if __name__ == "__main__":
    # 下载模型
    if download_translator_model():
        # 测试模型加载
        test_model_loading()
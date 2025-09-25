from doctest import debug
import os
import sys
import json
import tempfile
import logging
import io
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入防御评估函数
from evaluator.evaluate_helper import evaluate_defense_with_response, evaluate_defense_with_prompt, evaluate_defense_with_output_list
# 导入防御策略和工具函数
from defense.run_defense_exp import defense_strategies
from defense.utility import load_llm_config, load_defense_prompt, load_attack_template

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('defense_web_interface')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'json', 'txt'}

# 存储推理日志的内存缓存
inference_logs = {}

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 设置正确的 LLM 配置文件路径
LLM_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'config', 'llm_config_list.json')

# Tee类用于同时将输出写入到文件和控制台
class Tee(io.TextIOBase):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
    
    def write(self, data):
        # 将数据写入两个文件
        self.file1.write(data)
        self.file2.write(data)
        return len(data)
    
    def flush(self):
        # 刷新两个文件
        self.file1.flush()
        self.file2.flush()

# 修复 load_llm_config 函数以使用正确的路径
def get_llm_config(model_name="gpt-3.5-turbo"):
    return load_llm_config(json_path=LLM_CONFIG_PATH, model_name=model_name)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # 获取可用的防御策略
    strategies = [{'name': s['name'], 'task_agency': s['task_agency'].__name__, 'defense_agency': s['defense_agency'].__name__} for s in defense_strategies]
    # 获取可用的模型名称 - 使用修复后的函数
    llm_config = get_llm_config()
    models = list(set([config['model'] for config in llm_config]))
    return render_template('index.html', strategies=strategies, models=models)

@app.route('/defense/response', methods=['POST'])
def defense_response():
    try:
        # 创建内存缓冲区来捕获stdout和stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # 保存原始的stdout和stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # 重定向stdout和stderr到捕获缓冲区和原始输出
            sys.stdout = Tee(stdout_capture, original_stdout)
            sys.stderr = Tee(stderr_capture, original_stderr)
            
            # 获取表单数据
            strategy_name = request.form.get('strategy')
            model_name = request.form.get('model')
            response_text = request.form.get('response_text')
            
            # 查找选中的防御策略
            selected_strategy = next((s for s in defense_strategies if s['name'] == strategy_name), None)
            if not selected_strategy:
                return jsonify({'status': 'error', 'message': f'策略 {strategy_name} 未找到'})
            
            # 生成临时文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], f'defense_output_{timestamp}.json')
            
            # 创建临时聊天文件
            temp_chat_data = [{"name": "test", "raw_response": response_text}]
            temp_chat_file = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_chat_{timestamp}.json')
            with open(temp_chat_file, 'w') as f:
                json.dump(temp_chat_data, f)
            
            # 记录日志 ID
            log_id = f"{strategy_name}_{timestamp}"
            
            # 执行防御评估
            logger.info(f'Starting defense with response: strategy={strategy_name}, model={model_name}')
            evaluate_defense_with_response(
                task_agency=selected_strategy['task_agency'],
                defense_agency=selected_strategy['defense_agency'],
                defense_output_name=output_file,
                chat_file=temp_chat_file,
                model_name=model_name,
                parallel=False  # 单条响应不需要并行
                
            )
            
            # 读取防御结果
            with open(output_file, 'r') as f:
                defense_result = json.load(f)[0]
            
            # 存储推理日志
            inference_logs[log_id] = {
                'timestamp': timestamp,
                'strategy': strategy_name,
                'model': model_name,
                'input': response_text,
                'result': defense_result,
                'log_type': 'response_defense',
                'stdout_content': stdout_capture.getvalue(),
                'stderr_content': stderr_capture.getvalue()
            }
            
            # 清理临时文件
            os.remove(temp_chat_file)
            
            return jsonify({
                'status': 'success',
                'result': defense_result,
                'log_id': log_id,
                'stdout_content': stdout_capture.getvalue(),
                'stderr_content': stderr_capture.getvalue()
            })
        finally:
            # 恢复原始的stdout和stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    except Exception as e:
        logger.error(f'Error in defense_response: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/defense/prompt', methods=['POST'])
def defense_prompt():
    try:
        # 创建内存缓冲区来捕获stdout和stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # 保存原始的stdout和stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # 重定向stdout和stderr到捕获缓冲区和原始输出
            sys.stdout = Tee(stdout_capture, original_stdout)
            sys.stderr = Tee(stderr_capture, original_stderr)
            
            # 获取表单数据
            strategy_name = request.form.get('strategy')
            model_name = request.form.get('model')
            prompt_text = request.form.get('prompt_text')
            
            # 查找选中的防御策略
            selected_strategy = next((s for s in defense_strategies if s['name'] == strategy_name), None)
            if not selected_strategy:
                return jsonify({'status': 'error', 'message': f'策略 {strategy_name} 未找到'})
            
            # 记录日志 ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_id = f"{strategy_name}_{timestamp}"
            
            # 执行防御评估
            logger.info(f'Starting defense with prompt: strategy={strategy_name}, model={model_name}')
            
            # 由于原函数是批量处理所有harmful_prompt，我们需要修改为处理单个prompt
            # 修复：使用get_llm_config替代直接调用load_llm_config
            llm_config = get_llm_config(model_name)
            defense = selected_strategy['defense_agency'](task_agency=selected_strategy['task_agency'](), config_list=llm_config)
            template = load_attack_template()
            
            # 使用提供的prompt替换模板中的[INSERT PROMPT HERE]
            final_prompt = template.replace("[INSERT PROMPT HERE]", prompt_text)
            final_output = defense.defense_with_prompt(final_prompt)["content"]
            
            # 构建结果
            defense_result = {
                "name": "custom_prompt",
                "raw_response": defense.taskagency_agent.last_message()["content"],
                "defense_response": final_output
            }
            
            # 存储推理日志
            inference_logs[log_id] = {
                'timestamp': timestamp,
                'strategy': strategy_name,
                'model': model_name,
                'input': prompt_text,
                'result': defense_result,
                'log_type': 'prompt_defense',
                'stdout_content': stdout_capture.getvalue(),
                'stderr_content': stderr_capture.getvalue()
            }
            
            return jsonify({
                'status': 'success',
                'result': defense_result,
                'log_id': log_id,
                'stdout_content': stdout_capture.getvalue(),
                'stderr_content': stderr_capture.getvalue()
            })
        finally:
            # 恢复原始的stdout和stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    except Exception as e:
        logger.error(f'Error in defense_prompt: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/defense/batch', methods=['POST'])
def defense_batch():
    try:
        # 创建内存缓冲区来捕获stdout和stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # 保存原始的stdout和stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # 重定向stdout和stderr到捕获缓冲区和原始输出
            sys.stdout = Tee(stdout_capture, original_stdout)
            sys.stderr = Tee(stderr_capture, original_stderr)
            
            # 获取表单数据
            strategy_name = request.form.get('strategy')
            model_name = request.form.get('model')
            
            # 检查是否有文件上传
            if 'batch_file' not in request.files:
                return jsonify({'status': 'error', 'message': '没有文件上传'})
            
            file = request.files['batch_file']
            
            # 检查文件名是否为空
            if file.filename == '':
                return jsonify({'status': 'error', 'message': '没有选择文件'})
            
            # 检查文件类型是否允许
            if file and allowed_file(file.filename):
                # 保存上传的文件
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # 读取文件内容
                with open(filepath, 'r') as f:
                    try:
                        # 尝试将文件内容解析为JSON
                        batch_data = json.load(f)
                        # 确保batch_data是一个列表
                        if not isinstance(batch_data, list):
                            # 如果是字典，尝试提取响应内容
                            if isinstance(batch_data, dict):
                                batch_data = [v for v in batch_data.values()]
                            else:
                                return jsonify({'status': 'error', 'message': '文件内容格式不正确，应为JSON列表或字典'})
                    except json.JSONDecodeError:
                        # 如果不是JSON，尝试按行读取
                        batch_data = [line.strip() for line in f if line.strip()]
                
                # 查找选中的防御策略
                selected_strategy = next((s for s in defense_strategies if s['name'] == strategy_name), None)
                if not selected_strategy:
                    return jsonify({'status': 'error', 'message': f'策略 {strategy_name} 未找到'})
                
                # 记录日志 ID
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_id = f"{strategy_name}_{timestamp}"
                
                # 执行批量防御评估
                logger.info(f'Starting batch defense: strategy={strategy_name}, model={model_name}, items={len(batch_data)}')
                
                # 调用评估函数
                defense_results = evaluate_defense_with_output_list(
                    task_agency=selected_strategy['task_agency'],
                    defense_agency=selected_strategy['defense_agency'],
                    output_list=batch_data,
                    model_name=model_name
                )
                
                # 构建结果列表
                batch_results = []
                for i, (input_text, defense_output) in enumerate(zip(batch_data, defense_results)):
                    batch_results.append({
                        'index': i,
                        'input': input_text,
                        'defense_response': defense_output
                    })
                
                # 存储推理日志
                inference_logs[log_id] = {
                    'timestamp': timestamp,
                    'strategy': strategy_name,
                    'model': model_name,
                    'input_count': len(batch_data),
                    'results': batch_results,
                    'log_type': 'batch_defense',
                    'stdout_content': stdout_capture.getvalue(),
                    'stderr_content': stderr_capture.getvalue()
                }
                
                # 清理临时文件
                os.remove(filepath)
                
                return jsonify({
                    'status': 'success',
                    'results': batch_results,
                    'log_id': log_id,
                    'stdout_content': stdout_capture.getvalue(),
                    'stderr_content': stderr_capture.getvalue()
                })
            else:
                return jsonify({'status': 'error', 'message': '不支持的文件类型'})
        finally:
            # 恢复原始的stdout和stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    except Exception as e:
        logger.error(f'Error in defense_batch: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/logs/<log_id>')
def get_log(log_id):
    if log_id in inference_logs:
        return jsonify({
            'status': 'success',
            'log': inference_logs[log_id]
        })
    else:
        return jsonify({
            'status': 'error',
            'message': '日志未找到'
        })

@app.route('/logs')
def list_logs():
    # 返回最近的10条日志
    recent_logs = sorted(inference_logs.items(), key=lambda x: x[1]['timestamp'], reverse=True)[:10]
    return jsonify({
        'status': 'success',
        'logs': [{k: v} for k, v in recent_logs]
    })

if __name__ == '__main__':
    # 启动Flask应用
    # 修复：纠正语法错误
    app.run(debug=True, host='0.0.0.0', port=5000)
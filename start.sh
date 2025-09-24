export start_port=9005         # 设置环境变量 start_port，定义服务器起始端口号为 9005
export host_name=127.0.0.1       # 设置环境变量 host_name，定义主机名为 dgxh-1

for GPU in  1                 # 开始一个 for 循环，循环变量 GPU 依次取值为 0 和 1[6,7,8](@ref)
do
    # 打印提示信息，说明正在哪个 GPU 上启动服务器，以及计算出的端口号（起始端口 + GPU 编号）
    echo "Starting server on GPU $GPU on port $(($GPU + $start_port))"

    # 使用 sed 命令流编辑器，就地修改 (-i 选项) 配置文件 server_config.json
    # 将配置文件中 "port": 后跟的任何数字序列 ([0-9]+)，替换为计算出的新端口号 ($GPU + $start_port)
    # 目的是为每个服务器实例配置不同的端口[9,10](@ref)
    sed -i "s/\"port\": [0-9]\+/\"port\": $(($GPU + $start_port))/g" data/config/server_config.json

    # 设置两个环境变量后启动 llama_cpp.server 模块：
    # HOST=$host_name: 设置当前命令的环境变量 HOST 为之前定义的 host_name (dgxh-1)
    # CUDA_VISIBLE_DEVICES=$GPU: 设置环境变量 CUDA_VISIBLE_DEVICES，指定该进程仅使用当前循环指定的 GPU[1](@ref)
    # python3 -m llama_cpp.server --config_file data/config/server_config.json &: 在后台启动 (&)Python 模块，使用修改后的配置文件
    HOST=$host_name CUDA_VISIBLE_DEVICES=$GPU python3 -m llama_cpp.server \
    --config_file data/config/server_config.json &

    sleep 5                    # 等待 5 秒，给刚启动的服务器一些初始化时间，再开始下一次循环启动下一个实例
done
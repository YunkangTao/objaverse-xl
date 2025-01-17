import json
import os
import subprocess
import time
import logging
from typing import List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("transfer_data_to_117.log"), logging.StreamHandler()],
)


def transfer_directory(local_dir: str, remote_user: str, remote_host: str, remote_dir: str, ssh_port: int = 22, max_retries: int = 5, delay: int = 5) -> bool:
    """
    使用rsync将本地目录传输到远程服务器，失败时自动重试。

    :param local_dir: 本地目录路径
    :param remote_user: 远程服务器用户名
    :param remote_host: 远程服务器IP或主机名
    :param remote_dir: 远程目录路径
    :param max_retries: 最大重试次数
    :param delay: 重试间隔（秒）
    :return: 成功返回True，否则返回False
    """
    # 构建包含端口号的SSH命令
    ssh_command = f"ssh -p {ssh_port}"

    rsync_command = [
        "rsync",
        "-avz",  # 归档模式，详细输出，压缩传输
        "--partial",  # 保留部分传输的数据，断点续传
        "--progress",
        "-e",
        ssh_command,  # 指定使用的SSH命令和端口
        local_dir,
        f"{remote_user}@{remote_host}:{remote_dir}",
    ]

    attempt = 0
    while attempt < max_retries:
        try:
            logging.info(f"开始传输目录: {local_dir} (尝试 {attempt + 1}/{max_retries})")
            result = subprocess.run(rsync_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logging.info(f"成功传输目录: {local_dir}")
            return True
        except subprocess.CalledProcessError as e:
            attempt += 1
            logging.error(f"传输目录失败: {local_dir}. 错误信息: {e.stderr.strip()}")
            if attempt < max_retries:
                logging.info(f"将在 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                logging.error(f"达到最大重试次数，无法传输目录: {local_dir}")
                return False


def get_files_name(render_dir, wrong_data):
    glb_names = set()
    for filename in os.listdir(render_dir):
        if filename.lower().endswith('.glb') and os.path.isfile(os.path.join(render_dir, filename)):
            base_name, _ = os.path.splitext(filename)
            glb_names.add(base_name)

    # 从 wrong_data 文件中加载错误文件列表
    with open(wrong_data, 'r', encoding='utf-8') as file:
        wrong_list = json.load(file)

    # 假设 wrong_list 中包含不含扩展名的文件名
    wrong_set = set(wrong_list)

    # 计算有效文件名（排除错误文件）
    valid_names = glb_names - wrong_set

    # 返回排序后的文件名列表
    return sorted(valid_names)


def main(reder_result_dir, render_dir, wrong_data):
    transfer_dir_name = get_files_name(render_dir, wrong_data)[3000:4000]

    # 远程服务器信息
    remote_user = "chenyang_lei"  # 替换为目标服务器的用户名
    remote_host = "175.45.13.117"  # 或者是IP地址
    remote_dir = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/objaverse_dataset/"  # 替换为目标服务器上的目标路径
    ssh_port = 2608

    for dir_name in transfer_dir_name:
        # 构建本地目录路径
        local_dir = os.path.join(reder_result_dir, dir_name)
        # # 构建远程目录路径
        # remote_dir = f"{remote_base_dir}/{dir_name}"

        success = transfer_directory(local_dir, remote_user, remote_host, remote_dir, ssh_port=ssh_port)
        if not success:
            logging.warning(f"跳过目录: {local_dir}")


if __name__ == "__main__":
    reder_result_dir = "data/objaverse-animation-HQ_render_results"
    render_dir = "data/objaverse-animation-HQ"
    wrong_data = "wrong_data.json"
    main(reder_result_dir, render_dir, wrong_data)

import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import itertools
import logging
import math
import shutil
import sys

sys.path.insert(0, '/home/yunkang/objaverse-xl/scripts/rendering')

import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import uuid
import zipfile
from typing import Any, Dict, List, Literal, Optional, Union

import psutil

import fire
import fsspec
import GPUtil
from zip2video import (
    blender_to_COLMAP,
    blender_to_upy,
    camera_to_world,
    create_camera_pose_txt,
    create_video,
    extract_zip,
    get_sorted_npy_files,
    get_sorted_png_files,
    world_to_camera,
)

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str

# from loguru import logger


# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(processName)s %(message)s', handlers=[logging.FileHandler("parallel_execution.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used.

    Args:
        csv_filename (str): Name of the CSV file to save the logs to.
        *args: Arguments to save to the CSV file.

    Returns:
        None
    """
    args = ",".join([str(arg) for arg in args])
    # log that this object was rendered successfully
    # saving locally to avoid excessive writes to the cloud
    dirname = os.path.expanduser(f"/home/yunkang/objaverse-xl/data/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")


def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure.

    Args:
        path (str): Path to the directory to zip.
        ziph (zipfile.ZipFile): ZipFile handler object to write to.

    Returns:
        None
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # 计算相对于目标目录的路径，避免包含顶层目录
            arcname = os.path.relpath(file_path, path)
            ziph.write(file_path, arcname=arcname)


def add_camera_settings(args, base_move, num_renders):
    if base_move == 'up':
        movement_sequence = ['up'] * num_renders
        rotation_sequence = []

        movement_step_size = random.uniform(0.01, 0.015)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = (0, random.uniform(-3.5, -3.0), -(movement_step_size * num_renders / 2) * random.uniform(0.9, 1.1))
        camera_initial_rotation = 0.0

    elif base_move == 'down':
        movement_sequence = ['down'] * num_renders
        rotation_sequence = []

        movement_step_size = random.uniform(0.01, 0.015)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = (0, random.uniform(-3.5, -3.0), (movement_step_size * num_renders / 2) * random.uniform(0.9, 1.1))
        camera_initial_rotation = 0.0

    elif base_move == 'left':
        movement_sequence = ['left'] * num_renders
        rotation_sequence = []

        movement_step_size = random.uniform(0.02, 0.03)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = ((movement_step_size * num_renders / 2) * random.uniform(0.9, 1.1), random.uniform(-4.0, -3.0), 0)
        camera_initial_rotation = 0.0

    elif base_move == 'right':
        movement_sequence = ['right'] * num_renders
        rotation_sequence = []

        movement_step_size = random.uniform(0.02, 0.03)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = (-(movement_step_size * num_renders / 2) * random.uniform(0.9, 1.1), random.uniform(-4.0, -3.0), 0)
        camera_initial_rotation = 0.0

    elif base_move == 'forward':
        movement_sequence = ['forward'] * num_renders
        rotation_sequence = []

        movement_step_size = random.uniform(0.01, 0.03)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = (0, -random.uniform(2.0, 2.5) - movement_step_size * num_renders, 0)
        camera_initial_rotation = 0.0

    elif base_move == 'backward':
        movement_sequence = ['backward'] * num_renders
        rotation_sequence = []

        movement_step_size = random.uniform(0.01, 0.03)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = (0, -random.uniform(2.0, 2.5), 0)
        camera_initial_rotation = 0.0

    elif base_move == 'rotate_up':
        movement_sequence = []
        rotation_sequence = ['rotate_up'] * num_renders

        movement_step_size = random.uniform(0.01, 0.04)
        rotation_angle_degrees = random.uniform(0.1, 0.15)

        camera_direction = (0, 1, -math.tan(math.radians(rotation_angle_degrees / 2 * num_renders)) * random.uniform(0.9, 1.1))
        camera_initial_location = (0, random.uniform(-4.0, -3.0), 0)
        camera_initial_rotation = 0.0

    elif base_move == 'rotate_down':
        movement_sequence = []
        rotation_sequence = ['rotate_down'] * num_renders

        movement_step_size = random.uniform(0.01, 0.04)
        rotation_angle_degrees = random.uniform(0.1, 0.15)

        camera_direction = (0, 1, math.tan(math.radians(rotation_angle_degrees / 2 * num_renders)) * random.uniform(0.9, 1.1))
        camera_initial_location = (0, random.uniform(-4.0, -3.0), 0)
        camera_initial_rotation = 0.0

    elif base_move == 'rotate_left':
        movement_sequence = []
        rotation_sequence = ['rotate_left'] * num_renders

        movement_step_size = random.uniform(0.01, 0.04)
        rotation_angle_degrees = random.uniform(0.2, 0.6)

        camera_direction = (math.tan(math.radians(rotation_angle_degrees / 2 * num_renders)) * random.uniform(0.9, 1.1), 1, 0)
        camera_initial_location = (0, -random.uniform(2.5, 3.0), 0)
        camera_initial_rotation = 0.0

    elif base_move == 'rotate_right':
        movement_sequence = []
        rotation_sequence = ['rotate_right'] * num_renders

        movement_step_size = random.uniform(0.01, 0.04)
        rotation_angle_degrees = random.uniform(0.2, 0.6)

        camera_direction = (-math.tan(math.radians(rotation_angle_degrees / 2 * num_renders)) * random.uniform(0.9, 1.1), 1, 0)
        camera_initial_location = (0, -random.uniform(2.5, 3.0), 0)
        camera_initial_rotation = 0.0

    elif base_move == 'rotate_clockwise':
        movement_sequence = []
        rotation_sequence = ['rotate_clockwise'] * num_renders

        movement_step_size = random.uniform(0.01, 0.04)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = (0, -random.uniform(2.5, 3.0), 0)
        camera_initial_rotation = -rotation_angle_degrees / 2 * num_renders * random.uniform(0.9, 1.1)

    elif base_move == 'rotate_counterclockwise':
        movement_sequence = []
        rotation_sequence = ['rotate_counterclockwise'] * num_renders

        movement_step_size = random.uniform(0.01, 0.04)
        rotation_angle_degrees = random.uniform(0.2, 0.8)

        camera_direction = (0, 1, 0)
        camera_initial_location = (0, -random.uniform(2.5, 3.0), 0)
        camera_initial_rotation = rotation_angle_degrees / 2 * num_renders * random.uniform(0.9, 1.1)

    elif base_move == 'random':
        return None

    args += f' --camera_initial_location "{camera_initial_location}" \
    --camera_direction "{camera_direction}" \
    --movement_sequence "{movement_sequence}" \
    --rotation_sequence "{rotation_sequence}" \
    --movement_step_size {movement_step_size} \
    --rotation_angle_degrees {rotation_angle_degrees} \
    --camera_initial_rotation {camera_initial_rotation}'

    return args


def get_dir_names(dir_path):
    entries = os.listdir(dir_path)
    subdirs = [entry for entry in entries if os.path.isdir(os.path.join(dir_path, entry))]
    return subdirs


def unzip_file(zip_path, output_dir, fps):
    # 解压ZIP文件
    extract_zip(zip_path, output_dir)
    os.remove(zip_path)

    dir_names = get_dir_names(output_dir)
    for dir_name in dir_names:
        # 创建视频
        dir_path = os.path.join(output_dir, dir_name)
        png_files = get_sorted_png_files(dir_path)
        video_filename = f"{dir_name}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        create_video(png_files, video_path, fps)

        # 创建pose file
        camera_npy_files = get_sorted_npy_files(dir_path, end_xtend='_camera.npy')
        blender_camera_pose_filename = f"{dir_name}_blender_camera.txt"
        blender_camera_pose_path = os.path.join(output_dir, blender_camera_pose_filename)
        create_camera_pose_txt(camera_npy_files, blender_camera_pose_path)

        # # 创建pose file
        # world_npy_files = get_sorted_npy_files(dir_path, end_xtend='_world.npy')
        # blender_world_pose_filename = f"{dir_name}_blender_world.txt"
        # blender_world_pose_path = os.path.join(output_dir, blender_world_pose_filename)
        # create_camera_pose_txt(world_npy_files, blender_world_pose_path)

        # # 相机坐标系转世界坐标系
        # blender_transfer2world_pose_filename = f"{zip_name}_transfer2world_blender.txt"
        # blender_transfer2world_camera_pose_path = os.path.join(uid_path, blender_transfer2world_pose_filename)
        # camera_to_world(blender_camera_pose_path, blender_transfer2world_camera_pose_path)

        # # blender相机轨迹转右手直角坐标系
        # up_y_world_pose_filename = f"{zip_name}_upy_world.txt"
        # up_y_world_pose_path = os.path.join(uid_path, up_y_world_pose_filename)
        # blender_to_upy(blender_world_pose_path, up_y_world_pose_path)

        # # 世界坐标系转相机坐标系
        # upy_camera_pose_filename = f"{zip_name}_upy_camera.txt"
        # upy_camera_pose_path = os.path.join(uid_path, upy_camera_pose_filename)
        # world_to_camera(up_y_world_pose_path, upy_camera_pose_path)

        # blender相机轨迹转COLMAP坐标系
        COLMAP_camera_pose_filename = f"{dir_name}_COLMAP_camera.txt"
        COLMAP_camera_pose_path = os.path.join(output_dir, COLMAP_camera_pose_filename)
        blender_to_COLMAP(blender_camera_pose_path, COLMAP_camera_pose_path)

        shutil.rmtree(dir_path)
        os.remove(blender_camera_pose_path)

        # # 世界坐标系转相机坐标系
        # COLMAP_camera_pose_filename = f"{zip_name}_COLMAP_camera.txt"
        # COLMAP_camera_pose_path = os.path.join(uid_path, COLMAP_camera_pose_filename)
        # world_to_camera(COLMAP_world_pose_path, COLMAP_camera_pose_path)


def handle_found_object(
    uid,
    local_path: str,
    # file_identifier: str,
    # sha256: str,
    # metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    base_move,
    # successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    # failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.
        successful_log_file (str): Name of the log file to save successful renders to.
        failed_log_file (str): Name of the log file to save failed renders to.

    Returns: True if the object was rendered successfully, False otherwise.
    """
    # print("FUNCTION: handle_found_object")
    save_uid = uid
    args = f"--object_path '{local_path}' --num_renders {num_renders}"

    # get the GPU to use for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}.")

    with tempfile.TemporaryDirectory() as temp_dir:
        # get the target directory for the rendering job
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir {target_directory}"

        # check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine BLENDER_EEVEE"
        elif platform.system() == "Darwin" or (platform.system() == "Linux" and not using_gpu):
            # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
            # rendering. Generally, I'd only recommend using MacOS for debugging and
            # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
            args += " --engine CYCLES"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # check if we should only render the northern hemisphere
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        if 'random' in base_move:
            args += ' --random_or_custom random'
        else:
            args += ' --random_or_custom custom'
            args = add_camera_settings(args, base_move, num_renders)

        # get the command to run
        # command = f"blender-3.2.2-linux-x64/blender --background --python blender_script.py -- {args}"
        command = f"/home/yunkang/objaverse-xl/scripts/rendering/blender-3.2.2-linux-x64/blender --background --python /home/yunkang/objaverse-xl/scripts/rendering/blender_custom_script.py -- {args}"

        if using_gpu:
            command = f"export DISPLAY=:0.{gpu_i} && {command}"
        # print(command)
        # render the object (put in dev null)
        subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # subprocess.run(["bash", "-c", command])

        # check that the renders were saved successfully
        # png_files = glob.glob(os.path.join(target_directory, "*.png"))
        # metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        # npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        # if (len(png_files) != num_renders) or (len(npy_files) != num_renders) or (len(npy_files) != 2 * num_renders) or (len(metadata_files) != 1):
        #     logger.error(f"Found object {file_identifier} was not rendered successfully!")
        #     if failed_log_file is not None:
        #         log_processed_object(failed_log_file, file_identifier, sha256)
        #     return False

        # update the metadata
        metadata_path = os.path.join(target_directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)
        # metadata_file["sha256"] = sha256
        # metadata_file["file_identifier"] = file_identifier
        metadata_file["save_uid"] = save_uid
        # metadata_file["metadata"] = metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        # Make a zip of the target_directory.
        # Keeps the {save_uid} directory structure when unzipped
        with zipfile.ZipFile(f"{target_directory}_{base_move}.zip", "w", zipfile.ZIP_DEFLATED) as ziph:
            zipdir(target_directory, ziph)

        # move the zip to the render_dir
        fs, path = fsspec.core.url_to_fs(render_dir)

        # move the zip to the render_dir
        fs.makedirs(os.path.join(path, f"{save_uid}"), exist_ok=True)
        fs.put(os.path.join(f"{target_directory}_{base_move}.zip"), os.path.join(path, f"{save_uid}", f"{base_move}.zip"))

        # log that this object was rendered successfully
        # if successful_log_file is not None:
        #     log_processed_object(successful_log_file, file_identifier, sha256)

    # logger.info(f"Rendered {base_move} for {save_uid}")

    # zip_name = f"{base_move}"
    zip_path = os.path.join(path, f"{save_uid}", f"{base_move}.zip")
    output_dir = os.path.join(path, f"{save_uid}/{base_move}")
    # uid_path = os.path.join(path, f"{save_uid}")
    fps = 25

    unzip_file(zip_path, output_dir, fps)

    return True


def get_uid_from_str(string: str) -> str:
    """Generates a UUID from a string.

    Args:
        string (str): String to generate a UUID from.

    Returns:
        str: UUID generated from the string.
    """
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, string))


def process_move(args):

    (uid, num_renders, glb_file, render_results_dir, only_northern_hemisphere, parsed_gpu_devices, render_timeout, base_move) = args

    try:
        handle_found_object(
            uid=uid,
            local_path=glb_file,
            num_renders=num_renders,
            render_dir=render_results_dir,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout,
            base_move=base_move,
        )
        logger.info(f"Successfully processed move: {base_move} for UID: {uid}")
    except Exception as e:
        logger.error(f"Error processing {base_move} for UID: {uid}: {e}")


def get_files_path(render_dir, wrong_data):
    """
    获取指定目录下所有 .glb 文件的绝对路径，不包括子目录中的文件。

    :param render_dir: 渲染目录的路径
    :return: 包含所有 .glb 文件绝对路径的列表
    """
    glb_files_path = []

    if wrong_data:
        with open(wrong_data, 'r', encoding='utf-8') as file:
            # 使用 json.load() 将 JSON 数据解析为 Python 字典
            wrong_list = json.load(file)

        for wrong_file in wrong_list:
            glb_files_path.append(os.path.join('/home/yunkang/objaverse-xl/data/objaverse-animation-HQ', wrong_file + '.glb'))

    else:
        # 遍历指定目录中的所有条目
        for filename in os.listdir(render_dir):
            # 构建完整的文件路径
            file_path = os.path.join(render_dir, filename)

            # 检查是否是文件且以 .glb 结尾（不区分大小写）
            if os.path.isfile(file_path) and filename.lower().endswith('.glb'):
                # 获取绝对路径并添加到列表中
                absolute_path = os.path.abspath(file_path)
                glb_files_path.append(absolute_path)

    return sorted(glb_files_path)


def get_uid_from_file_path(glb_file):
    base_name = os.path.basename(glb_file)
    uid, _ = os.path.splitext(base_name)
    return uid


# 定义一个生成器来逐步生成任务
def generate_tasks(glb_files_path, base_camera_movements, num_renders, render_results_dir, only_northern_hemisphere, parsed_gpu_devices, render_timeout):
    for glb_file in glb_files_path:
        uid = get_uid_from_file_path(glb_file)
        for base_move in base_camera_movements:
            yield (uid, num_renders, glb_file, render_results_dir, only_northern_hemisphere, parsed_gpu_devices, render_timeout, base_move)


# 定义一个将可迭代对象分割成固定大小块的函数
def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def get_current_usage():
    """获取当前内存和CPU使用情况"""
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)  # 转换为MB
    cpu = psutil.cpu_percent(interval=1)
    return mem, cpu


def render_objects(
    render_dir: str = "/home/yunkang/objaverse-xl/data/objaverse-animation-HQ",
    render_results_dir: str = "/home/yunkang/objaverse-xl/data/objaverse-animation-HQ_render_results",
    # download_dir: Optional[str] = '/home/yunkang/objaverse-xl/data',
    num_renders: int = 72,
    processes: Optional[int] = None,
    # save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = "zip",
    only_northern_hemisphere: bool = True,
    render_timeout: int = 300000,
    gpu_devices: Optional[Union[int, List[int]]] = None,
    wrong_data=None,
) -> None:
    """Renders objects in the Objaverse-XL dataset with Blender

    Args:
        render_dir (str, optional): Directory where the objects will be rendered.
        download_dir (Optional[str], optional): Directory where the objects will be
            downloaded. If None, the objects will not be downloaded. Defaults to None.
        num_renders (int, optional): Number of renders to save of the object. Defaults
            to 12.
        processes (Optional[int], optional): Number of processes to use for downloading
            the objects. If None, defaults to multiprocessing.cpu_count() * 3. Defaults
            to None.
        save_repo_format (Optional[Literal["zip", "tar", "tar.gz", "files"]], optional):
            If not None, the GitHub repo will be deleted after rendering each object
            from it.
        only_northern_hemisphere (bool, optional): Only render the northern hemisphere
            of the object. Useful for rendering objects that are obtained from
            photogrammetry, since the southern hemisphere is often has holes. Defaults
            to False.
        render_timeout (int, optional): Number of seconds to wait for the rendering job
            to complete. Defaults to 300.
        gpu_devices (Optional[Union[int, List[int]]], optional): GPU device(s) to use
            for rendering. If an int, the GPU device will be randomly selected from 0 to
            gpu_devices - 1. If a list, the GPU device will be randomly selected from
            the list. If 0, the CPU will be used for rendering. If None, all available
            GPUs will be used. Defaults to None.

    Returns:
        None
    """

    # get the gpu devices to use
    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    if processes is None:
        processes = max(1, multiprocessing.cpu_count() // 2)
        logging.info(f"使用的进程数: {processes}")

    os.makedirs(render_results_dir, exist_ok=True)

    base_camera_movements = [
        'up',
        'down',
        'left',
        'right',
        'forward',
        'backward',
        'rotate_up',
        'rotate_down',
        'rotate_left',
        'rotate_right',
        'rotate_clockwise',
        'rotate_counterclockwise',
    ] + [f'random_{i}' for i in range(1, 13)]

    glb_files_path = get_files_path(render_dir, wrong_data)[4974:6000]
    tasks_generator = generate_tasks(glb_files_path, base_camera_movements, num_renders, render_results_dir, only_northern_hemisphere, parsed_gpu_devices, render_timeout)
    batch_size = 24

    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        for batch_number, batch in enumerate(chunked_iterable(tasks_generator, batch_size), 1):
            logging.info(f"提交第 {batch_number} 批次，共 {len(batch)} 个任务")
            futures = [executor.submit(process_move, task) for task in batch]

            # 等待所有Future完成
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            if not_done:
                logging.warning("部分任务超时未完成")
                # 可选：取消未完成的任务
                for future in not_done:
                    result = future.result()
                    future.cancel()

            for future in done:
                try:
                    result = future.result()
                    logging.info(result)
                except Exception as e:
                    logging.error(f"任务出错: {e}")

            # 当前批次完成
            logging.info(f"第 {batch_number} 批次完成")

            # 清理内存
            del batch
            del futures
            gc.collect()
            logging.info("触发垃圾回收")

            # 获取并记录当前系统资源使用情况
            mem, cpu = get_current_usage()
            logging.info(f"批次 {batch_number} 结束后: 内存使用: {mem:.2f} MB, CPU 使用: {cpu:.2f}%")

        logging.info("所有批次处理完成")


if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间
    fire.Fire(render_objects)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"程序运行总耗时: {elapsed_time} 秒")

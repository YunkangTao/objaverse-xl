import os
import zipfile
import numpy as np
from tqdm import tqdm

# from moviepy.editor import ImageSequenceClip, ColorClip, CompositeVideoClip
import cv2
import os
from PIL import Image


def extract_zip(zip_path, extract_to):
    """解压ZIP文件到指定目录"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        # print(f"解压完成: {zip_path}")
    except zipfile.BadZipFile:
        print(f"无法解压（文件可能损坏）: {zip_path}")


def get_sorted_png_files(directory):
    """获取目录下按名称排序的所有PNG文件路径"""
    files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    files.sort()  # 按名称排序
    return [os.path.join(directory, f) for f in files]


def get_sorted_npy_files(directory, end_xtend):
    """获取目录下按名称排序的所有NPY文件路径"""
    files = [f for f in os.listdir(directory) if f.lower().endswith(end_xtend)]
    files.sort()  # 按名称排序
    return [os.path.join(directory, f) for f in files]


def create_video(image_files, video_path, fps=30, output_size=None):
    """将图片序列合成为视频"""
    if not image_files:
        print("没有找到PNG文件，跳过视频生成。")
        return

    # print(f'图片共有{len(image_files)}张')

    # 打开第一张图片以获取尺寸
    first_image = Image.open(image_files[0]).convert("RGBA")
    if output_size is None:
        width, height = first_image.size
    else:
        width, height = output_size

    # 定义视频编解码器和VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 H.264 编码
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for idx, img_path in enumerate(image_files):
        # 打开图片并转换为RGBA模式以处理透明度
        img = Image.open(img_path).convert("RGBA")

        # 创建白色背景
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))

        # 计算图片放置的位置以居中
        img_width, img_height = img.size
        position = ((width - img_width) // 2, (height - img_height) // 2)

        # 将图片粘贴到白色背景上
        background.paste(img, position, img)

        # 转换为BGR格式以适应OpenCV
        frame = background.convert("RGB")
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # 写入视频帧
        video_writer.write(frame)

        # if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
        #     print(f"已处理 {idx + 1}/{len(image_files)} 张图片")

    # 释放资源
    video_writer.release()
    # print(f"视频已保存至 {video_path}")


def create_camera_pose_txt(npy_files, pose_path, delimiter=' '):
    """
    读取一组.npy文件并将其内容写入一个txt文件中。

    参数：
    - npy_file_list: List[str]，.npy文件的路径列表。
    - output_txt_path: str，输出txt文件的路径。
    - delimiter: str，分隔符，默认为空格。
    """
    with open(pose_path, 'w') as txt_file:
        for npy_file in npy_files:
            if not os.path.isfile(npy_file):
                print(f"警告：文件 {npy_file} 不存在，已跳过。")
                continue
            try:
                data = np.load(npy_file)
                # 将数组展平成一维以便写入
                flat_data = data.flatten()
                # 将数值转换为字符串并用分隔符连接
                line = delimiter.join(f"{x:.9f}" for x in flat_data)
                txt_file.write(line + '\n')
            except Exception as e:
                print(f"错误：无法处理文件 {npy_file}。错误信息：{e}")


def camera_to_world(input_file, output_file):
    """
    将blender坐标系系统下的相机坐标系的外参转换为世界坐标系的外参。

    参数:
    - input_file: 输入的txt文件路径，每行12个用空格分隔的值。
    - output_file: 输出的txt文件路径，每行12个用空格分隔的值。
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            parts = line.strip().split()
            if len(parts) != 12:
                print(f"警告: 第 {line_num} 行不包含12个值，跳过该行。")
                continue
            try:
                values = list(map(float, parts))
            except ValueError:
                print(f"警告: 第 {line_num} 行包含非数值数据，跳过该行。")
                continue
            # 构建3x4矩阵
            extrinsic = np.array(values).reshape(3, 4)
            R = extrinsic[:, :3]
            t = extrinsic[:, 3]
            # 计算R的转置
            R_transpose = R.T
            # 计算新的平移向量
            t_new = -R_transpose @ t
            # 构建新的外参矩阵
            world_extrinsic = np.hstack((R_transpose, t_new.reshape(3, 1)))
            # 将矩阵展开为12个值
            output_values = world_extrinsic.flatten()
            # 格式化为字符串并写入输出文件
            output_line = ' '.join(f"{v:.9f}" for v in output_values)
            outfile.write(output_line + '\n')


def world_to_camera(up_y_world_pose_path, upy_camera_pose_path):
    """
    将相机坐标系的外参转换为世界坐标系的外参。

    参数:
    - input_file: 输入的txt文件路径，有若干行，每行12个用空格分隔的值。
    - output_file: 输出的txt文件路径，有若干行，每行12个用空格分隔的值。
    """
    try:
        with open(up_y_world_pose_path, 'r') as fin, open(upy_camera_pose_path, 'w') as fout:
            line_num = 0
            for line in fin:
                line_num += 1
                # 去除首尾空白字符并按空格分割
                parts = line.strip().split()

                # 检查每行是否有12个数值
                if len(parts) != 12:
                    raise ValueError(f"第 {line_num} 行的数值数量不是12个")

                # 将字符串转换为浮点数
                values = list(map(float, parts))

                # 构建3x4矩阵
                mat = np.array(values).reshape(3, 4)

                # 分离旋转矩阵 R 和平移向量 t
                R_world = mat[:, :3]
                t_world = mat[:, 3].reshape(3, 1)

                R_camera = R_world.T
                t_camera = -np.dot(R_camera, t_world)

                # 合并新的 R 和 t 成3x4矩阵
                cam_extrinsics = np.hstack((R_camera, t_camera))

                # 将矩阵展平成一行，格式化为字符串
                cam_extrinsics_flat = ' '.join(f"{num:.9f}" for num in cam_extrinsics.flatten())

                # 写入输出文件
                fout.write(cam_extrinsics_flat + '\n')

        print("转换完成，结果已保存到", upy_camera_pose_path)

    except FileNotFoundError as e:
        print(f"文件未找到: {e.filename}")
    except ValueError as ve:
        print("值错误:", ve)
    except Exception as ex:
        print("发生错误:", ex)


def blender_to_upy(blender_world_pose_path, up_y_world_pose_path):
    """
    将 Blender 坐标系（Z 轴向上）的世界坐标相机外参转换为 Y 轴向上的右手直角坐标系的世界坐标相机外参。

    参数:
    - blender_world_pose_path (str): Blender 相机外参的输入文件路径。
    - up_y_world_pose_path (str): 转换后相机外参的输出文件路径。
    """
    # 定义转换旋转矩阵 (绕 X 轴 +90 度)
    R_convert = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    try:
        with open(blender_world_pose_path, 'r') as infile, open(up_y_world_pose_path, 'w') as outfile:
            line_num = 0
            for line in infile:
                line_num += 1
                # 去除首尾空白字符并按空格分割
                parts = line.strip().split()

                # 检查每行是否有12个数
                if len(parts) != 12:
                    print(f"警告: 第 {line_num} 行不包含12个值，已跳过。")
                    continue

                try:
                    # 将字符串转换为浮点数
                    values = [float(part) for part in parts]
                except ValueError:
                    print(f"警告: 第 {line_num} 行包含非数值数据，已跳过。")
                    continue

                # 重塑为 3x4 矩阵
                pose_matrix = np.array(values).reshape(3, 4)

                # 分离旋转矩阵 R_b 和平移向量 t_b
                R_b = pose_matrix[:, :3]
                t_b = pose_matrix[:, 3]

                # 应用转换旋转矩阵
                R_t = R_convert @ R_b
                t_t = R_convert @ t_b

                # 合并转换后的旋转矩阵和平移向量
                transformed_pose = np.hstack((R_t, t_t.reshape(3, 1)))

                # 将矩阵展平成 12 个数，并格式化为字符串
                transformed_values = ' '.join(['{:.9f}'.format(num) for num in transformed_pose.flatten()])

                # 写入输出文件
                outfile.write(transformed_values + '\n')

        print(f"转换完成。转换后的相机外参已保存到: {up_y_world_pose_path}")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}")
    except Exception as e:
        print(f"发生错误: {e}")


def blender_to_COLMAP(blender_camera_pose_path, COLMAP_camera_pose_path):
    """camera pose to camera pose

    Args:
        blender_world_pose_path (_type_): _description_
        ROS_world_pose_path (_type_): _description_
    """
    # R_convert = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    R_convert = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T_convert = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    with open(COLMAP_camera_pose_path, 'w') as outfile:
        outfile.write(f'{COLMAP_camera_pose_path}' + '\n')

    try:
        with open(blender_camera_pose_path, 'r') as infile, open(COLMAP_camera_pose_path, 'a') as outfile:
            line_num = 0
            for line in infile:
                line_num += 1
                # 去除首尾空白字符并按空格分割
                parts = line.strip().split()

                # 检查每行是否有12个数
                if len(parts) != 12:
                    print(f"警告: 第 {line_num} 行不包含12个值，已跳过。")
                    continue

                try:
                    # 将字符串转换为浮点数
                    values = [float(part) for part in parts]
                except ValueError:
                    print(f"警告: 第 {line_num} 行包含非数值数据，已跳过。")
                    continue

                # 重塑为 3x4 矩阵
                pose_matrix = np.array(values).reshape(3, 4)

                # 分离旋转矩阵 R_b 和平移向量 t_b
                R_b = pose_matrix[:, :3]
                t_b = pose_matrix[:, 3]

                # 应用转换旋转矩阵
                R_t = R_convert @ R_b
                t_t = T_convert @ t_b

                # 合并转换后的旋转矩阵和平移向量
                transformed_pose = np.hstack((R_t, t_t.reshape(3, 1)))

                # 将矩阵展平成 12 个数，并格式化为字符串
                transformed_values = ' '.join(['{:.9f}'.format(num) for num in transformed_pose.flatten()])
                prefix = ["{:08}".format(line_num), f"{1.0:.9f}", f"{1.75:.9f}", f"{0.500000000:.9f}", f"{0.500000000:.9f}", f"{0.000000000:.9f}", f"{0.000000000:.9f}"]
                prefix_str = ' '.join(prefix)
                transformed_values = prefix_str + ' ' + transformed_values

                # 写入输出文件
                outfile.write(transformed_values + '\n')

        # print(f"转换完成。转换后的相机外参已保存到: {COLMAP_camera_pose_path}")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}")
    except Exception as e:
        print(f"发生错误: {e}")


def process_zip_files(directory, output_dir, fps=30):
    """处理目录中的所有ZIP文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zip_files = [f for f in os.listdir(directory) if f.lower().endswith('.zip')]
    if not zip_files:
        print("未找到ZIP文件。")
        return

    for zip_file in tqdm(zip_files, desc="处理ZIP文件"):
        zip_path = os.path.join(directory, zip_file)
        # 为每个ZIP文件创建一个独立的文件夹
        zip_name = os.path.splitext(zip_file)[0]
        extract_path = os.path.join(output_dir, zip_name)
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        # 解压ZIP文件
        extract_zip(zip_path, output_dir)

        # 获取排序后的PNG文件
        png_files = get_sorted_png_files(extract_path)

        # 定义视频输出路径
        video_filename = f"{zip_name}.mp4"
        video_path = os.path.join(output_dir, video_filename)

        # 创建视频
        create_video(png_files, video_path, fps)

        # 获取排序后的NPY文件
        npy_files = get_sorted_npy_files(extract_path)

        # 定义视频输出路径
        blender_camera_pose_filename = f"{zip_name}_camera_blender.txt"
        blender_camera_pose_path = os.path.join(output_dir, blender_camera_pose_filename)

        # 创建pose file
        create_camera_pose_txt(npy_files, blender_camera_pose_path)

        # 相机坐标系转世界坐标系
        # 定义视频输出路径
        blender_world_pose_filename = f"{zip_name}_world_blender.txt"
        blender_world_camera_pose_path = os.path.join(output_dir, blender_world_pose_filename)
        camera_to_world(blender_camera_pose_path, blender_world_camera_pose_path)

        # blender相机轨迹转右手直角坐标系
        up_y_world_pose_filename = f"{zip_name}_world_upy.txt"
        up_y_world_pose_path = os.path.join(output_dir, up_y_world_pose_filename)
        blender_to_upy(blender_world_camera_pose_path, up_y_world_pose_path)

        # 世界坐标系转相机坐标系
        # 定义视频输出路径
        upy_camera_pose_filename = f"{zip_name}_camera_upy.txt"
        upy_camera_pose_path = os.path.join(output_dir, upy_camera_pose_filename)
        world_to_camera(up_y_world_pose_path, upy_camera_pose_path)

    print("所有ZIP文件处理完成。")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="解压ZIP文件并将PNG图片合成为视频。")
    # parser.add_argument("--input_dir", default='/home/yunkang/objaverse-xl/data/smithsonian/rendered_results/backward0.1', help="包含ZIP文件的输入目录")
    # parser.add_argument("--output_dir", default='/home/yunkang/objaverse-xl/data/smithsonian/rendered_results/backward0.1', help="输出视频和解压文件的目录")
    # parser.add_argument("--fps", type=int, default=25, help="视频的帧率 (默认: 30)")

    # args = parser.parse_args()

    # process_zip_files(args.input_dir, args.output_dir, args.fps)
    png_files = get_sorted_png_files(
        "/home/yunkang/objaverse-xl/data/objaverse-animation-HQ_render_results/fbdb761754c04d639faac2f353877042/backward/animation.diamond_ender_colossus.attack"
    )

    # 定义视频输出路径
    # video_filename =
    video_path = f"/home/yunkang/objaverse-xl/data/objaverse-animation-HQ_render_results/fbdb761754c04d639faac2f353877042/backward.animation.diamond_ender_colossus.attack.mp4"

    # 创建视频
    create_video(png_files, video_path, 25)

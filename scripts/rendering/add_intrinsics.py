import os


def process_txt_file_inplace(file_path):
    """
    处理指定的txt文件：
    1. 在第一行添加文件的绝对路径。
    2. 在每一行的数据前面添加7个0。
    3. 将结果写回到原始文件中。

    :param file_path: 要处理的txt文件的路径
    """
    # 获取文件的绝对路径
    abs_path = os.path.abspath(file_path)

    try:
        # 读取文件所有行
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return
    except IOError as e:
        print(f"读取文件时出错: {e}")
        return

    # 创建新的内容列表，并在第一行添加文件路径
    new_lines = [abs_path + '\n']

    for idx, line in enumerate(lines, start=1):
        # 去除每行的首尾空白字符（包括换行符），并按空白字符分割成列表
        data = line.strip().split()

        # 在数据前面添加7个'0'。如果需要添加其他数据，可以修改此处
        # prefix = ['0'] * 7
        prefix = ["{:08}".format(idx), f"{1.0:.9f}", f"{1.75:.9f}", f"{0.500000000:.9f}", f"{0.500000000:.9f}", f"{0.000000000:.9f}", f"{0.000000000:.9f}"]
        new_data = prefix + data

        # 将修改后的数据列表重新合并成字符串，并添加换行符
        new_line = ' '.join(new_data) + '\n'
        new_lines.append(new_line)

    try:
        # 将新的内容写回原文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(new_lines)
        print(f"处理完成！文件已更新为: {file_path}")
    except IOError as e:
        print(f"写入文件时出错: {e}")


def main():
    file_path = "/home/yunkang/objaverse-xl/scripts/rendering/Armature|idle_anim_COLMAP_camera.txt"
    process_txt_file_inplace(file_path)


if __name__ == "__main__":
    main()

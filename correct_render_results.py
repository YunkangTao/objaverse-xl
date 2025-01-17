import json
import os


def find_problem_directories(parent_dir):
    # 存储子目录中子子目录数量不是24的子目录名称
    not_24_subsubdirs = []
    # 存储在拥有24个子子目录但.mp4文件数量不一致的子目录名称
    mp4_count_issues = []

    # 遍历父目录下的所有子目录
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)
        if os.path.isdir(subdir_path):
            # 获取子目录下的所有子子目录
            subsubdirs = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
            if len(subsubdirs) != 24:
                not_24_subsubdirs.append(subdir)
            else:
                mp4_counts = []
                for subsubdir in subsubdirs:
                    subsubdir_path = os.path.join(subdir_path, subsubdir)
                    mp4_files = [f for f in os.listdir(subsubdir_path) if f.lower().endswith('.mp4')]
                    mp4_counts.append(len(mp4_files))

                # 检查所有.mp4文件数量是否相同
                if len(set(mp4_counts)) > 1:
                    mp4_count_issues.append(subdir)

    return not_24_subsubdirs, mp4_count_issues


def write_to_json(not_24_dirs, mp4_issues_dirs, output_file):
    data = {"子目录中子子目录数量不是24的子目录": not_24_dirs, "在拥有24个子子目录的情况下，.mp4文件数量不一致的子目录": mp4_issues_dirs}
    data = not_24_dirs + mp4_issues_dirs
    print(f"共有{len(data)}个数据有问题")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"结果已写入 {output_file}")


if __name__ == "__main__":
    # 替换为你的父目录路径
    parent_directory = "/home/yunkang/objaverse-xl/data/objaverse-animation-HQ_render_results"
    # 输出的JSON文件路径
    output_json = "/home/yunkang/objaverse-xl/wrong_data.json"

    not_24_dirs, mp4_issues_dirs = find_problem_directories(parent_directory)
    write_to_json(not_24_dirs, mp4_issues_dirs, output_json)

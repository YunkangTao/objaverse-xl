import os
from moviepy.editor import ImageSequenceClip


def create_video_from_images(image_folder, output_video, fps=10):
    # 获取所有PNG文件并排序
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith('.png')]
    images.sort()  # 按文件名排序

    if not images:
        raise ValueError("目录中没有找到PNG文件。")

    # 创建视频剪辑
    clip = ImageSequenceClip(images, fps=fps)
    # 写入视频文件
    clip.write_videofile(output_video, codec='libx264')
    print(f"视频已成功保存为 {output_video}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='将目录中的PNG图片合成为视频。')
    parser.add_argument(
        '--image_folder',
        default='/home/yunkang/objaverse-xl/data/objaverse-animation-HQ_render_results/000_top10/0a0b504f51a94d95a2d492d3c372ebe5/000_render_animation/Armature|idle_anim',
        help='PNG图片所在的目录路径',
    )
    parser.add_argument(
        '--output_video',
        default='/home/yunkang/objaverse-xl/data/objaverse-animation-HQ_render_results/000_top10/0a0b504f51a94d95a2d492d3c372ebe5/000_render_animation/Armature|idle_anim.mp4',
        help='输出视频文件的路径，例如 output.mp4',
    )
    parser.add_argument('--fps', type=int, default=10, help='视频帧率 (默认: 30)')
    args = parser.parse_args()

    create_video_from_images(args.image_folder, args.output_video, fps=args.fps)

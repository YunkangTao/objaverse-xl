from ast import Tuple
import os
import bpy
from mathutils import Matrix
import numpy as np


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object):
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    rt_matrix_camera = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )

    # 为了构建从摄像机到世界的矩阵，使用原始旋转矩阵和位置
    R_cam_to_world = rotation.to_matrix()
    T_cam_to_world = location

    # 构建从摄像机到世界的3x4 RT矩阵
    rt_matrix_world = Matrix(
        (
            R_cam_to_world[0][:] + (T_cam_to_world[0],),
            R_cam_to_world[1][:] + (T_cam_to_world[1],),
            R_cam_to_world[2][:] + (T_cam_to_world[2],),
        )
    )

    return rt_matrix_camera, rt_matrix_world


def main():
    # 在场景中查找相机对象
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']

    if cameras:
        cam = cameras[0]
        print(f"使用现有相机: {cam.name}")
    else:
        # 如果没有相机，创建一个默认相机
        cam_data = bpy.data.cameras.new(name="Test_Camera")
        cam = bpy.data.objects.new(name="Test_Camera", object_data=cam_data)
        bpy.context.collection.objects.link(cam)
        print("创建了一个新的相机: Test_Camera")

    # 获取RT矩阵
    output_dir = '/home/yunkang/objaverse-xl'
    i = 0
    rt_matrix_camera, rt_matrix_world = get_3x4_RT_matrix_from_blender(cam)
    rt_matrix_path_camera = os.path.join(output_dir, f"{i:03d}_camera.npy")
    rt_matrix_path_world = os.path.join(output_dir, f"{i:03d}_world.npy")
    np.save(rt_matrix_path_camera, rt_matrix_camera)
    np.save(rt_matrix_path_world, rt_matrix_world)

    # 打印结果
    print(f"相机 '{cam.name}' 的 3x4 RT 世界矩阵:")
    print(rt_matrix_world)

    # 打印结果
    print(f"相机 '{cam.name}' 的 3x4 RT 相机矩阵:")
    print(rt_matrix_camera)


if __name__ == "__main__":
    main()

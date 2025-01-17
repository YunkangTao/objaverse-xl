"""Blender script to render images of 3D models."""

from ast import literal_eval
import sys
import argparse
import json
import math
import os
import random

import time
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector
import mathutils
from math import radians

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz)
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def randomize_camera_position(
    radius_min: float = 3.0,
    radius_max: float = 4.0,
    maxz: float = 4.0,
    minz: float = -4.0,
    only_northern_hemisphere: bool = False,
) -> Vector:
    x, y, z = _sample_spherical(radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz)
    if only_northern_hemisphere:
        z = abs(z)
    return Vector((x, y, z))


def update_camera(initial_loc: Vector, target_loc: Vector, i: int, num_renders: int) -> bpy.types.Object:
    """
    更新相机的位置和旋转，使其在初始位置和目标位置之间线性插值。

    Args:
        initial_loc (Vector): 初始相机位置。
        target_loc (Vector): 目标相机位置。
        i (int): 当前渲染循环的索引。
        num_renders (int): 总的渲染次数。

    Returns:
        bpy.types.Object: 更新后的相机对象。
    """
    if num_renders <= 1:
        t = 1.0
    else:
        t = i / (num_renders - 1)

    # 线性插值计算新位置
    new_location = initial_loc.lerp(target_loc, t)

    # 获取相机对象
    camera = bpy.data.objects["Camera"]

    # 更新相机位置
    camera.location = new_location

    # 计算指向原点的方向
    direction = Vector((0, 0, 0)) - new_location
    direction.normalize()

    # 计算旋转四元数，使相机朝向原点
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        # from io_scene_usdz.import_usdz import import_usdz

        # import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object):
    """从给定的Blender摄像机对象中返回两个3x4的RT矩阵。

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Tuple[Matrix, Matrix]: 第一个是相机的RT矩阵，第二个是世界的RT矩阵。
    """
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
    # location, rotation = cam.matrix_world
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


def get_3x4_RT_world_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    # 分解世界矩阵为位置、旋转、缩放
    loc, rot = cam.matrix_world.decompose()[0:2]

    # 获取旋转矩阵 (3x3)
    R = rot.to_matrix()

    # 获取平移向量
    t = loc

    # 构建 3x4 外参矩阵
    RT = Matrix(
        (
            R[0][:] + (t[0],),
            R[1][:] + (t[1],),
            R[2][:] + (t[2],),
        )
    )

    return RT


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs["Base Color"].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(obj: bpy.types.Object, color: Tuple[float, float, float, float]) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = image_filepaths | material_filepaths | linked_libraries_filepaths
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += len(shape_keys.key_blocks) - 1  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def center_object_to_origin(obj):
    # 选择对象并确保处于对象模式
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # 计算几何中心
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector((min([v.x for v in bbox_corners]), min([v.y for v in bbox_corners]), min([v.z for v in bbox_corners])))
    max_corner = mathutils.Vector((max([v.x for v in bbox_corners]), max([v.y for v in bbox_corners]), max([v.z for v in bbox_corners])))
    center = (min_corner + max_corner) / 2

    # 移动几何数据
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (-center.x, -center.y, -center.z)


# --- 移动相机函数定义 ---


def move_camera_up(cam, step_size=1.0):
    cam.location.z += step_size


def move_camera_down(cam, step_size=1.0):
    cam.location.z -= step_size


def move_camera_left(cam, step_size=1.0):
    cam.location.x -= step_size


def move_camera_right(cam, step_size=1.0):
    cam.location.x += step_size


def move_camera_forward(cam, step_size=1.0):
    cam.location.y += step_size


def move_camera_backward(cam, step_size=1.0):
    cam.location.y -= step_size


# --- 旋转相机函数定义 ---


def rotate_camera_up(cam, angle_degrees=10):
    angle_radians = math.radians(angle_degrees)
    cam.rotation_euler.x += angle_radians


def rotate_camera_down(cam, angle_degrees=10):
    angle_radians = math.radians(angle_degrees)
    cam.rotation_euler.x -= angle_radians


def rotate_camera_left(cam, angle_degrees=10):
    angle_radians = math.radians(angle_degrees)
    cam.rotation_euler.z += angle_radians


def rotate_camera_right(cam, angle_degrees=10):
    angle_radians = math.radians(angle_degrees)
    cam.rotation_euler.z -= angle_radians


def rotate_camera_clockwise(cam, angle_degrees=10):
    angle_radians = math.radians(angle_degrees)
    cam.rotation_euler.y += angle_radians  # 顺时针旋转为负方向


def rotate_camera_counterclockwise(cam, angle_degrees=10):
    angle_radians = math.radians(angle_degrees)
    cam.rotation_euler.y -= angle_radians  # 逆时针旋转为正方向


def get_animation_list_from_file_path():
    all_animations = set()  # 初始化一个空集合 all_animations
    for obj in bpy.data.objects:  # 遍历Blender场景中的所有对象
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            if not obj.animation_data.nla_tracks.get(action.name.split("_")[0]):
                new_track = obj.animation_data.nla_tracks.new()
                new_track.name = action.name.split("_")[0]
                new_strip = new_track.strips.new(action.name, int(action.frame_start), action)
            obj.animation_data.action = None
        if obj.animation_data and obj.animation_data.nla_tracks:
            all_animations.update([track.name for track in obj.animation_data.nla_tracks])

    return all_animations


def render_one_track(output_dir, track_name, initial_camera, target_camera, num_renders):
    current_animation_output_dir = os.path.join(output_dir, track_name)

    frame_start, frame_end = 10000000, -10000000
    for obj in bpy.data.objects:
        if obj.animation_data:
            track = obj.animation_data.nla_tracks.get(track_name)
            if track and track.strips:
                frame_start = min(frame_start, min([strip.frame_start for strip in track.strips]))
                frame_end = max(frame_end, max([strip.frame_end for strip in track.strips]))

    frame_start, frame_end = int(frame_start), int(frame_end)

    if frame_end - frame_start < 10:
        os.removedirs(current_animation_output_dir)
        return

    for obj in bpy.data.objects:
        if obj.animation_data:
            for obj_track in obj.animation_data.nla_tracks:
                obj_track.mute = obj_track.name != track_name

    camera = update_camera(initial_camera, target_camera, 0, num_renders)
    # for frame_idx in range(frame_start, frame_end):
    for frame_idx in range(frame_start, num_renders + frame_start):
        cycle_length = frame_end - frame_start + 1
        current_frame = ((frame_idx - frame_start) % cycle_length) + frame_start
        scene.frame_set(current_frame)
        render_path = os.path.join(current_animation_output_dir, f"0static_{frame_idx:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # 保存RT矩阵，相机坐标系和世界坐标系的外参
        rt_matrix_camera, rt_matrix_world = get_3x4_RT_matrix_from_blender(camera)
        rt_matrix_path_camera = os.path.join(current_animation_output_dir, f"0static_{frame_idx:03d}_camera.npy")
        rt_matrix_path_world = os.path.join(current_animation_output_dir, f"0static_{frame_idx:03d}_world.npy")
        np.save(rt_matrix_path_camera, rt_matrix_camera)
        np.save(rt_matrix_path_world, rt_matrix_world)

    # 进行多次渲染，根据num_renders指定的次数
    for i, frame_idx in zip(range(num_renders), range(frame_start, num_renders + frame_start)):

        camera = update_camera(initial_camera, target_camera, i, num_renders)

        cycle_length = frame_end - frame_start + 1
        current_frame = ((frame_idx - frame_start) % cycle_length) + frame_start
        scene.frame_set(current_frame)
        render_path = os.path.join(current_animation_output_dir, f"1dynamic_{i:03d}_{frame_idx:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # 保存RT矩阵，相机坐标系和世界坐标系的外参
        rt_matrix_camera, rt_matrix_world = get_3x4_RT_matrix_from_blender(camera)
        rt_matrix_path_camera = os.path.join(current_animation_output_dir, f"1dynamic_{i:03d}_{frame_idx:03d}_camera.npy")
        rt_matrix_path_world = os.path.join(current_animation_output_dir, f"1dynamic_{i:03d}_{frame_idx:03d}_world.npy")
        np.save(rt_matrix_path_camera, rt_matrix_camera)
        np.save(rt_matrix_path_world, rt_matrix_world)


def render_object_random(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # Set up cameras
    cam = scene.objects["Camera"]  # 获取场景中的摄像机对象
    cam.data.lens = 35  # 设置摄像机的镜头焦距为35mm
    cam.data.sensor_width = 35  # 设置摄像机传感器的宽度为32mm
    cam.data.sensor_height = 20

    # 设置摄像机约束，使其始终指向一个空对象
    cam_constraint = cam.constraints.new(type="TRACK_TO")  # 创建一个新类型为TRACK_TO的约束
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"  # 设置摄像机的跟踪轴为负Z轴
    cam_constraint.up_axis = "UP_Y"  # 设置摄像机的上方向为Y轴
    empty = bpy.data.objects.new("Empty", None)  # 创建一个新的空对象，命名为"Empty"
    scene.collection.objects.link(empty)  # 将空对象链接到当前场景中
    cam_constraint.target = empty  # 将空对象设为摄像机约束的目标

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(object_path=object_file, scene=scene, bdata=bpy.data)
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    # 规范化场景，确保对象和摄像机的位置、缩放等符合预期
    normalize_scene()

    # 随机化场景中的灯光设置，以增加渲染的多样性
    randomize_lighting()

    all_animations = get_animation_list_from_file_path()
    for track_name in all_animations:
        os.makedirs(os.path.join(args.output_dir, track_name), exist_ok=True)

    # 随机设置摄像机的位置和方向
    initial_camera = randomize_camera_position(only_northern_hemisphere=only_northern_hemisphere)
    target_camera = randomize_camera_position(only_northern_hemisphere=only_northern_hemisphere)

    if all_animations:
        for track_name in all_animations.copy():
            render_one_track(output_dir, track_name, initial_camera, target_camera, num_renders)


def render_with_movement_and_rotation(
    object_file: str,
    num_renders: int,
    output_dir: str,
    camera_initial_location: tuple = (10, 0, 10),
    camera_direction: tuple = (-1, 0, -1),
    movement_sequence: list = [],
    rotation_sequence: list = [],
    movement_step_size: float = 0.5,
    rotation_angle_degrees: float = 5,
    camera_initial_rotation: float = 0.0,
) -> None:
    """保存渲染的图像及其摄像机矩阵和对象元数据，结合移动和旋转。

    Args:
        object_file (str): 对象文件的路径。
        num_renders (int): 要保存的对象渲染次数。
        only_northern_hemisphere (bool): 是否仅渲染对象在北半球的部分。
        output_dir (str): 渲染图像和元数据将保存到的目录路径。
        camera_initial_location (tuple, optional): 相机的初始位置 (x, y, z)。默认是 (10, 0, 10)。
        camera_direction (tuple, optional): 相机的固定朝向方向向量 (dx, dy, dz)。默认是 (-1, 0, -1)。
        movement_sequence (list, optional): 渲染过程中每步要执行的移动指令列表。
            可选值：'up', 'down', 'left', 'right', 'forward', 'backward'
        rotation_sequence (list, optional): 渲染过程中每步要执行的旋转指令列表。
            可选值：'rotate_up', 'rotate_down', 'rotate_left', 'rotate_right', 'rotate_clockwise', 'rotate_counterclockwise'
        movement_step_size (float, optional): 移动步长。默认是0.5。
        rotation_angle_degrees (float, optional): 旋转角度（度）。默认是5度。

    Returns:
        None
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载对象
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # 获取和设置摄像机
    cam = bpy.data.objects.get("Camera")
    if not cam:
        raise ValueError("场景中不存在名为 'Camera' 的相机对象。")
    cam.location = camera_initial_location
    cam.data.lens = 35
    cam.data.sensor_width = 35
    cam.data.sensor_height = 20

    # 设置相机朝向
    direction_vector = mathutils.Vector(camera_direction).normalized()
    rot_quat = direction_vector.to_track_quat('-Z', 'Y')
    # rot_quat = direction_vector.to_track_quat('Z', '-Y')
    cam.rotation_euler = rot_quat.to_euler()
    angle_radians = math.radians(
        camera_initial_rotation
    )  # camera_initial_rotation为负，表示初始状态逆时针旋转指定度数；camera_initial_rotation为正，表示初始状态顺时针旋转指定度数
    cam.rotation_euler.y += angle_radians  # 逆时针旋转为正方向

    # 提取元数据
    metadata_extractor = MetadataExtractor(object_path=object_file, scene=bpy.context.scene, bdata=bpy.data)
    metadata = metadata_extractor.get_metadata()

    # 删除非网格对象
    if object_file.lower().endswith(".usdz"):
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # 应用随机颜色
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # 保存元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    # 规范化场景
    normalize_scene()

    # 随机化灯光
    randomize_lighting()

    all_animations = get_animation_list_from_file_path()
    for track_name in all_animations:
        os.makedirs(os.path.join(args.output_dir, track_name), exist_ok=True)

    # print('all_animations', all_animations)

    if all_animations:
        for track_name in all_animations.copy():
            cam.location = camera_initial_location
            direction_vector = mathutils.Vector(camera_direction).normalized()
            rot_quat = direction_vector.to_track_quat('-Z', 'Y')
            cam.rotation_euler = rot_quat.to_euler()
            angle_radians = math.radians(camera_initial_rotation)
            cam.rotation_euler.y += angle_radians  # 逆时针旋转为正方向

            current_animation_output_dir = os.path.join(output_dir, track_name)

            frame_start, frame_end = 10000000, -10000000
            for obj in bpy.data.objects:
                if obj.animation_data:
                    track = obj.animation_data.nla_tracks.get(track_name)
                    if track and track.strips:
                        frame_start = min(frame_start, min([strip.frame_start for strip in track.strips]))
                        frame_end = max(frame_end, max([strip.frame_end for strip in track.strips]))

            frame_start, frame_end = int(frame_start), int(frame_end)

            # if frame_end - frame_start < 10:
            #     os.removedirs(current_animation_output_dir)
            #     return

            for obj in bpy.data.objects:
                if obj.animation_data:
                    for obj_track in obj.animation_data.nla_tracks:
                        obj_track.mute = obj_track.name != track_name

            # for frame_idx in range(frame_start, frame_end):
            for frame_idx in range(frame_start, num_renders + frame_start):
                cycle_length = frame_end - frame_start + 1
                current_frame = ((frame_idx - frame_start) % cycle_length) + frame_start
                scene.frame_set(current_frame)
                render_path = os.path.join(current_animation_output_dir, f"0staticshot_{frame_idx:03d}.png")
                scene.render.filepath = render_path
                bpy.ops.render.render(write_still=True)

                # 保存RT矩阵，相机坐标系和世界坐标系的外参
                rt_matrix_camera, rt_matrix_world = get_3x4_RT_matrix_from_blender(cam)
                rt_matrix_path_camera = os.path.join(current_animation_output_dir, f"0staticshot_{frame_idx:03d}_camera.npy")
                rt_matrix_path_world = os.path.join(current_animation_output_dir, f"0staticshot_{frame_idx:03d}_world.npy")
                np.save(rt_matrix_path_camera, rt_matrix_camera)
                np.save(rt_matrix_path_world, rt_matrix_world)

            # 渲染循环
            for i, frame_idx in zip(range(num_renders), range(frame_start, num_renders + frame_start)):

                cycle_length = frame_end - frame_start + 1
                current_frame = ((frame_idx - frame_start) % cycle_length) + frame_start
                scene.frame_set(current_frame)
                render_path = os.path.join(current_animation_output_dir, f"1dynamic_{i:03d}_{frame_idx:03d}.png")
                scene.render.filepath = render_path
                bpy.ops.render.render(write_still=True)

                # 保存RT矩阵，相机坐标系和世界坐标系的外参
                rt_matrix_camera, rt_matrix_world = get_3x4_RT_matrix_from_blender(cam)
                rt_matrix_path_camera = os.path.join(current_animation_output_dir, f"1dynamic_{i:03d}_{frame_idx:03d}_camera.npy")
                rt_matrix_path_world = os.path.join(current_animation_output_dir, f"1dynamic_{i:03d}_{frame_idx:03d}_world.npy")
                np.save(rt_matrix_path_camera, rt_matrix_camera)
                np.save(rt_matrix_path_world, rt_matrix_world)

                # 执行移动
                if i < len(movement_sequence):
                    move_cmd = movement_sequence[i]
                    if move_cmd == 'up':
                        move_camera_up(cam, step_size=movement_step_size)
                    elif move_cmd == 'down':
                        move_camera_down(cam, step_size=movement_step_size)
                    elif move_cmd == 'left':
                        move_camera_left(cam, step_size=movement_step_size)
                    elif move_cmd == 'right':
                        move_camera_right(cam, step_size=movement_step_size)
                    elif move_cmd == 'forward':
                        move_camera_forward(cam, step_size=movement_step_size)
                    elif move_cmd == 'backward':
                        move_camera_backward(cam, step_size=movement_step_size)

                # 执行旋转
                if i < len(rotation_sequence):
                    rotate_cmd = rotation_sequence[i]
                    if rotate_cmd == 'rotate_up':
                        rotate_camera_up(cam, angle_degrees=rotation_angle_degrees)
                    elif rotate_cmd == 'rotate_down':
                        rotate_camera_down(cam, angle_degrees=rotation_angle_degrees)
                    elif rotate_cmd == 'rotate_left':
                        rotate_camera_left(cam, angle_degrees=rotation_angle_degrees)
                    elif rotate_cmd == 'rotate_right':
                        rotate_camera_right(cam, angle_degrees=rotation_angle_degrees)
                    elif rotate_cmd == 'rotate_clockwise':
                        rotate_camera_clockwise(cam, angle_degrees=rotation_angle_degrees)
                    elif rotate_cmd == 'rotate_counterclockwise':
                        rotate_camera_counterclockwise(cam, angle_degrees=rotation_angle_degrees)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=str, required=True, help="Path to the object file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the rendered images and metadata will be saved.")
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--only_northern_hemisphere", action="store_true", help="Only render the northern hemisphere of the object.", default=False)
    parser.add_argument("--num_renders", type=int, default=12, help="Number of renders to save of the object.")
    parser.add_argument("--camera_initial_location", type=str, default="(0, 0, 0)", help="")
    parser.add_argument("--camera_direction", type=str, default="(0, 0, 0)", help="")
    parser.add_argument("--camera_initial_rotation", type=float, default=0.0, help="")
    parser.add_argument("--movement_sequence", type=str, default="[]", help="")
    parser.add_argument("--rotation_sequence", type=str, default="[]", help="")
    parser.add_argument("--movement_step_size", type=float, default=0.05, help="")
    parser.add_argument("--rotation_angle_degrees", type=float, default=0.5, help="")
    parser.add_argument("--random_or_custom", type=str, required=True, choices=["random", "custom"])

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 672
    render.resolution_y = 384
    render.resolution_percentage = 100

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # print('args.random_or_custom: ', args.random_or_custom)

    if args.random_or_custom == "custom":
        render_with_movement_and_rotation(
            object_file=args.object_path,
            num_renders=args.num_renders,
            output_dir=args.output_dir,
            camera_initial_location=literal_eval(args.camera_initial_location),  # 相机起始位置，(X, Y, Z), (右, 上, 后)
            camera_direction=literal_eval(args.camera_direction),  # 相机朝向方向向量
            movement_sequence=literal_eval(args.movement_sequence),  # 移动指令序列
            rotation_sequence=literal_eval(args.rotation_sequence),  # 旋转指令序列
            movement_step_size=args.movement_step_size,  # 每步移动0.05单位
            rotation_angle_degrees=args.rotation_angle_degrees,  # 每步旋转5度
            camera_initial_rotation=args.camera_initial_rotation,
        )
    elif args.random_or_custom == "random":
        render_object_random(
            object_file=args.object_path,
            num_renders=args.num_renders,
            only_northern_hemisphere=args.only_northern_hemisphere,
            output_dir=args.output_dir,
        )

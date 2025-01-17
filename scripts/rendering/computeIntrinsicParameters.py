import bpy

scene = bpy.context.scene
cam = bpy.data.objects.get("Camera")
if not cam:
    raise ValueError("场景中不存在名为 'Camera' 的相机对象。")

cam.data.lens = 35
# cam.data.sensor_width = 32
# cam.data.sensor_height = 16

scene.render.resolution_x = 720
scene.render.resolution_y = 384
scene.render.resolution_percentage = 100


# 获取渲染设置中的分辨率
resolution_x = scene.render.resolution_x  # 水平分辨率，像素
resolution_y = scene.render.resolution_y  # 垂直分辨率，像素
resolution_percentage = scene.render.resolution_percentage / 100  # 分辨率百分比
scaled_resolution_x = resolution_x * resolution_percentage  # 缩放后的水平分辨率
scaled_resolution_y = resolution_y * resolution_percentage  # 缩放后的垂直分辨率

# 获取传感器的实际尺寸
sensor_width = cam.data.sensor_width  # 传感器宽度，毫米
sensor_height = cam.data.sensor_height  # 传感器高度，毫米

# 计算焦距对应的像素值
# fx = (cam.data.lens * scaled_resolution_x) / sensor_width
# fy = (cam.data.lens * scaled_resolution_y) / sensor_height

fx = cam.data.lens / sensor_width
fy = cam.data.lens / sensor_height

# 计算主点（通常位于图像中心）
cx = 1 / 2
cy = 1 / 2

# 打印结果
print(f"fx: {fx}")
print(f"fy: {fy}")
print(f"cx: {cx}")
print(f"cy: {cy}")
print(f'sensor_width: {sensor_width}')
print(f'sensor_height: {sensor_height}')

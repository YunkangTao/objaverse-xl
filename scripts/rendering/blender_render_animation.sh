/home/yunkang/objaverse-xl/scripts/rendering/blender-3.2.2-linux-x64/blender -noaudio -b -P blender_render_animation.py \
-- \
--object_path /home/yunkang/objaverse-xl/data/objaverse-animation-HQ/000_top10/0a0b504f51a94d95a2d492d3c372ebe5.glb \
--output_dir /home/yunkang/objaverse-xl/data/objaverse-animation-HQ_render_results/000_top10/0a0b504f51a94d95a2d492d3c372ebe5/000_render_animation \
--only_northern_hemisphere \
--engine CYCLES \
--num_renders 12 \
--max_n_frames 32 \
--uniform_azimuth --render
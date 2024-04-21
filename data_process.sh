#!/bin/bash

# Ask for variable inputs
read -p "Enter Video Name: " Video_Name
read -p "Enter Video Path: " Video_Path
read -p "Enter Device ID: " Device_id
read -p "Enter Reference Image Path: " Reference_Image_Path
read -p "Enter Result Name: " Result_Name

export PATH=/home/lfz/workspace/blender-3.6.0-linux-x64:$PATH
mkdir -p driving_videos/${Video_Name}/images
ffmpeg -i ${Video_Path} -c:v png driving_videos/${Video_Name}/images/%04d.png
CUDA_VISIBLE_DEVICES=${Device_id} python -m scripts.data_processors.smpl.generate_smpls --reference_imgs_folder ${Reference_Image_Path} --driving_video_path driving_videos/${Video_Name} --device 0
CUDA_VISIBLE_DEVICES=${Device_id} python -m scripts.data_processors.smpl.smpl_transfer --reference_path ${Reference_Image_Path}/smpl_results/img.npy --driving_path driving_videos/${Video_Name} --output_folder ${Result_Name} --figure_transfer --view_transfer
blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render_condition_maps.py -- --driving_path ${Result_Name}/smpl_results --reference_path ${Reference_Image_Path}/images/img.png
python -m scripts.data_processors.dwpose.generate_dwpose --input ${Result_Name}/normal --output ${Result_Name}/dwpose

echo "Workflow completed!"

# Enter Video Name: Video_4
# Enter Video Path: my_data/clip.mp4
# Enter Device ID: 1
# Enter Reference Image Path: rj_imgs
# Enter Result Name: clip_rj
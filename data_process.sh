#!/bin/bash

Video_Name="Video_7"
Video_Path="my_data/clip.mp4"
Device_id="1"
Reference_Image_Path="fat"
Result_Name="fat_test_smpl_mon"
SMPL_param=-7

export PATH=/home/lfz/workspace/blender-3.6.0-linux-x64:$PATH
# mkdir -p driving_videos/${Video_Name}/images
# ffmpeg -i ${Video_Path} -c:v png driving_videos/${Video_Name}/images/%04d.png

# CUDA_VISIBLE_DEVICES=${Device_id} python -m scripts.data_processors.smpl.generate_smpls --reference_imgs_folder reference_imgs_set/${Reference_Image_Path} --driving_video_path driving_videos/${Video_Name} --device 0 # --figure_scale ${SMPL_param}
#blender --background --python scripts/data_processors/smpl/smooth_smpls.py -- --smpls_group_path driving_videos/${Video_Name}/smpl_results/smpls_group.npz --smoothed_result_path driving_videos/${Video_Name}/smpl_results/smpls_group.npz
CUDA_VISIBLE_DEVICES=${Device_id} python -m scripts.data_processors.smpl.smpl_transfer --reference_path reference_imgs_set/${Reference_Image_Path}/smpl_results/img.npy --driving_path driving_videos/${Video_Name} --output_folder transfer_set/${Result_Name} --view_transfer  # Here ignore the --figure_transfer 
blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render_condition_maps.py -- --driving_path transfer_set/${Result_Name}/smpl_results --reference_path reference_imgs_set/${Reference_Image_Path}/images/img.png  
python -m scripts.data_processors.dwpose.generate_dwpose --input transfer_set/${Result_Name}/normal --output transfer_set/${Result_Name}/dwpose
python inference.py --config configs/inference/inference.yaml
echo "Workflow completed!"


# CUDA_VISIBLE_DEVICES=${Device_id} python -m scripts.data_processors.smpl.smpl_transfer --reference_path reference_imgs_set/${Reference_Image_Path}/smpl_results/img.npy --driving_path driving_videos/${Video_Name} --output_folder transfer_set/${Result_Name} --view_transfer  # Here ignore the --figure_transfer 

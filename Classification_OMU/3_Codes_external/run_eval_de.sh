#!/bin/bash
SECONDS=0
export CUDA_VISIBLE_DEVICES=0,1,2,3:

## Setting ##
WORKERS=32 # 32, cpu cores
IMG_SIZE=300 # 300, size of Center crop 
BATCH_KEY=2 # batch size for key-slice selector
BATCH_3D=1 # batch size for 3D-CNN and 3D-CAM
KEY=0.5
# KEY=0

# Pre-processing of the ID change, making for 5-folds, extraction from dicom to jpg, and extraction of csv for dicom header  
#python 1_pre-processing.py 

# Inference for key slice

python "$(dirname $0)/2_external_keyslice_de.py" --image_size ${IMG_SIZE} --kernel_type cbct_90_denoising --batch_size ${BATCH_KEY} --num_workers ${WORKERS} --model b0

# Inference for 3D CNN

python "$(dirname $0)/3_external_eval_de.py" --crop_size ${IMG_SIZE} --save_name cbct_90_denoising_resnet18_batch16_aug0.5_key0.5 --batch_size ${BATCH_3D} --key_thresh ${KEY}


# # Generate for Grad-Cam

 python "$(dirname $0)/4_gradcam_de.py" --save_name cbct_90_denoising_resnet18_batch16_aug0.5_key0.5 --batch_size ${BATCH_3D} --fold 0  --key_thresh ${KEY}

duration=$SECONDS
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

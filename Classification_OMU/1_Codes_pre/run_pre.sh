#!/bin/bash
SECONDS=0
export CUDA_VISIBLE_DEVICES=0,1,2,3:

## Setting ##
MODEL=b0  # b0 has best performance, b0-b7, se_resnext50_32x4d, se_resnext101_32x4d
DEBUG=F # T, F
WORKERS=32 # 32, Running cpu cores
INIT_LR=1e-5 # 1e-5,initial learning rate
N_EPOCHS=30 # 30, Epochs for molels of key slice selector
SAVE_NAME=cbct_90_noise
BATCH=256 # 256, Train and test batch size
IMG_SIZE=300 # 300, Center crop image size

## Pre-processing of the ID change, making for 5-folds, extraction from dicom to jpg, and extraction of csv for dicom header  

# Training for key slice selector
for FOLD in 0 1 2 3 4
do
    python "$(dirname $0)/2_train_keyslice.py" --num_workers ${WORKERS} --n_epochs ${N_EPOCHS} --init_lr ${INIT_LR} --fold ${FOLD} --batch_size ${BATCH} --image_size ${IMG_SIZE} --kernel_type ${SAVE_NAME} --DEBUG ${DEBUG} --model ${MODEL}
done

## Inference for key slice
for FOLD in 0 1 2 3 4
do 
    WEIGHT="$(dirname $0)"/key_weights/${SAVE_NAME}/${SAVE_NAME}_best_fold${FOLD}.pth
    echo $WEIGHT
    python "$(dirname $0)/3_inference_keyslice.py" --image_size ${IMG_SIZE} --kernel_type ${SAVE_NAME} --fold ${FOLD} --batch_size ${BATCH} --weight_dir ${WEIGHT} --num_workers ${WORKERS} --model ${MODEL} --DEBUG ${DEBUG}
done

duration=$SECONDS
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

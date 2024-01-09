#!/bin/bash
SECONDS=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7:
export CUDA_VISIBLE_DEVICES=0,1,2:

#DATA_DIR=/home/mars/workspace/cy_workspace/OMU/Classfication_OMU/input/ # Path of data
SEED=711 # Random SEED
DEBUG=F # T, F
WORKERS=32 # cpu cores
N_EPOCHS=50 # 50, training epochs
INIT_LR=1e-5 #1e-5, initial learning rate
KEY=0.5 # 0.5, Threshold of probablity in slice selector 
MODEL=resnet18 #densenet121, resnet18, resnet34, resnet50
CROP_SIZE=300 # 300, Center crop image size
Z_RESIZE=160 # 160 # resize of z axis
RESIZE=160 # 160 # resize of x, y axis
#DF_DIR=/1_Codes_pre/key_logs/cbct_90_denoising/ # path for dataframe of keyslice selector

DATE=cbct_90_denoising
AUGMENT=0.5 # Probablity for augmentation 0~1
BATCH=16 #16

SAVE_NAME=${DATE}_${MODEL}_batch${BATCH}_aug${AUGMENT}_key${KEY}

## Training for each fold
 for FOLD in 0 1 2 3 4
 do
     python "$(dirname $0)/train_de.py" --num_workers ${WORKERS} --n_epochs ${N_EPOCHS} --init_lr ${INIT_LR} --fold ${FOLD} --batch_size ${BATCH} --crop_size ${CROP_SIZE} --resize ${RESIZE} --z_resize ${Z_RESIZE} --save_name ${SAVE_NAME} --DEBUG ${DEBUG} --SEED ${SEED} --model ${MODEL} --key_thresh ${KEY} --augment ${AUGMENT}
 echo "${Fold} is done"
 done

# Inference for 5-folds
echo "$5-folds evaluation"
python "$(dirname $0)/eval_de.py"  --crop_size ${CROP_SIZE} --resize ${RESIZE} --z_resize ${Z_RESIZE} --DEBUG ${DEBUG} --batch_size ${BATCH} --num_workers ${WORKERS} --model ${MODEL} --key_thresh ${KEY} --save_name ${SAVE_NAME}

## Generate for 3D-CAM 
# echo "Generate for 3D-CAM"
# python 3D_Cam.py --save_name ${SAVE_NAME} --df_dir ${DF_DIR} --crop_size ${CROP_SIZE} --resize ${RESIZE} --z_resize ${Z_RESIZE} --DEBUG ${DEBUG} --fold 2

# duration=$SECONDS

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

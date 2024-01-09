#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3:
# python ISTA-U-Net-main/ista_unet/train_ct.py #train ISTA-U-NET
python "$(dirname $0)/"ISTA-U-Net-main/ista_unet/evaluation.py
# train (noise)
sh $(dirname $(realpath $0))/1_Codes_pre/run_pre.sh
sh $(dirname $(realpath $0))/2_Codes_3D/run_train.sh

# train (denoising)
sh $(dirname $(realpath $0))/1_Codes_pre/run_pre_de.sh
sh $(dirname $(realpath $0))/2_Codes_3D/run_train_de.sh

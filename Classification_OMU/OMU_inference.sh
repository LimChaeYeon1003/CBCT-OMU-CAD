#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3:


#inference 3D CNN (external datset)
#inference Noise
sh $(dirname $(realpath $0))/3_Codes_external/run_eval.sh
#inference Denoising
sh $(dirname $(realpath $0))/3_Codes_external/run_eval_de.sh

#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python -u main_adaptive.py "configs/strider/strider_R50_4b_random.yaml" --resume "out/strider/strider_R50_4b_random/checkpoint.pth.tar" --evaluate
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_8b_random.yaml" --resume "out/strider/strider_R50_8b_random/checkpoint.pth.tar" --evaluate
#CUDA_VISIBLE_DEVICES=2 python -u main_adaptive.py "configs/strider/strider_R50_12b_random.yaml" --resume "out/strider/strider_R50_12b_random/checkpoint.pth.tar" --evaluate
CUDA_VISIBLE_DEVICES=3 python -u main_adaptive.py "configs/strider/strider_R50_16b_random.yaml" --resume "out/strider/strider_R50_16b_random/checkpoint.pth.tar" --evaluate


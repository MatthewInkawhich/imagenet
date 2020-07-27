#!/bin/bash

### Baselines
#python main.py "configs/resnet50.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_fpn_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_lrr-2-4-8_control.yaml" --workers 8

### Strider
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py -b 64 "configs/strider/strider_play.yaml"
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_16b_random.yaml" --workers 8 -b 64
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --workers 8 -b 64

# Random strides
#python -u main_adaptive.py "configs/strider/strider_R50_4b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_8b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_12b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_16b_random.yaml" --workers 8

#python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_C_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45


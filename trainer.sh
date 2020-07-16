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

# Random strides
python -u main_adaptive.py "configs/strider/strider_R50_4b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_8b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_12b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_16b_random.yaml" --workers 8

#python -u main_adaptive.py "configs/strider/strider_R50_16b_manual.yaml" --workers 8

#python -u main_adaptive.py "configs/strider/strider_R50_16b.yaml" --workers 8

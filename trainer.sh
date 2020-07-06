#!/bin/bash

### Baselines
#python main.py "configs/resnet50.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_fpn_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_lrr-2-4-8_control.yaml" --workers 8

### Strider
CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py -b 64 "configs/strider/strider_play.yaml"

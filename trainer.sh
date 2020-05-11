#!/bin/bash

### Baselines
#python main.py "configs/resnet50.yaml" --workers 8
#python main.py "configs/strider_R50_control.yaml" --workers 8
#python main.py "configs/strider_R50_fpn_control.yaml" --workers 8

### Strider v1
#python main.py "configs/striderv1_R50_vanilla.yaml" --workers 8
python main.py "configs/striderv2_R50_vanilla.yaml" --workers 8

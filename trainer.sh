#!/bin/bash

### Baselines
#python main.py "configs/resnet50.yaml" --workers 8
#python main.py "configs/strider_R50_control.yaml" --workers 8
#python main.py "configs/strider_R50_fpn_control.yaml" --workers 8

### Strider v2
python -u main.py --workers 8 "configs/striderv2-5_R50_fpn.yaml"

#!/bin/bash

### Baselines
#python main.py "configs/resnet50.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_fpn_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_lrr-2-4-8_control.yaml" --workers 8

### Strider
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/striderr/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 10 -b 2
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/striderr/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 10 -b 2

# Full runs 
#python -u main_adaptive.py "configs/striderr/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 1 --stage2-epochs-per-cycle 1 -b 8
#python -u main_adaptive.py "configs/striderr/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 1 --stage2-epochs-per-cycle 0 -b 128

# Static S1
#python -u main_adaptive.py "configs/striderr/strider_R50_ABCD_static.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 0

# Stage1
#python -u main_adaptive.py "configs/striderr/strider_R50_A_random.yaml" --resume-stage2 "out/striderr/strider_R50_ABCD_static/model_best.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 30 --stage2-epochs-per-cycle 0 --lr1 0.01 --lr1-decay-every 10
#python -u main_adaptive.py "configs/striderr/strider_R50_AB_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 0
#python -u main_adaptive.py "configs/striderr/strider_R50_ABC_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 0

# Create selector_truth
#python -u main_adaptive.py "configs/striderr/strider_R50_A_random.yaml" --resume-stage2 "out/striderr/strider_R50_A_random/C0_post_S1.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 0 
#python -u main_adaptive.py "configs/striderr/strider_R50_AB_random.yaml" --resume-stage2 "out/striderr/strider_R50_AB_random/C0_post_S1.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 0 
#python -u main_adaptive.py "configs/striderr/strider_R50_ABCD_random.yaml" --resume-stage2 "out/striderr/strider_R50_ABCD_random/C0_post_S1.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 0 -b 256

# Stage 2
#python -u main_adaptive.py "configs/striderr/strider_R50_A_random.yaml" --resume-stage2 "out/striderr/strider_R50_A_random/C0_post_S1.pth.tar" --load-selector-truth "out/striderr/strider_R50_A_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 30 --lr2 0.001 --run-name "A_plain"
#python -u main_adaptive.py "configs/striderr/strider_R50_AB_random.yaml" --resume-stage2 "out/striderr/strider_R50_AB_random/C0_post_S1.pth.tar" --load-selector-truth "out/striderr/strider_R50_AB_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.001
#python -u main_adaptive.py "configs/striderr/strider_R50_ABCD_random.yaml" --resume-stage2 "out/striderr/strider_R50_ABCD_random/C0_post_S1.pth.tar" --load-selector-truth "out/striderr/strider_R50_ABCD_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.001 --run-name "test"

#python -u main_adaptive.py "configs/striderr/strider_R50_AB_random.yaml" --resume-stage2 "out/striderr/strider_R50_AB_random/C0_post_S1.pth.tar" --load-selector-truth "out/striderr/strider_R50_AB_random/selector_truth.pth.tar" --load-selector-targets "out/striderr/strider_R50_AB_random/selector_targets.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.001 -b 8


#python -u main_adaptive.py "configs/striderr/strider_R50_A_random.yaml" --resume-stage2 "out/striderr/strider_R50_A_random/C0_post_S1.pth.tar" --load-selector-truth "out/striderr/strider_R50_A_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.001 --lr2-decay-every 5 --evaluate-freq 2 --run-name "A_plain"
python -u main_adaptive.py "configs/striderr/strider_R50_A_random.yaml" --resume-stage2 "out/striderr/strider_R50_A_random/C0_post_S1.pth.tar" --load-selector-truth "out/striderr/strider_R50_A_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --ls-alpha 0.1 --lr2 0.001 --lr2-decay-every 5 --evaluate-freq 2 --run-name "A_ls0.1"

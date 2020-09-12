#!/bin/bash

### Baselines
#python main.py "configs/resnet50.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_fpn_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_lrr-2-4-8_control.yaml" --workers 8

### Strider
#python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_C_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45

#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 150 --stage2-epochs-per-cycle 0
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume "out/strider/strider_R50_lrr-2-4_ABCD_random/C0_post_S1.pth.tar"



#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_play.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45 -b 4

#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume-stage2 "out/strider/strider_R50_A_random/C0_post_S1.pth.tar" -b 2
#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume-stage2 "out/strider/strider_R50_ABCD_random/C0_post_S1.pth.tar"


#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume-stage2 "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider/strider_R50_B_random/selector_truth.pth.tar" -b 2
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume-stage2 "out/strider/strider_R50_ABCD_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider/strider_R50_B_random/selector_truth.pth.tar" -b 2

python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume-stage2 "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider/strider_R50_B_random/selector_truth.pth.tar" -b 8
#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume-stage2 "out/strider/strider_R50_ABCD_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider/strider_R50_B_random/selector_truth.pth.tar" -b 8

#python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume-stage2 "out/strider/strider_R50_B_random/C0_post_S1.pth.tar"


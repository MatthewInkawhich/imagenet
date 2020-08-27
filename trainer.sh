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
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --workers 8 -b 64 --resume-stage2 "out/strider/strider_R50_A_random/model_best.pth.tar"
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --workers 8 --resume-stage2 "out/strider/strider_R50_A_random/model_best_original.pth.tar" --lr2 0.1 #SGD + bn.eval

# Random strides
#python -u main_adaptive.py "configs/strider/strider_R50_4b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_8b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_12b_random.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_16b_random.yaml" --workers 8

#python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_C_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45
#CUDA_VISIBLE_DEVICES=1,2,3 python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --workers 6 -b 192 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 45

#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 150 --stage2-epochs-per-cycle 0
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume "out/strider/strider_R50_lrr-2-4_ABCD_random/C0_post_S1.pth.tar"


#python -u main_adaptive.py "configs/strider/strider_R50_C_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 1 --resume "out/strider/strider_R50_C_random/C0_post_S2.pth.tar" --lr2 0.001

#python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 60 --stage2-epochs-per-cycle 0 --resume "out/strider/strider_R50_D_random/C0_post_S1.pth.tar" --lr1 0.0001
#python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 45 --resume "out/strider/strider_R50_D_random/C0_post_S1.pth.tar"

#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 60 --stage2-epochs-per-cycle 45 --resume "out/strider/strider_R50_ABCD_random/C0_post_S1.pth.tar" --lr1 0.0001

python -u main_adaptive.py "configs/strider/strider_R50_control_150.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 150 --stage2-epochs-per-cycle 0

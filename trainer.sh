#!/bin/bash

### Baselines
#python main.py "configs/resnet50.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_fpn_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2_control.yaml" --workers 8
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_control.yaml" --workers 8
#python -u main.py "configs/strider/strider_R50_lrr-2-4-8_control.yaml" --workers 8

### Strider
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 10 -b 2
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive2.py "configs/strider2/strider_R50_ABCD_random.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 10 -b 2

# Static model
#python -u main_adaptive2.py "configs/strider2/strider_R50_ABCD_static.yaml" --workers 8 --cycles 1 --stage1-epochs-per-cycle 90 --stage2-epochs-per-cycle 0

# Create selector_truth
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 0
#python -u main_adaptive2.py "configs/strider2/strider_R50_ABCD_random.yaml" --resume-old "out/strider2/strider_R50_ABCD_random/C0_post_S1.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10

# Stage 2
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.001 --run-name "minover_lr001"
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.0001 --run-name "minover_lr0001"
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.01 --run-name "minover_lr01"
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.001 --run-name "focal"
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --lr2 0.001 --run-name "weighted_sampler"
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 30 --lr2 0.001 --lr2-decay-every 10 --run-name "weighted_sampler2"
python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 30 --lr2 0.001 --lr2-decay-every 10 --gamma 2.0 --run-name "weighted_sampler_focal"

#echo "eta1.0"; echo; echo; echo;
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --eta 1.25 --run-name "eta1.25"
#echo "eta1.25"; echo; echo; echo;
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --eta 1.5 --run-name "eta1.5"
#echo "eta1.5"; echo; echo; echo;
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --eta 1.75 --run-name "eta1.75"
#echo "eta1.75"; echo; echo; echo;
#python -u main_adaptive2.py "configs/strider2/strider_R50_B_random.yaml" --resume-old "out/strider2/strider_R50_B_random/C0_post_S1.pth.tar" --load-selector-truth "out/strider2/strider_R50_B_random/selector_truth.pth.tar" --workers 8 --cycles 1 --stage1-epochs-per-cycle 0 --stage2-epochs-per-cycle 10 --eta 2.0 --run-name "eta2.0"
#echo "eta2.0"; echo; echo; echo;

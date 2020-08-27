#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python -u main_adaptive.py "configs/strider/strider_R50_4b_random.yaml" --resume "out/strider/strider_R50_4b_random/checkpoint.pth.tar" --evaluate
#CUDA_VISIBLE_DEVICES=1 python -u main_adaptive.py "configs/strider/strider_R50_8b_random.yaml" --resume "out/strider/strider_R50_8b_random/checkpoint.pth.tar" --evaluate
#CUDA_VISIBLE_DEVICES=2 python -u main_adaptive.py "configs/strider/strider_R50_12b_random.yaml" --resume "out/strider/strider_R50_12b_random/checkpoint.pth.tar" --evaluate
#CUDA_VISIBLE_DEVICES=3 python -u main_adaptive.py "configs/strider/strider_R50_16b_random.yaml" --resume "out/strider/strider_R50_16b_random/checkpoint.pth.tar" --evaluate


#python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --resume "out/strider/strider_R50_A_random/C0_post_S1.pth.tar" --evaluate --random-eval
#python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --resume "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --evaluate --random-eval
#python -u main_adaptive.py "configs/strider/strider_R50_C_random.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --random-eval
#python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --resume "out/strider/strider_R50_D_random/C0_post_S1.pth.tar" --evaluate --random-eval
#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --resume "out/strider/strider_R50_ABCD_random_bak/C0_post_S1.pth.tar" --evaluate --random-eval
#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --resume "out/strider/strider_R50_ABCD_random/C0_post_S1.pth.tar" --evaluate --random-eval
#echo "ABCD+60"
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_ABCD_random.yaml" --resume "out/strider/strider_R50_lrr-2-4_ABCD_random/C0_post_S1.pth.tar" --evaluate --random-eval
#echo "ABCD_lrr-2-4"



#python -u main_adaptive.py "configs/strider/manual/A_0.yaml" --resume "out/strider/strider_R50_A_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "A_0"
#python -u main_adaptive.py "configs/strider/manual/A_1.yaml" --resume "out/strider/strider_R50_A_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "A_1"
#python -u main_adaptive.py "configs/strider/manual/A_2.yaml" --resume "out/strider/strider_R50_A_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "A_2"
#python -u main_adaptive.py "configs/strider/manual/A_3.yaml" --resume "out/strider/strider_R50_A_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "A_3"

#python -u main_adaptive.py "configs/strider/manual/B_0.yaml" --resume "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "B_0"
#python -u main_adaptive.py "configs/strider/manual/B_1.yaml" --resume "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "B_1"
#python -u main_adaptive.py "configs/strider/manual/B_2.yaml" --resume "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "B_2"
#python -u main_adaptive.py "configs/strider/manual/B_3.yaml" --resume "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "B_3"

#python -u main_adaptive.py "configs/strider/manual/C_0.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "C_0"
#python -u main_adaptive.py "configs/strider/manual/C_1.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "C_1"
#python -u main_adaptive.py "configs/strider/manual/C_2.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "C_2"
#python -u main_adaptive.py "configs/strider/manual/C_3.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "C_3"
#python -u main_adaptive.py "configs/strider/manual/C_4.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "C_4"
#python -u main_adaptive.py "configs/strider/manual/C_5.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "C_5"
#python -u main_adaptive.py "configs/strider/manual/C_6.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "C_6"

#python -u main_adaptive.py "configs/strider/manual/D_0.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "D_0"
#python -u main_adaptive.py "configs/strider/manual/D_1.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "D_1"
#python -u main_adaptive.py "configs/strider/manual/D_2.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "D_2"
#python -u main_adaptive.py "configs/strider/manual/D_3.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "D_3"
#python -u main_adaptive.py "configs/strider/manual/D_4.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "D_4"
#python -u main_adaptive.py "configs/strider/manual/D_5.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "D_5"
#python -u main_adaptive.py "configs/strider/manual/D_6.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --batch-eval
#echo "D_6"

#python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --resume "out/strider/strider_R50_A_random/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
python -u main_adaptive.py "configs/strider/strider_R50_A_random.yaml" --resume "out/strider/strider_R50_A_random/C0_post_S2.pth.tar" --evaluate --oracle --batch-eval
#echo "A"
#python -u main_adaptive.py "configs/strider/strider_R50_B_random.yaml" --resume "out/strider/strider_R50_B_random/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
#echo "B"
#python -u main_adaptive.py "configs/strider/strider_R50_C_random.yaml" --resume "out/strider/strider_R50_C_random/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
#echo "C"
#python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --resume "out/strider/strider_R50_D_random_bak/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
#echo "D"
#python -u main_adaptive.py "configs/strider/strider_R50_D_random.yaml" --resume "out/strider/strider_R50_D_random/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
#echo "D+60"
#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --resume "out/strider/strider_R50_ABCD_random_bak/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
#echo "ABCD"
#python -u main_adaptive.py "configs/strider/strider_R50_ABCD_random.yaml" --resume "out/strider/strider_R50_ABCD_random/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
#echo "ABCD+60"
#python -u main_adaptive.py "configs/strider/strider_R50_lrr-2-4_ABCD_random.yaml" --resume "out/strider/strider_R50_lrr-2-4_ABCD_random/C0_post_S1.pth.tar" --evaluate --oracle --batch-eval
#echo "ABCD_lrr-2-4+60"

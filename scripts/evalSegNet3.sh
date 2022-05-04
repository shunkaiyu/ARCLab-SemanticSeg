#!/bin/bash
dice_loss_factor=0.0
focal_loss_factor=0.0
fold="fold2_23_1"
save_test=True
checkpoint_dir="/home/arcseg/Desktop/Shunkai-working/results/smp_DeepLabV3+/cholec_fold2_23_1/368640_random_crop/dice_factor_0.0_focal_factor_0.0/bs_train12_val4/imsize_480x848_wd_0.00001_optim_Adam_lr1e-3_steps_4_gamma_0.1/e50_seed6210"
checkpoint_title="smp_DeepLabV3+_cholec_bs12lr0.001e50_checkpoint"
checkpoint="${checkpoint_dir}/${checkpoint_title}"
save_dir="../results/${fold}_dice_${dice_loss_factor}_focal_${focal_loss_factor}/${checkpoint_title}"
mkdir save_dir
data_dir = "../src/data/datasets/cholec_23_1"


python ../src/evalSegNet.py \
--save-dir $save_dir \
--saveTest $save_test \
--model $checkpoint \
--data_path "../src/data/datasets/cholec_23_1" \
|& tee -a "${save_dir}/eval.log"

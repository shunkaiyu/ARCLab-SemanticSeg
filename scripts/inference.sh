#!/bin/bash

model_dir="/home/arcseg/Desktop/Shunkai-working/results/smp_DeepLabV3+/cholec_fold2_12_3/368640_random_crop/dice_factor_0.1_focal_factor_0.1/bs_train12_val4/imsize_480x848_wd_0.00001_optim_Adam_lr1e-3_steps_4_gamma_0.1/e50_seed6210/smp_DeepLabV3+_cholec_bs12lr0.001e50_checkpoint"
imdir_path="/home/arcseg/Desktop/result_images/01_frame_16345_endo.png"
classes_path="../src/data/classes/cholecSegClasses.json"
save_dir="../results/"
fold="fold2_12_3"

python ../src/inference.py \
--model_dir $model_dir \
--imdir_path $imdir_path \
--classes_path $classes_path \
--save_dir "$save_dir/$fold/smp_DeepLabV3+_cholec_bs12lr0.001e50_checkpoint" \
|& tee -a "${save_dir}/eval.log"

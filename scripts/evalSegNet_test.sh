#!/bin/bash
model="smp_DeepLabV3+"

epochs=50
workers=0

train_batch_size=12 # [1, 11, 37]
val_batch_size=4 # [1, 3, 29]
full_res_validation="True"

lr=1e-3
optimizer="Adam"
wd=0.00001
lr_steps=4
step_gamma=0.1
dice_loss_factor=0.3333
focal_loss_factor=0.3333

resized_height=256 # for random crop
resized_width=256 # for random crop
cropSize=-1
crop_size=360

fold="cholec_test2"
dataset="cholec"
save_test=True
# checkpoint_dir="/home/arcseg/Desktop/Shunkai-working/results/smp_DeepLabV3+/cholec_test/480x848_test/dice_factor_0.1_focal_factor_0.1/bs_train4_val2/imsize_480x848_wd_0.00001_optim_Adam_lr1e-3_steps_4_gamma_0.1/e10_seed6210"
checkpoint_dir = "/home/arcseg/Desktop/Shunkai-working/results/smp_DeepLabV3+T/cholec_test/480x848_test/dice_factor_0.0_focal_factor_0.0/bs_train4_val2/imsize_480x848_wd_0.00001_optim_Adam_lr1e-3_steps_4_gamma_0.1/e10_seed6210"
checkpoint_title="smp_DeepLabV3+T_cholec_bs4lr0.001e10_checkpoint"
checkpoint="${checkpoint_dir}/${checkpoint_title}"
mkdir save_dir
save_dir="../results/${fold}/${checkpoint_title}"
data_dir = "../src/data/datasets/fold2_12_3"


python ../src/evalSegNet.py \
    --save-dir $save_dir \
    --saveTest $save_test \
    --model $checkpoint \
    --data_path "/home/arcseg/Desktop/Shunkai-working/src/data/datasets/cholec_test2" \
|& tee -a "${save_dir}/eval.log"

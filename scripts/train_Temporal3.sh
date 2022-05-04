#!/bin/bash

model="smp_DeepLabV3+T"

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
dice_loss_factor=0.0
focal_loss_factor=0.0

resized_height=480 # for random crop
resized_width=848 # for random crop
RandomCropSize=368640
cropSize=-1

dataset="cholec"
data_dir="../src/data/datasets/fold2_23_1"
json_path="../src/data/classes/cholecSegClasses.json"
#checkpoint_path="/home/arcseg/jonathan/ARCSeg/results/resnet18_unet/cholec/bs_32_lr1e-3_e100/resnet18_unet_cholec_bs32lr0.001e100_checkpoint"

display_samples="False"
save_samples="True"

use_high_level="True"
use_low_level="True"

	save_dir="../results/${model}_hl_${use_high_level}_ll_${use_low_level}/${dataset}_fold2_23_1/${RandomCropSize}_random_crop/dice_factor_${dice_loss_factor}_focal_factor_${focal_loss_factor}/bs_train${train_batch_size}_val${val_batch_size}/imsize_${resized_height}x${resized_width}_wd_${wd}_optim_${optimizer}_lr${lr}_steps_${lr_steps}_gamma_${step_gamma}/e${epochs}_seed6210"
	seg_save_dir="${save_dir}/seg_results"

	mkdir -p $save_dir

	python ../src/trainSegNet2T.py \
	    --model $model \
	    --workers $workers \
	    --trainBatchSize $train_batch_size \
	    --valBatchSize $val_batch_size \
	    --full_res_validation $full_res_validation \
	    --resizedHeight $resized_height \
	    --resizedWidth $resized_width \
	    --cropSize $cropSize \
	    --lr $lr \
	    --dice_loss_factor $dice_loss_factor \
	    --focal_loss_factor $focal_loss_factor \
	    --epochs $epochs \
	    --lr_steps $lr_steps \
	    --step_gamma $step_gamma \
	    --optimizer $optimizer \
	    --wd $wd \
	    --dataset $dataset \
	    --display_samples $display_samples \
	    --save_samples $save_samples \
	    --data_dir $data_dir \
	    --json_path $json_path \
	    --save_dir $save_dir \
	    --seg_save_dir $seg_save_dir \
        --use_high_level $use_high_level \
		--use_low_level $use_low_level \
	    |& tee -a "${save_dir}/debug.log"

use_high_level="False"
use_low_level="True"

	save_dir="../results/${model}_hl_${use_high_level}_ll_${use_low_level}/${dataset}_fold2_23_1/${RandomCropSize}_random_crop/dice_factor_${dice_loss_factor}_focal_factor_${focal_loss_factor}/bs_train${train_batch_size}_val${val_batch_size}/imsize_${resized_height}x${resized_width}_wd_${wd}_optim_${optimizer}_lr${lr}_steps_${lr_steps}_gamma_${step_gamma}/e${epochs}_seed6210"
	seg_save_dir="${save_dir}/seg_results"

	mkdir -p $save_dir

	python ../src/trainSegNet2T.py \
	    --model $model \
	    --workers $workers \
	    --trainBatchSize $train_batch_size \
	    --valBatchSize $val_batch_size \
	    --full_res_validation $full_res_validation \
	    --resizedHeight $resized_height \
	    --resizedWidth $resized_width \
	    --cropSize $cropSize \
	    --lr $lr \
	    --dice_loss_factor $dice_loss_factor \
	    --focal_loss_factor $focal_loss_factor \
	    --epochs $epochs \
	    --lr_steps $lr_steps \
	    --step_gamma $step_gamma \
	    --optimizer $optimizer \
	    --wd $wd \
	    --dataset $dataset \
	    --display_samples $display_samples \
	    --save_samples $save_samples \
	    --data_dir $data_dir \
	    --json_path $json_path \
	    --save_dir $save_dir \
	    --seg_save_dir $seg_save_dir \
        --use_high_level $use_high_level \
		--use_low_level $use_low_level \
	    |& tee -a "${save_dir}/debug.log"

use_high_level="True"
use_low_level="False"

	save_dir="../results/${model}_hl_${use_high_level}_ll_${use_low_level}/${dataset}_fold2_23_1/${RandomCropSize}_random_crop/dice_factor_${dice_loss_factor}_focal_factor_${focal_loss_factor}/bs_train${train_batch_size}_val${val_batch_size}/imsize_${resized_height}x${resized_width}_wd_${wd}_optim_${optimizer}_lr${lr}_steps_${lr_steps}_gamma_${step_gamma}/e${epochs}_seed6210"
	seg_save_dir="${save_dir}/seg_results"

	mkdir -p $save_dir

	python ../src/trainSegNet2T.py \
	    --model $model \
	    --workers $workers \
	    --trainBatchSize $train_batch_size \
	    --valBatchSize $val_batch_size \
	    --full_res_validation $full_res_validation \
	    --resizedHeight $resized_height \
	    --resizedWidth $resized_width \
	    --cropSize $cropSize \
	    --lr $lr \
	    --dice_loss_factor $dice_loss_factor \
	    --focal_loss_factor $focal_loss_factor \
	    --epochs $epochs \
	    --lr_steps $lr_steps \
	    --step_gamma $step_gamma \
	    --optimizer $optimizer \
	    --wd $wd \
	    --dataset $dataset \
	    --display_samples $display_samples \
	    --save_samples $save_samples \
	    --data_dir $data_dir \
	    --json_path $json_path \
	    --save_dir $save_dir \
	    --seg_save_dir $seg_save_dir \
        --use_high_level $use_high_level \
		--use_low_level $use_low_level \
	    |& tee -a "${save_dir}/debug.log"


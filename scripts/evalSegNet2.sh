#!/bin/bash

save_test=True
checkpoint_dir="/home/arcseg/Desktop/Shunkai-working/results/checkpts"
checkpoint_title="smp_DeepLabV3+_cholec_13_2_bs12lr0.001e50_checkpoint"
checkpoint="${checkpoint_dir}/${checkpoint_title}"
save_dir="../results/${checkpoint_title}"
data_dir = "../src/data/datasets/cholec_13_2"


python ../src/evalSegNet.py \
--save-dir $save_dir \
--saveTest $save_test \
--model $checkpoint \
--data_path "../src/data/datasets/cholec_13_2" \
|& tee -a "${save_dir}/eval.log"

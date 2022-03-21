#!/bin/bash

save_test=True
checkpoint_dir="/home/arcseg/Desktop/Shunkai-working/scripts"
checkpoint_title="smp_DeepLabV3+_cholec_bs12lr0.001e50_checkpoint"
checkpoint="${checkpoint_dir}/${checkpoint_title}"
save_dir="../results/${checkpoint_dir}/${checkpoint_title}"


python ../src/evalSegNet.py \
--save-dir $save_dir \
--saveTest $save_test \
--model $checkpoint \
|& tee -a "${save_dir}/eval.log"

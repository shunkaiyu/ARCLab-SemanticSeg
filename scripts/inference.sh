#!/bin/bash

model_dir="smp_DeepLabV3+_cholec_bs12lr0.001e50_checkpoint"
imdir_path="../src/data/datasets/cholec_12_3/test/images/37_frame_528_endo.png"
classes_path="../src/data/classes/cholecSegClasses.json"
save_dir="../results/"

python ../src/inference.py \
--model_dir $model_dir \
--imdir_path $imdir_path \
--classes_path $classes_path \
--save_dir $save_dir \
|& tee -a "${save_dir}/eval.log"

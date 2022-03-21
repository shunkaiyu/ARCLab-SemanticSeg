#!/bin/bash

dataset="cholec"
fold1="cholec_12_3"
fold2="cholec_13_2"
fold3="cholec_23_1"
save_dir="./vis_plots/"
#mkdir -p $save_dir

python ../src/data/datasets/data_analysis.py \
	--folder1 $fold1 \
	--folder2 $fold2 \
	--folder3 $fold3 \
	--save_dir $save_dir \


#!/bin/bash

dataset="cholec"
fold1="fold_12_3"
fold2="fold_13_2"
fold3="fold_23_1"
save_dir="/home/arcseg/Desktop/Shunkai-working/src/data/datasets/vis_plots_newSplit/"
mkdir -p $save_dir

python ../src/data/datasets/data_analysis.py \
	--folder1 $fold1 \
	--folder2 $fold2 \
	--folder3 $fold3 \
	--save_dir $save_dir \


#!/bin/bash

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'ade20k', 'coco']
# method: ['unimatch_v2', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='yh_chest'
method='unimatch_v2'
exp='dinov2_base'
split='all'
train_label_json=/yinghepool/yinghe/downstream_data/data_path/chest_organ_11class/data_3d_chest_train.json
train_unlabel_json=/yinghepool/hujunhao/data/v0.9.1-CT-chest-ProjectX-15w/train_7w.json
val_json=/yinghepool/yinghe/downstream_data/data_path/chest_organ_11class/data_3d_chest_test.json

config=configs/${dataset}.yaml
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=29051 \
    $method.py \
    --config=$config  \
    --save-path $save_path \
    --train_label_json $train_label_json \
    --train_unlabel_json $train_unlabel_json \
    --val_json $val_json \
    --port 29051 2>&1 | tee $save_path/out.log

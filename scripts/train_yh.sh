dataset='yh_qixiongjiye'
method='unimatch_v2'
exp='dinov2_base'
split='100_pos'
train_unlabel_json=/yinghepool/yinghe/datalist/v0.9/v0.9.1-CT-chest-ProjectX-2d.json
train_label_json=/yinghepool/downstream_data/data_path/气胸积液/data_2d_气胸积液_train.json 
val_json=/yinghepool/downstream_data/data_path/气胸积液/data_2d_气胸积液_test_all.json

config=configs/${dataset}.yaml
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --infer=$3 \
    --config=$config  \
    --save-path $save_path \
    --train_label_json $train_label_json \
    --train_unlabel_json $train_unlabel_json \
    --val_json $val_json --port $2 2>&1 | tee $save_path/out.log

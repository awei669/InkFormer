#!/bin/bash
# export NCCL_DEBUG=INFO    # for debugging

export CUDA_VISIBLE_DEVICES=0,1,2,3
# TORCH_DISTRIBUTED_DEBUG=DETAIL    # for debugging

epoch_begin=0
epoch_end=5000

train_hw=(
    "data_ckpt/datas/train/Pot1.0Train_pp1.npy"
    "data_ckpt/datas/train/Pot1.1Train_pp1.npy"
    "data_ckpt/datas/train/Pot1.2Train_pp1.npy"
    "data_ckpt/datas/train/Pot1.0Test_pp1.npy"
    "data_ckpt/datas/train/Pot1.1Test_pp1.npy"
    "data_ckpt/datas/train/Pot1.2Test_pp1.npy"
)

log_name=train

### train
args=(
    --log-name ${log_name}
    --batch-user 5
    --sample-per-user 4
    --n-reference 15
    --format 3
    --hw-path ${train_hw[@]}
    --std-path data_ckpt/datas/standard/stand_zi.npy
    --lr 1e-4
    --epoch-begin ${epoch_begin}
    --epoch-end ${epoch_end}
    --save-epoch 500
    --vis-epoch 500
)
accelerate launch main.py "${args[@]}"

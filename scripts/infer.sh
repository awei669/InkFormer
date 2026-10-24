#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

Epochs=5000

log_name=infer

### inferenece
hw_path=data_ckpt/datas/test/competition_set_pp1.npy
args=(
    --log-name ${log_name}
    --epoch "${Epochs}"
    --n-reference 15
    --format 3
    --hw-path "${hw_path}"
    --std-path data_ckpt/datas/standard/stand_zi.npy
)
python inference.py "${args[@]}"

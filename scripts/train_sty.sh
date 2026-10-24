#!/bin/bash
# 训练风格分类器（评测器）（若已有推理产物 syn_saver_ref_15.npy 会额外监控 Acc_syn，无则自动跳过）
export CUDA_VISIBLE_DEVICES=0
python -m classifiers.StyCls_train

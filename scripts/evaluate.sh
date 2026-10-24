#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
log_name=infer
save_name='syn_saver_ref_15.npy'
### evaluation
python -m evaluation.evaluate_ndtw_1 --log-name ${log_name} --save_name ${save_name}
python -m evaluation.evaluate_content_score --log-name ${log_name} --format 3 --save_name ${save_name}
python -m evaluation.evaluate_style_score --log-name ${log_name} --save_name ${save_name}

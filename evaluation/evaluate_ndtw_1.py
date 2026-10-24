import numpy as np
import os
from tqdm import tqdm
import logging
import argparse
from fastdtw import fastdtw
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from utils import *

'''format of syn_saver:
syn_saver = {
    writer_id: {
        key: {
            'gt': hw_char,
            'std': std_char,
            'syn': syn_char,
        }, 
        ...
    }, 
    ...
}
all the chars are in numpy format.
'''


    

def eval_dtw_single(writer_dict, w):
    gt_vs_syn = np.array([]) ; info = [] ; gt_vs_std = np.array([])
    for key in writer_dict.keys():
        gt_std_syn = writer_dict[key]
        gt_char = gt_std_syn['gt']
        syn_char = gt_std_syn['syn']
        # std_char = gt_std_syn['std']
        try:
            # gt vs syn
            dis, _ = fastdtw(gt_char[:,:2], syn_char[:,:2], dist=2)
            norm_coff = len(gt_char) * np.sqrt((gt_char[:,0].max() - gt_char[:,0].min())**2 + (gt_char[:,1].max() - gt_char[:,1].min())**2)
            # norm_coff = len(gt_char)
            dis_nDTW = dis / norm_coff
            gt_vs_syn = np.append(gt_vs_syn, dis_nDTW)
            # # gt vs std
            # dis, _ = fastdtw(gt_char[:,:2], std_char[:,:2], dist=2)
            # dis_nDTW = dis / norm_coff
            # gt_vs_std = np.append(gt_vs_std, dis_nDTW)
            # save the info
            info.append(f'{w}_{key}')
        except:
            print(f'Error in {w} {key}')
            # raise Exception(f'Error in {w} {key}')

    return gt_vs_syn, info

def count(syn_saver):
    count = 0
    for w in syn_saver.keys():
        count += len(syn_saver[w].keys())
    return count


def eval_dtw(args):
    logging.basicConfig(
        filename=f"{args.log_path}/inference.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    os.write(1, f'Loading the saved numpy file: {args.saver_npy}\n'.encode('utf-8'))
    syn_saver = np.load(args.saver_npy, allow_pickle=True).item()
    dis_all = [] ; info = []

    pbar = tqdm(total=count(syn_saver))
    ### 串行计算
    # for w in syn_saver.keys():
    #     dis_chars, info_chars = eval_dtw_single(syn_saver[w], w)
    #     dis_all.extend(dis_chars) ; info.extend(info_chars)
    #     pbar.update(len(info_chars))
    #     pbar.set_postfix_str(f'Current mean DTW distance: {np.array(dis_all).mean()}')
    ### 并行计算
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(eval_dtw_single, syn_saver[w], w) for w in syn_saver.keys()]
        for future in as_completed(futures):
            dis_chars, info_chars = future.result()
            dis_all.extend(dis_chars) ; info.extend(info_chars)
            pbar.update(len(info_chars))
            pbar.set_postfix_str(f'Current mean DTW distance: {np.array(dis_all).mean()}')
    dis_all = np.array(dis_all)
    dis_mean = dis_all.mean()
    print(f'Mean DTW distance: {dis_mean}')
    logging.info(f'Mean DTW distance: {dis_mean}')



    
    pass



def get_args_evaluate():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--log-name', type=str, default=None, help='Path to the log file.')
    parser.add_argument('--save_name', type=str, default='syn_saver.npy', help='result to eval')
    args = parser.parse_args()

    args.log_path = os.path.join('logs', args.log_name)
    args.saver_npy = os.path.join(args.log_path, args.save_name)

    return args


if __name__ == '__main__':
    args = get_args_evaluate()
    eval_dtw(args)
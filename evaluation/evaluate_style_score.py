import numpy as np
import os
from tqdm import tqdm
import logging
from accelerate import Accelerator
import argparse
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import *
from classifiers.StyCls_train import dataset_char_modified, collate_fn_modified, channel_in
from classifiers.StyCls_train import StyCls_WLY, StyCls_Ren, StyCls_res18
from dataset_char import Test_Batch_Sampler


'''format of syn_saver:
syn_saver = {
    writer_char: {
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


def eval_style_score(args):
    logging.basicConfig(
        filename=f"{args.log_path}/inference.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    os.write(1, f'Loading the saved numpy file: {args.saver_npy}\n'.encode('utf-8'))
    syn_saver = np.load(args.saver_npy, allow_pickle=True).item()
    only_syn = dict()
    for w in syn_saver.keys():
        only_syn[w] = dict()
        for key in syn_saver[w].keys():
            only_syn[w][key] = syn_saver[w][key]['syn']
    del syn_saver

    ### Load Test dataset
    dset = dataset_char_modified(args) ; dset.training = False
    dset.load_hw("../casia_ol_make/data/competition_set_pp1.npy")
    dset.load_std("../standard_zi_make/data/stand_zi.npy")
    ## overwrite the hw with the syn
    # we must make sure that the label of classifier (int) and the label of syn (char) are matched 
    lookup_idx2ori = np.load(args.lookup_idx2ori, allow_pickle=True).item()
    lookup_ori2idx = {v: k for k, v in lookup_idx2ori.items()}
    for w_ori in only_syn.keys():
        try:
            w_idx = lookup_ori2idx[w_ori]
            dset.hw[w_idx] = only_syn[w_ori]
        except:
            raise ValueError(f"Error in {w_ori}")
    ### pack the dataloader
    sampler = Test_Batch_Sampler(dset, 100)
    collate_fn = collate_fn_modified()
    test_loader = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)

    ### Pepare model
    # Sty_cls = StyCls_Ren(
    #     in_dim=channel_in, 
    #     cls_num=len(dset.hw)
    #     ).eval()
    Sty_cls = StyCls_res18(
        in_dim=channel_in, 
        cls_num=len(dset.hw)
        ).eval()
    check_point = torch.load(args.model_path)
    Sty_cls.load_state_dict({k.replace('module.', ''): v for k, v in check_point.items()})

    ### Accelearte
    accelerate = Accelerator()
    Sty_cls, test_loader = accelerate.prepare(Sty_cls, test_loader)

    ### Evaluate
    Acc = np.zeros(len(dset.hw.keys()))
    alike = dict(zip(dset.hw.keys(), [np.array([])]*len(dset.hw)))

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating Style Score')
        for batched_data in pbar:
            writer_id = batched_data['writer_id']
            chars = batched_data['chars'] ; mask = batched_data['mask']
            tar_sty = writer_id.to(torch.long)
            
            ###
            logits = Sty_cls(chars, mask)
            Acc_batch = (logits.argmax(1) == tar_sty).sum().item() / len(tar_sty)
            alike[writer_id[0].item()] = np.append(alike[writer_id[0].item()], logits.argmax(1).cpu().numpy())

            Acc[writer_id[0].item()] += Acc_batch * len(tar_sty) / len(dset.hw[writer_id[0].item()])

            pbar.set_postfix({'Acc_batch': f'{Acc_batch*100:.2f}%'})

    Acc = np.mean(Acc)
    print(f'Style Score: {Acc*100:.2f}%')
    logging.info(f'Style score: {Acc*100:.2f}%')

    pass



def get_args_evaluate():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--log-name', type=str, default=None, help='Path to the log file.')
    parser.add_argument('--n_reference', type=int, default=15, help='Number of reference styles.')
    parser.add_argument('--save_name', type=str, default='syn_saver.npy', help='result to eval')
    args = parser.parse_args()

    args.log_path = os.path.join('logs', args.log_name)
    args.saver_npy = os.path.join(args.log_path, args.save_name)

    args.model_path = "data_ckpt/ckpt/sty_evaluator_10000.pth"
    args.lookup_idx2ori = "data_ckpt/ckpt/lookup_idx2ori_on.npy"

    args.format = 3

    return args


def init():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


if __name__ == '__main__':
    init()
    args = get_args_evaluate()
    eval_style_score(args)
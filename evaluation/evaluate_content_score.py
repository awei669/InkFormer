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
from classifiers.ContCls_train import content_classifier, dataset_char_Cont, collate_fn_Cont
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


def draw_syn_n_gt(syn, gt, path=None):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(draw_n_save(syn, None, 256, 2, 1, color=True))
    ax[0].set_title('Synthesized Char')
    ax[2].imshow(draw_n_save(gt, None, 256, 2, 1, color=True))
    ax[2].set_title('Ground Truth')
    ax[1].plot(syn[:, 0], syn[:, 1], 'b', marker='o', label='Synthesized')
    ax[1].plot(gt[:, 0], gt[:, 1], 'r', marker='o', label='Ground Truth')
    ax[1].legend()
    if path is not None:
        plt.savefig(path)
    else:
        plt.savefig('syn_gt.png')
    plt.close()
    return
    


def eval_content_score(args):
    logging.basicConfig(
        filename=f"{args.log_path}/inference.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    
    os.write(1, f'Loading the saved numpy file: {args.saver_npy}\n'.encode('utf-8'))
    syn_saver = np.load(args.saver_npy, allow_pickle=True).item()
    only_syn = dict()
    for w in syn_saver.keys():
        only_syn[w] = dict()
        for key in syn_saver[w].keys():
            only_syn[w][key] = syn_saver[w][key]['syn']
    del syn_saver

    ### Load Test dataset
    dset = dataset_char_Cont() ; dset.format = args.format
    dset.load_hw("../casia_ol_make/data/competition_set_pp1.npy")
    ### overwrite the hw with the syn
    ## in content testing, we dont care if the writer-ids are matched
    for i in range(len(dset.hw)):
        key = list(dset.hw.keys())[i]
        value = list(only_syn.values())[i]
        dset.hw[key] = value
    ### pack the dataloader
    sampler = Test_Batch_Sampler(dset)
    collate_fn = collate_fn_Cont(256, 3)
    test_loader = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)
    
    # lookup from keys str to keys int
    lookup_char2int = np.load("data_ckpt/ckpt/lookup_char2int.npy", allow_pickle=True).item()

    ### Pepare model
    Cont_cls = content_classifier(
        in_dim=3, 
        cls_num=len(lookup_char2int)
        )
    check_point = torch.load(args.model_path)
    Cont_cls.load_state_dict({k.replace('module.', ''): v for k, v in check_point.items()})
    Cont_cls.eval()

    ### Accelearte
    accelerate = Accelerator()
    Cont_cls, test_loader = accelerate.prepare(Cont_cls, test_loader)

    ### Evaluate
    Acc_content = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Content Evaluation')
        for batched_data in pbar:
            writer_id = batched_data['writer_id']
            keys_char = batched_data['keys']
            hw_chars = batched_data['hw_chars']
            hw_mask = batched_data['hw_mask']
            device = hw_chars.device

            keys_int = torch.tensor([lookup_char2int[key] for key in keys_char], device=device)

            ###
            logits = Cont_cls(hw_chars, hw_mask)
            Acc_batch = (logits.argmax(1) == keys_int).sum().item() / len(keys_int)

            Acc_content += Acc_batch / len(test_loader)
            
            pbar.set_postfix({'Acc_batch': f'{Acc_batch*100:.2f}%'})

    print(f'Content score: {Acc_content*100:.2f}%')
    logging.info(f'Content score: {Acc_content*100:.2f}%')

    pass



def get_args_evaluate():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--log-name', type=str, default=None, help='Path to the log file.')
    parser.add_argument('--hw-path', type=str, default=None, help='Path to the hw file.')
    parser.add_argument('--format', type=int, default=3, help='Format of the hw file.')
    parser.add_argument('--save_name', type=str, default='syn_saver.npy', help='result to eval')
    args = parser.parse_args()

    args.log_path = os.path.join('logs', args.log_name)
    args.saver_npy = os.path.join(args.log_path, args.save_name)

    args.model_path = "data_ckpt/ckpt/cont_evaluator_2000.pth"

    return args


def init():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


if __name__ == '__main__':
    init()
    args = get_args_evaluate()
    eval_content_score(args)
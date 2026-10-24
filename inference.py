import numpy as np
from tqdm import tqdm
import logging
import time
from accelerate import Accelerator
import os
import argparse
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard.writer import SummaryWriter

import dataset_char
from modules import Transformer_AR

from utils import *


def inference(args):
    '''Inference the model and save the results.'''
    log_folder = f'logs/{args.log_name}'
    os.makedirs(log_folder, exist_ok=True)
    log_img = f'{log_folder}/img_infer'
    os.makedirs(log_img, exist_ok=True)

    logging.basicConfig(
    filename=f"{log_folder}/inference.log",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

    logging.info(f'Args: {args}')

    ### prepare accelerator
    accelerator = Accelerator(
        even_batches=False, 
    )

    dset = dataset_char.Dataset_Char(args)
    dset.load_hw(args.hw_path)
    dset.load_std(args.std_path)
    sampler = dataset_char.Test_Batch_Sampler(dset)
    collate_fn = dataset_char.collate_fn(dset.Nmax, dset.format)
    loader = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)

    ### model preparation
    model = Transformer_AR(
        in_dim=args.format, 
        embed_dim=512, 
        n_heads=16, 
        n_layers=4, 
        dropout=0.1, 
        M=20, 
    )
    model_checkpoint = torch.load(f'data_ckpt/ckpt/ar_generator_{args.epoch}.pth')
    model.load_state_dict({k.replace('module.', ''): v for k, v in model_checkpoint.items()})
    model.eval()

    ### accelerator preparation
    model, loader = accelerator.prepare(
        model, loader
    )

    ### prepare saver dict for synthesis
    '''format of syn_saver:
    syn_saver = {
        writer_char: {
            key: {
                'gt': hw_char,
                'std': std_char,
                'syn': syn_char,
            }
        }
    }
    all the chars are in numpy format.
    '''
    syn_saver = dict()

    pbar = tqdm(range(len(loader)), disable=not accelerator.is_main_process)
    for b_i, batched_data in enumerate(loader):
        writer_id = batched_data['writer_id']
        keys = batched_data['keys']
        hw_chars = batched_data['hw_chars'] ; hw_mask = batched_data['hw_mask']
        std_chars = batched_data['std_chars'] ; std_mask = batched_data['std_mask']
        sty_refs = batched_data['sty_refs'] ; sty_mask = batched_data['sty_mask']
        sty_stds = batched_data['sty_stds'] ; sty_stds_mask = batched_data['sty_stds_mask']

        device = hw_chars.device

        with torch.no_grad():
            syn_seq = model.inference(
                hw_chars.size(1),
                sty_refs, sty_mask, 
                sty_stds, sty_stds_mask,
                std_chars, std_mask, 
                )

        pbar.update(1)

        ### save the results
        if accelerator.is_main_process:
            # pull the data to cpu and numpy
            writer_id = writer_id.cpu().numpy()
            hw_chars = hw_chars.cpu().numpy() ; hw_lens = hw_mask.cpu().numpy().sum(axis=1)
            std_chars = std_chars.cpu().numpy() ; std_lens = std_mask.cpu().numpy().sum(axis=1)
            syn = syn_seq.cpu().numpy()
            # post-process for syn
            syn[..., :2] = np.clip(syn[..., :2], -1, 1)
            if len(syn.shape) == 3:
                syn[..., 2] = np.sign(syn[..., 2])
            if len(syn.shape) == 5:
                logits = syn[..., 2:]
                one_hot = np.eye(3)[logits.argmax(axis=-1)]
                syn = np.concatenate((syn[..., :2], one_hot), axis=-1)
            #
            for i, writer in enumerate(writer_id):
                writer_char = dset.lookup_idx2ori[writer]
                if writer_char not in syn_saver:
                    syn_saver[writer_char] = dict()
                # we only save the valid part of the chars, avoiding huge storage cost by massive padding
                gt_this = hw_chars[i][:int(hw_lens[i])]
                gt_this = v2x(gt_this)
                std_this = std_chars[i][:int(std_lens[i])]
                std_this = v2x(std_this)
                syn_this = rmv_pads(syn[i])
                syn_this = v2x(syn_this)
                syn_saver[writer_char][keys[i]] = {
                    'gt': gt_this,
                    'std': std_this,
                    'syn': syn_this,
                }
            
            ### visualize the results
            visualize_1w(syn_saver[writer_char], os.path.join(log_img, f'writer_{writer_char}.png'), 5)

    ### save the results
    if accelerator.is_main_process:
        np.save(f'{log_folder}/syn_saver_ref_15.npy', syn_saver)
        logging.info(f'Synthesized results saved to {log_folder}/syn_saver_ref_15.npy.')
        print(f'Synthesized results saved to {log_folder}/syn_saver_ref_15.npy.')
    
    return


def visualize_1w(writer_dict, path, n):
    canvas_size = 256 ; line_width = 2 ; point_radius = 1
    # only n samples for each writer
    picked_keys = np.random.choice(list(writer_dict.keys()), n, replace=False)
    fig, axs = plt.subplots(n, 3, figsize=(3*3, n*3))
    for i, key in enumerate(picked_keys):
        gt_std_syn = writer_dict[key]
        gt_char = gt_std_syn['gt'] ; std_char = gt_std_syn['std'] ; syn_char = gt_std_syn['syn']
        image_gt = draw_n_save(gt_char, None, canvas_size, line_width, point_radius, color=True)
        image_std = draw_n_save(std_char, None, canvas_size, line_width, point_radius, color=True)
        image_syn = draw_n_save(syn_char, None, canvas_size, line_width, point_radius, color=True)
        axs[i, 0].imshow(image_std)
        axs[i, 0].set_title('Standard Char')
        axs[i, 2].imshow(image_gt)
        axs[i, 2].set_title('Ground Truth')
        axs[i, 1].imshow(image_syn)
        axs[i, 1].set_title('Synthesized Char')
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return


def get_args_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-name', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--n-reference', type=int, default=8)
    parser.add_argument('--format', type=int, default=3)
    parser.add_argument('--hw-path', type=str, default=None)
    parser.add_argument('--std-path', type=str, default=None)
    args = parser.parse_args()

    if args.hw_path is None:
        args.hw_path = 'data_ckpt/datas/test/competition_set_pp1.npy'
    if args.std_path is None:
        args.std_path = 'data_ckpt/datas/standard/stand_zi.npy'

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
    
    args = get_args_inference()
    inference(args)


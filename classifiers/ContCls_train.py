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

import dataset_char
from modules.ResNet1d import resnet18_1d

from utils import *

channel_in = 3

class content_classifier(nn.Module):
    def __init__(self, 
                 in_dim, 
                 cls_num,
                 ) -> None:
        super().__init__()
        self.enc = nn.LSTM(in_dim, 512, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512*2, cls_num)

    def forward(self, x, x_mask):
        B = x.size(0)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_mask.sum(1).to(torch.int64).cpu(), batch_first=True, enforce_sorted=False)
        _, x = self.enc(x_packed)
        x = x[0].permute(1, 0, 2).reshape(B, -1)
        x = self.fc(x)
        return x


class dataset_char_Cont(dataset_char.Dataset_Char):
    
    def normalize(self, char):
        '''we concat coords and velocity together'''
        char = normalize_pts(char)      # 只需要norm到[0, 1]即可，不在乎之前做了什么预处理
        coords = char[:, :2] ; ps = char[:, 2][..., None]   # 2, 1
        velo = np.diff(coords, axis=0, prepend=0)   # 2
        # velo_norm = np.linalg.norm(velo, axis=-1, keepdims=True)  # 1
        # accel = np.diff(velo, axis=0, prepend=0)   # 2
        # accel_norm = np.linalg.norm(accel, axis=-1, keepdims=True)  # 1
        char = np.concatenate(
            [
                velo, 
                # velo_norm,
                # accel,
                # accel_norm,
                ps
                ], 
            axis=-1
            )
        return char

    def __getitem__(self, uk_tuple):
        user, key = uk_tuple
        user_chars = self.hw[user]
        picked_gt_chars = user_chars[key]

        # corrdinate to velocity
        picked_gt_chars = self.normalize(picked_gt_chars)

        return (
            user, 
            key, 
            picked_gt_chars,
            )


class collate_fn_Cont(dataset_char.collate_fn):
    def padding(self, chars):
        '''only suits format-3, so the channel for each char is  2(velo) + 1(ps) = 3'''
        mask = np.zeros((len(chars), self.Nmax))
        for i in range(len(chars)):
            char = chars[i]
            if len(char) < self.Nmax:
                pad = np.zeros((self.Nmax-len(char), channel_in))
                pad[:,-1] = -1
                chars[i] = np.concatenate([char, pad], axis=0)
            else: pass
            mask[i, :len(char)] = 1
            assert len(chars[i]) == self.Nmax
        return np.stack(chars, axis=0), mask

    def __call__(self, batch):
        writer_id, keys, hw_chars = zip(*batch)

        writer_id = np.stack(writer_id)
        writer_id = torch.from_numpy(writer_id).long()
        keys = np.stack(keys, dtype=object)

        hw_chars = list(hw_chars)
        self.Nmax = self.batch_Nmax([hw_chars])
        hw_chars, hw_mask = self.padding(hw_chars)
        hw_chars = torch.from_numpy(hw_chars).float() 
        hw_mask = torch.from_numpy(hw_mask).float()

        return {
            'writer_id': writer_id,
            'keys': keys,
            'hw_chars': hw_chars,
            'hw_mask': hw_mask
        }



def train(args):
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f"logs/cont_train.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    datasets = [
        "data_ckpt/datas/train/Pot1.0Train_pp1.npy",
        "data_ckpt/datas/train/Pot1.1Train_pp1.npy",
        "data_ckpt/datas/train/Pot1.2Train_pp1.npy",
        "data_ckpt/datas/train/Pot1.0Test_pp1.npy",
        "data_ckpt/datas/train/Pot1.1Test_pp1.npy",
        "data_ckpt/datas/train/Pot1.2Test_pp1.npy"
    ]
    ### Load train dataset
    train_dset = dataset_char_Cont()
    for dataset in datasets:
        train_dset.load_hw(dataset)
    train_dset.load_std("data_ckpt/datas/standard/stand_zi.npy")
    train_dset.update_contained_chars()
    sampler = dataset_char.Train_Batch_Sampler(args, train_dset)
    collate_fn = collate_fn_Cont(256, 3)
    train_loader = DataLoader(train_dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)
    
    # lookup from keys str to keys int
    lookup_char2int = dict(zip(train_dset.contained_chars, range(len(train_dset.contained_chars))))

    ### Load Test dataset
    test_dset_gt = dataset_char_Cont()
    test_dset_gt.load_hw("data_ckpt/datas/test/competition_set_pp1.npy")
    test_dset_gt.load_std("data_ckpt/datas/standard/stand_zi.npy")
    test_dset_gt.update_contained_chars()
    sampler = dataset_char.Test_Batch_Sampler(test_dset_gt, batch_size=100)
    collate_fn = collate_fn_Cont(256, 3)
    test_loader_gt = DataLoader(test_dset_gt, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)

    assert test_dset_gt.contained_chars - train_dset.contained_chars == set(), 'Test set contains chars not in train set'

    ### Load Syn dataset
    ## load syn.npy
    syn_path = "logs/infer/syn_saver_ref_15.npy"
    if os.path.exists(syn_path):
        os.write(1, f'Loading the saved numpy file: {syn_path}\n'.encode('utf-8'))
        syn_saver = np.load(syn_path, allow_pickle=True).item()
        only_syn = dict()
        for w in syn_saver.keys():
            only_syn[w] = dict()
            for key in syn_saver[w].keys():
                only_syn[w][key] = syn_saver[w][key]['syn']
        del syn_saver
        ## dateset
        test_dset_syn = dataset_char_Cont()
        test_dset_syn.load_hw("data_ckpt/datas/test/competition_set_pp1.npy")
        ### overwrite the hw with the syn
        ## in content testing, we dont care if the writer-ids are matched
        for i in range(len(test_dset_syn.hw)):
            key = list(test_dset_syn.hw.keys())[i]
            value = list(only_syn.values())[i]
            test_dset_syn.hw[key] = value
        ### pack the dataloader
        sampler = dataset_char.Test_Batch_Sampler(test_dset_syn, batch_size=100)
        collate_fn = collate_fn_Cont(256, 3)
        test_loader_syn = DataLoader(test_dset_syn, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)
    else:
        test_loader_syn = None
        print(f'Syn result not found, skip Acc_syn evaluation.')
    
    ### Pepare model
    Cont_cls = content_classifier(
        in_dim=channel_in, 
        cls_num=len(train_dset.contained_chars)
        )
    Cont_cls.train()

    ### Pepare optimizer
    optimizer = optim.Adam(Cont_cls.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ### Accelerator
    accelerator = Accelerator(
        even_batches=False,
    )
    train_loader, test_loader_gt, test_loader_syn, Cont_cls, optimizer, lr_scheduler = accelerator.prepare(
        train_loader, test_loader_gt, test_loader_syn, Cont_cls, optimizer, lr_scheduler
    )

    ### record everything
    if accelerator.is_main_process:
        logging.info(f"Args: {args}")
        logging.info(f"Model: {Cont_cls}")
        logging.info(f"Optimizer: {optimizer}")
        logging.info(f"LR Scheduler: {vars(lr_scheduler)}")

    log_losses = {
        'loss_ce': 0., 
        'Acc_batch': 0.,
    } ; StepsPerEpoch = len(train_loader)

    ### Train
    pbar = tqdm(range(args.epochs * len(train_loader)), disable=not accelerator.is_main_process)
    for epoch in range(args.epochs):
        pbar.set_description(f'Epoch: {epoch+1}/{args.epochs}')
        for batched_data in train_loader:
            writer_id = batched_data['writer_id']
            keys = batched_data['keys']
            hw_chars = batched_data['hw_chars']
            hw_mask = batched_data['hw_mask']
            device = hw_chars.device

            keys = torch.tensor([lookup_char2int[key] for key in keys], device=device)

            ###
            logits = Cont_cls(hw_chars, hw_mask)
            loss_ce = nn.CrossEntropyLoss()(logits, keys) 

            accelerator.backward(loss_ce)
            
            # Check for NaN values in gradients
            for name, param in Cont_cls.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad.zero_()
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(Cont_cls.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                Acc_batch = (logits.argmax(1) == keys).sum().item() / len(keys)
                pbar.update(1)
                pbar.set_postfix(
                    {
                        'loss_ce': loss_ce.item(), 
                        'Acc_batch': f'{Acc_batch*100:.2f}%'
                        }
                    )
                log_losses['loss_ce'] += loss_ce.item() / StepsPerEpoch
                log_losses['Acc_batch'] += Acc_batch / StepsPerEpoch

        lr_scheduler.step()

        if accelerator.is_main_process:
            ### log the information
            log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Epoch {epoch} => "
            for key, value in log_losses.items():
                log_message += f"{key}: {value:.6f} "
            log_message += f"lr: {optimizer.param_groups[0]['lr']:.2e} "
                # writer.add_scalar(key, value, epoch*len(loader)+pbar.n)
            logging.info(log_message)
            log_losses = {key: 0. for key in log_losses.keys()}   # reset the log_losses

            if (epoch+1) % args.test_epoch == 0:
                Cont_cls.eval()
                # test on real data
                Acc_real = 0
                with torch.no_grad():
                    for batched_data in test_loader_gt:
                        writer_id = batched_data['writer_id']
                        keys = batched_data['keys']
                        hw_chars = batched_data['hw_chars']
                        hw_mask = batched_data['hw_mask']
                        device = hw_chars.device

                        keys = torch.tensor([lookup_char2int[key] for key in keys], device=device)

                        ###
                        logits = Cont_cls(hw_chars, hw_mask)
                        Acc_batch = (logits.argmax(1) == keys).sum().item() / len(keys)

                        Acc_real += Acc_batch / len(test_loader_gt)
                print(f'Acc_real after epoch {epoch+1}: {Acc_real*100:.2f}%')
                logging.info(f'Acc_real after epoch {epoch+1}: {Acc_real*100:.2f}%')

                # test on syn data
                if test_loader_syn is not None:
                    Acc_syn = 0
                    with torch.no_grad():
                        for batched_data in test_loader_syn:
                            writer_id = batched_data['writer_id']
                            keys = batched_data['keys']
                            hw_chars = batched_data['hw_chars']
                            hw_mask = batched_data['hw_mask']
                            device = hw_chars.device

                            keys = torch.tensor([lookup_char2int[key] for key in keys], device=device)

                            ###
                            logits = Cont_cls(hw_chars, hw_mask)
                            Acc_batch = (logits.argmax(1) == keys).sum().item() / len(keys)

                            Acc_syn += Acc_batch / len(test_loader_syn)
                    print(f'Acc_syn after epoch {epoch+1}: {Acc_syn*100:.2f}%')
                    logging.info(f'Acc_syn after epoch {epoch+1}: {Acc_syn*100:.2f}%')

                # save model
                torch.save(Cont_cls.state_dict(), os.path.join(args.save_path, f'cont_evaluator_{epoch+1}.pth'))
                np.save(os.path.join(args.save_path, f'lookup_char2int.npy'), lookup_char2int)

                Cont_cls.train()

    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-name', type=str, default='train')
    args = parser.parse_args()

    args.batch_user = 16
    args.sample_per_user = 64
    args.lr = 1e-3
    args.epochs = 2000
    args.test_epoch = 500
    args.save_path = 'data_ckpt/ckpt'

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
    args = get_args()
    train(args)
    pass

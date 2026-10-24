import numpy as np
from tqdm import tqdm
import logging
import time
from accelerate import Accelerator
import os
import argparse
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset_char
from modules.encoders import att_pooling
from modules.ResNet1d import BasicBlock
from modules.ResNet1d import resnet18_1d

from utils import *


channel_in = 3

class StyCls_WLY(nn.Module):
    def __init__(self, 
                 in_dim, 
                 cls_num,
                 ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, 256, 4, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256*2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, cls_num)

        self.att_pool_tem = att_pooling(256*2)

    def forward(self, x, x_mask=None):
        '''
        x: (B, n, L, c)
        x_mask: (B, n, L)
        '''
        B, n, L, c = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, L, n*c)
        x, _ = self.lstm(x)

        if x_mask is None:
            x_mask = torch.ones(x.shape[:2], device=x.device)
        x = self.att_pool_tem(x, x_mask)    # (B*n, 256*2)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class StyCls_Ren(nn.Module): 
    def __init__(self, 
                 in_dim, 
                 cls_num, 
                 ):
        super().__init__()
        self.stem = nn.ModuleList([
            nn.Conv1d(in_dim, 128, 3, 1, 1), 
            nn.LayerNorm(128),
        ])
        self.block1 = nn.Sequential(
            self.A_layer(128, 3, 1), 
            self.A_layer(128, 3, 1), 
            self.A_layer(128, 3, 1),
        )
        self.down1 = nn.ModuleList([
            nn.Conv1d(128, 256, 4, 2, 1), 
            nn.LayerNorm(256)
        ])
        self.block2 = nn.Sequential(
            self.A_layer(256, 3, 3), 
            self.A_layer(256, 3, 3), 
            self.A_layer(256, 3, 3),
        )
        self.down2 = nn.ModuleList([
            nn.Conv1d(256, 512, 4, 2, 1), 
            nn.LayerNorm(512)
        ])
        self.block3 = nn.Sequential(
            self.A_layer(512, 3, 5), 
            self.A_layer(512, 3, 5), 
            self.A_layer(512, 3, 5),
        )
        self.down3 = nn.ModuleList([
            nn.Conv1d(512, 1024, 4, 2, 1), 
            nn.LayerNorm(1024)
        ])

        self.cls = nn.Linear(1024, cls_num)
    
    class A_layer(nn.Module): 
        def __init__(self, 
                     in_dim, 
                     kernel = 3, 
                     dilation = 1,
                     ):
            super().__init__()
            stride = 1 
            padding = (kernel-1)//2 * dilation
            self.conv = nn.Conv1d(in_dim, in_dim, kernel, stride, padding, dilation=dilation)
            self.norm = nn.LayerNorm(in_dim)
            self.act = nn.GELU()

        def forward(self, x):
            input = x
            x = self.conv(x)
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.act(x)
            return x + input
        
    def forward(self, x, x_mask):
        B, n, L, c = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B*n, c, L)

        x = self.stem[0](x)
        x = self.stem[1](x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.block1(x)
        x = self.down1[0](x)
        x = self.down1[1](x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.block2(x)
        x = self.down2[0](x)
        x = self.down2[1](x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.block3(x)
        x = self.down3[0](x)
        x = self.down3[1](x.permute(0, 2, 1)).permute(0, 2, 1)

        _, c1, L1 = x.shape
        x = x.reshape(B, n, c1, L1)
        x_mask = nn.Upsample(size=(L1,), mode='nearest')(x_mask.reshape(B*n, 1, L)).reshape(B, n, 1, L1)
        x1 = (x * x_mask).sum(-1) / (x_mask.sum(-1) + 1e-6)  # (B, n, c1)
        x2 = x1.mean(1)  # (B, c1)
        logits = self.cls(x2)
        
        return logits


##### Style Classifier from RenMS original #####
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, dilation=dilation, stride=stride)
        #self.dp = nn.Dropout(p=0.02)
                              
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = F.pad(x, [0, self.pad])
        x = self.conv(x)
        return x

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        # residual in == out stride = 1
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1:
            kernel = stride * 2
        else:
            kernel = 1

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=5),
            nn.ELU(),
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)

class style_encoder_testset(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear = Conv1d(in_channels=3, out_channels=dim, kernel_size=3)
        
        self.en_block1 = nn.Sequential(
              EncoderBlock(dim, dim*2, 2))
              
        self.en_block2 = nn.Sequential(
              EncoderBlock(dim*2, dim*4, 2))
              
        self.en_block3 = nn.Sequential(
              EncoderBlock(dim*4, dim*8, 2))
        self.elu = nn.ELU()

        ### train ###
        self.func_c = nn.Linear(1024, 60)

    def forward(self, x):
        x = self.linear(x) # b 128 L
        en1 = self.en_block1(self.elu(x))  # b 256 L/2 
        en2 = self.en_block2(self.elu(en1))
        en3 = self.en_block3(self.elu(en2))

        f = en3.mean(-1) # b 1024
        logits = self.func_c(f) # b 60
        return logits
##### ------------------------------------ #####


class StyCls_res18(nn.Module): 
    def __init__(self, in_dim=3, cls_num=60):
        super().__init__()
        self.res18 = resnet18_1d(n_in=in_dim)
        self.cls = nn.Linear(512, cls_num)

    def forward(self, x, x_mask):
        b, n, l, c = x.shape
        x = einops.rearrange(x, 'b n l c -> (b n) c l')
        x = self.res18(x)
        
        ## GAP along time axis
        x_mask = einops.rearrange(x_mask, 'b n l -> (b n) 1 l')
        x_mask = nn.Upsample(size=(x.shape[-1],), mode='nearest')(x_mask)
        x = (x * x_mask).sum(-1) / (x_mask.sum(-1) + 1e-6)

        ## GAP along style axis
        x = einops.rearrange(x, '(b n) c -> b n c', b=b)
        x = x.mean(1)  # (b, c)

        logits = self.cls(x)
        return logits

    def get_style_feat(self, x, x_mask):
        b, n, l, c = x.shape
        x = einops.rearrange(x, 'b n l c -> (b n) c l')
        x = self.res18(x)
        
        ## GAP along time axis
        x_mask = einops.rearrange(x_mask, 'b n l -> (b n) 1 l')
        x_mask = nn.Upsample(size=(x.shape[-1],), mode='nearest')(x_mask)
        x = (x * x_mask).sum(-1) / (x_mask.sum(-1) + 1e-6)

        ## GAP along style axis
        x = einops.rearrange(x, '(b n) c -> b n c', b=b)
        x = x.mean(1)  # (b, c)

        return x




class StyleOnlyTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    # ----------- 仿射变换类 -----------
    def random_scaling(self, data, scale_range=(0.8, 1.2)):
        if len(data) < 10:
            return data
        scale_x = np.random.uniform(*scale_range)
        scale_y = np.random.uniform(*scale_range)
        data[:, 0] *= scale_x
        data[:, 1] *= scale_y
        return data

    # ----------- 时序变换类 -----------
    def time_warping(self, data, sigma=0.2, knot=4):
        if len(data) < 10:
            return data
        coords, states = data[:, :2], data[:, 2:]
        orig_steps = len(data)
        warp_steps = (np.linspace(0, orig_steps, num=knot + 2)).astype(int)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        cs = CubicSpline(warp_steps, warp_steps * random_warps)
        warped_steps = cs(np.arange(orig_steps))
        warped_coords = np.zeros_like(coords)
        for i in range(2):
            warped_coords[:, i] = np.interp(warped_steps, np.arange(orig_steps), coords[:, i])
        return np.hstack((warped_coords, states))

    def local_warping(self, data):
        if len(data) < 10:
            return data
        seg_len = max(1, int(len(data) * 0.1))
        start = np.random.randint(0, len(data) - seg_len)
        warped = self.time_warping(data[start:start + seg_len], sigma=0.1, knot=2)
        data[start:start + seg_len] = warped
        return data

    # ----------- 局部扰动类 -----------
    def random_local_corruption(self, data, max_ratio=0.2, noise_std=0.5, mode='either'):
        if len(data) < 10:
            return data
        seg_len = max(1, int(len(data) * np.random.uniform(0.05, max_ratio)))
        start = np.random.randint(0, len(data) - seg_len)
        end = start + seg_len
        op = np.random.choice(['mask', 'noise']) if mode == 'either' else mode
        if op == 'mask':
            data[start:end, :2] = 0
        elif op == 'noise':
            noise = np.random.normal(scale=noise_std, size=(seg_len, 2))
            data[start:end, :2] += noise
        return data

    # ----------- 点操作类 -----------
    def point_dropping(self, data, drop_prob=0.05):
        if len(data) < 10:
            return data
        keep_mask = np.random.rand(len(data)) > drop_prob
        if keep_mask.sum() < 2:
            keep_mask[np.random.choice(len(data), 2, replace=False)] = True
        coords = data[:, :2]
        states = data[:, 2:]
        kept_idx = np.where(keep_mask)[0]
        x_interp = np.interp(np.arange(len(data)), kept_idx, coords[kept_idx][:, 0])
        y_interp = np.interp(np.arange(len(data)), kept_idx, coords[kept_idx][:, 1])
        coords_interp = np.stack([x_interp, y_interp], axis=1)
        return np.hstack((coords_interp, states))

    def point_repeating(self, data, repeat_prob=0.05):
        if len(data) < 10:
            return data
        result = []
        for pt in data:
            result.append(pt)
            if np.random.rand() < repeat_prob:
                result.append(pt.copy())
        return np.array(result)

    # ----------- 总调度 -----------
    def augment(self, data):
        if np.random.rand() < self.prob:
            data = self.random_scaling(data)

        if np.random.rand() < self.prob:
            data = self.time_warping(data)

        if np.random.rand() < self.prob:
            data = self.local_warping(data)

        if np.random.rand() < self.prob:
            data = self.point_dropping(data)

        if np.random.rand() < self.prob:
            data = self.point_repeating(data)

        if np.random.rand() < self.prob:
            data = self.random_local_corruption(data, mode='either')

        return data

    def __call__(self, data):
        return self.augment(data)

class dataset_char_modified(dataset_char.Dataset_Char):
    training = True
    augment = StyleOnlyTransform()

    def __getitem__(self, uk_tuple):
        user, key = uk_tuple
        user_chars = self.hw[user]
        picked_chars = user_chars[key]
        
        # pickup style reference
        picked_keys = np.random.choice(list(user_chars.keys()), self.n_sty-1, replace=False)
        picked_chars = [picked_chars] + [user_chars[k] for k in picked_keys]

        if self.training:
            picked_chars = [self.augment(char) for char in picked_chars]
        picked_chars = [normalize_pts(char) for char in picked_chars]
        picked_chars = [x2v(char) for char in picked_chars]

        return (
            user, 
            key, 
            picked_chars
            )

class collate_fn_modified():
    def __init__(self):
        self.Nmax = 1024

    def padding(self, chars):
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

    def batch_Nmax(self, chars_lists):
        '''calculate the Nmax of the batch.'''
        Nmax = []
        for chars in chars_lists:
            Nmax.append(max([char.shape[0] for char in chars]))
        Nmax = max(Nmax)
        Nmax = 8 * np.ceil(Nmax / 8)        # makes length efficient for 3-times down-sampling
        return int(Nmax)
    
    def __call__(self, batch):
        writer_id, key, chars_Bn = zip(*batch)

        writer_id = np.stack(writer_id)
        writer_id = torch.from_numpy(writer_id).long()

        key = list(key)

        chars_Bn = list(chars_Bn)
        self.Nmax = self.batch_Nmax(chars_Bn)
        mask = []
        for i in range(len(chars_Bn)):
            chars_Bn[i], mask_this = self.padding(chars_Bn[i])
            mask.append(mask_this)
        chars_Bn = np.stack(chars_Bn, axis=0) # (B, n, L, c)
        mask = np.stack(mask, axis=0)

        chars_Bn = torch.from_numpy(chars_Bn).float()
        mask = torch.from_numpy(mask).float()

        return {
            'writer_id': writer_id,
            'keys': key,
            'chars': chars_Bn,
            'mask': mask,
        }


def train(args):
    os.makedirs(f"logs/{args.log_name}", exist_ok=True)
    os.makedirs('data_ckpt/ckpt', exist_ok=True)
    logging.basicConfig(
        filename=f"logs/{args.log_name}/train.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    ### Load train dataset
    train_dset = dataset_char_modified(args) ; train_dset.training = True
    train_dset.load_hw("data_ckpt/datas/test/competition_set_pp1.npy")
    sampler = dataset_char.Train_Batch_Sampler(args, train_dset)
    collate_fn = collate_fn_modified()
    train_loader = DataLoader(train_dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)

    ### Load Test dataset
    # which is the same as train dataset, simply switch the batch sampler and we are good to go
    test_dset = dataset_char_modified(args) ; test_dset.training = False
    test_dset.load_hw("data_ckpt/datas/test/competition_set_pp1.npy")
    test_dset.switch_lookup(train_dset.lookup_idx2ori)
    sampler = dataset_char.Test_Batch_Sampler(test_dset, 100)
    test_loader = DataLoader(test_dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)

    ### Load syn dataset
    # optional: monitor accuracy on synthesized chars
    syn_dset = dataset_char_modified(args) ; syn_dset.training = False
    syn_dset.load_hw("data_ckpt/datas/test/competition_set_pp1.npy")
    syn_dset.switch_lookup(train_dset.lookup_idx2ori)
    ## replace hw by syn
    # syn_data = np.load('logs/SDT/syn_saver.npy', allow_pickle=True).item()
    syn_path = 'logs/infer/syn_saver_ref_15.npy'
    if os.path.exists(syn_path):
        syn_data = np.load(syn_path, allow_pickle=True).item()
        for idx in syn_dset.lookup_idx2ori.keys():
            tgt_syn_writer = syn_data[syn_dset.lookup_idx2ori[idx]]
            for key in syn_dset.hw[idx].keys():
                syn_dset.hw[idx][key] = tgt_syn_writer[key]['syn']
        del syn_data
        sampler = dataset_char.Test_Batch_Sampler(syn_dset, 100)
        syn_loader = DataLoader(syn_dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16)
    else:
        syn_loader = None
        print(f'Syn result not found, skip Acc_syn evaluation.')

    ### Pepare model
    # Sty_cls = StyCls_WLY(
    #     in_dim=channel_in*args.n_reference, 
    #     cls_num=len(train_dset.hw)
    #     )
    # Sty_cls = StyCls_Ren(
    #     in_dim=channel_in, 
    #     cls_num=len(train_dset.hw)
    #     )
    # Sty_cls = style_encoder_testset()
    Sty_cls = StyCls_res18(
        in_dim=channel_in, 
        cls_num=len(train_dset.hw)
        )
    # check_point = torch.load('models_zoo/StyCls_Ep5000_Acc98.94.pth')
    # Sty_cls.load_state_dict({k.replace('module.', ''): v for k, v in check_point.items()})

    ### Pepare optimizer
    optimizer = optim.Adam([{'params': Sty_cls.parameters(), 'initial_lr': args.lr}], lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_end, eta_min=1e-6, last_epoch=args.epoch_begin-1)

    ### Accelerator
    accelerator = Accelerator(
        even_batches=False,
    )
    train_loader, test_loader, syn_loader = accelerator.prepare(train_loader, test_loader, syn_loader)
    Sty_cls, optimizer = accelerator.prepare(Sty_cls, optimizer)

    ### record everything
    if accelerator.is_main_process:
        logging.info(f"Args: {args}")
        logging.info(f"Model: {Sty_cls}")
        logging.info(f"Optimizer: {optimizer}")
        logging.info(f"LR Scheduler: {vars(lr_scheduler)}")

    log_losses = {
        'loss_ce': 0., 
        'Acc_batch': 0.,
    } ; StepsPerEpoch = len(train_loader)

    ### Train Iteration
    pbar = tqdm(range((args.epoch_end - args.epoch_begin)*len(train_loader)), disable=not accelerator.is_local_main_process)
    for epoch in range(args.epoch_begin, args.epoch_end):
        pbar.set_description(f"Epoch {epoch}/{args.epoch_end}")
        for batched_data in train_loader:
            writer_id = batched_data['writer_id']
            chars = batched_data['chars'] ; mask = batched_data['mask']
            tar_sty = writer_id.to(torch.long)

            ###
            logits = Sty_cls(chars, mask)
            loss_ce = F.cross_entropy(logits, tar_sty)

            accelerator.backward(loss_ce)

            # Check for NaN values in gradients
            for name, param in Sty_cls.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad.zero_()
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(Sty_cls.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                Acc_batch = (logits.argmax(1) == tar_sty).sum().item() / len(tar_sty)
                pbar.update(1)
                pbar.set_postfix(
                    {
                        'loss_ce': loss_ce.item(), 
                        'Acc_batch': Acc_batch,
                        'lr': optimizer.param_groups[0]['lr'], 
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

            ### testing
            if (epoch+1) % args.test_epoch == 0:
                Sty_cls.eval()
                # test on real data
                Acc_real = 0
                char_total = test_dset.hw_total()
                with torch.no_grad():
                    for batched_data in test_loader:
                        writer_id = batched_data['writer_id']
                        chars = batched_data['chars'] ; mask = batched_data['mask']
                        tar_sty = writer_id.to(torch.long)
                        
                        ###
                        logits = Sty_cls(chars, mask)
                        Acc_batch = (logits.argmax(1) == tar_sty).sum().item() / len(tar_sty)

                        Acc_real += Acc_batch * len(tar_sty) / char_total
                print(f'Acc_real after epoch {epoch+1}: {Acc_real*100:.2f}%')
                logging.info(f'Acc_real after epoch {epoch+1}: {Acc_real*100:.2f}%')
                
                # test on syn data
                if syn_loader is not None:
                    Acc_syn = 0
                    char_total = syn_dset.hw_total()
                    with torch.no_grad():
                        for batched_data in syn_loader:
                            writer_id = batched_data['writer_id']
                            chars = batched_data['chars'] ; mask = batched_data['mask']
                            tar_sty = writer_id.to(torch.long)
                        
                            ###
                            logits = Sty_cls(chars, mask)
                            Acc_batch = (logits.argmax(1) == tar_sty).sum().item() / len(tar_sty)

                            Acc_syn += Acc_batch * len(tar_sty) / char_total
                    print(f'Acc_syn after epoch {epoch+1}: {Acc_syn*100:.2f}%')
                    logging.info(f'Acc_syn after epoch {epoch+1}: {Acc_syn*100:.2f}%')

                # save model
                torch.save(Sty_cls.state_dict(), os.path.join('data_ckpt/ckpt', f'sty_evaluator_{epoch+1}.pth'))
                # save lookup table
                np.save(os.path.join('data_ckpt/ckpt', f'lookup_idx2ori_on.npy'), train_dset.lookup_idx2ori)
                Sty_cls.train()

    pass


def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    args.log_name = "StyCls_on_res18_aug"
    args.batch_user = 60    # make sure that writer/batch can be divided by number of writers in dataset
    args.sample_per_user = 1
    args.n_reference = 15
    args.lr = 1e-3
    args.epoch_begin = 0
    args.epoch_end = 10000
    args.test_epoch = 1000
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
    args = get_args()
    train(args)
    pass

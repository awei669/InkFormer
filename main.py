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

'''This is a auto-regressive transformer-alike model for generating handwritting Chinese characters from standard characters.'''


def reconstruction_loss(
        param_GMM:  torch.FloatTensor, 
        pen_state: torch.FloatTensor,
        GT: torch.FloatTensor, 
        mask: torch.FloatTensor, 
        ):
    r"""
    Args:
        param_GMM (`torch.FloatTensor`): Parameters of GMM. tuple of (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        pen_state (`torch.FloatTensor`): Pen state. shape of (B,L,1)
        GT (`torch.FloatTensor`): Ground truth. shape of (B,L,3)
        mask (`torch.FloatTensor`): Mask. shape of (B,L)
    """
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = param_GMM
    GT_x = GT[:, :, 0:1] ; GT_y = GT[:, :, 1:2] ; GT_pen = GT[:, :, 2:]

    z_x = ((GT_x - mu_x) / sigma_x) ** 2
    z_y = ((GT_y - mu_y) / sigma_y) ** 2
    z_xy = (GT_x - mu_x) * (GT_y - mu_y) / (sigma_x * sigma_y)
    z = z_x + z_y - 2 * rho_xy * z_xy
    exp = torch.exp(-z / (2 * (1 - rho_xy ** 2)))
    norm = 2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)
    pdf = pi * exp / norm
    loss_xy = -torch.log(torch.sum(pdf, dim=-1) + 1e-6)
    loss_xy = torch.sum(loss_xy * mask) / torch.sum(mask)

    loss_pen = nn.BCEWithLogitsLoss(reduction='mean')(pen_state, (GT_pen + 1)/2) # Convert -1/1 to 0/1

    return loss_xy, loss_pen


def inter_writer_nce(f_sty: torch.FloatTensor, 
                     f_sty_mask: torch.FloatTensor,
                     w: int, 
                     tau: float = 0.07
                     ):
    '''
    inter-writer nce loss
    f_sty: shape=(B, n*l, 512). 
    f_sty_mask: shape=(B, n*l).
    w: int, number of writer in batch. 
    tau: float, temperature.
    '''
    s = len(f_sty) // w
    labels = torch.arange(w).unsqueeze(1).repeat(1, s).reshape(-1)
    pick_pos = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(f_sty.device, f_sty.dtype)
    pick_neg = 1. - pick_pos

    # simply compressing with global-ave-pooling
    stys_com = (f_sty * f_sty_mask.unsqueeze(-1)).sum(1) / f_sty_mask.sum(1, keepdim=True) # (B, 512)
    stys_com = stys_com / (stys_com.norm(dim=-1, keepdim=True) + 1e-6)  # normalize

    # similarity matrix
    sim = torch.matmul(stys_com, stys_com.t()) / tau
    sim_exp = torch.exp(sim)
    
    # nce of every pair
    nce = sim_exp / (sim_exp + (sim_exp * pick_neg).sum(1, keepdim=True))
    nce = torch.log(nce + 1e-6)
 
    # pick up positive pairs, instead of self pairs
    pick_pos_Nself = pick_pos * (1. - torch.eye(w*s)).to(f_sty.device, f_sty.dtype)
    nce = nce * pick_pos_Nself

    # loss
    loss = -nce.sum() / pick_pos_Nself.sum()

    return loss



def train(args):
    '''train function for training diffusion model.'''
    log_folder = f'logs/{args.log_name}'
    os.makedirs(log_folder, exist_ok=True)
    log_model = f'{log_folder}/model'
    os.makedirs(log_model, exist_ok=True)
    log_img = f'{log_folder}/img_train'
    os.makedirs(log_img, exist_ok=True)

    logging.basicConfig(
        filename=f"{log_folder}/train.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    ### prepare accelerator
    accelerator = Accelerator(
        even_batches=False,
    )

    dset = dataset_char.Dataset_Char(args)
    for hw_path in args.hw_path:
        dset.load_hw(hw_path)
    dset.load_std(args.std_path)
    sampler = dataset_char.Train_Batch_Sampler(args, dset)
    collate_fn = dataset_char.collate_fn(dset.Nmax, dset.format)
    loader = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers)

    ### model preparation
    model = Transformer_AR(
        in_dim=args.format, 
        embed_dim=512, 
        n_heads=16, 
        n_layers=4, 
        dropout=0.1, 
        M=20, 
    )
    ## load previous model for continue training
    if args.epoch_begin > 0:
        # must be breakdown in last training, so load the last model
        check_point = torch.load(f'data_ckpt/ckpt/ar_generator_{args.epoch_begin}.pth')
        model.load_state_dict({k.replace('module.', ''): v for k, v in check_point.items()})    # bad habits in multi-gpu training, `module` prefix occurs when using torch.save()
    # load pretrained model
    if args.pretrained is not None:
        check_point = torch.load(args.pretrained)
        check_point = {k.replace('module.', ''): v for k, v in check_point.items()}
        model.load_state_dict(check_point, strict=False)    # load the pretrained model, but not all parameters are needed

    optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_end, verbose=False, last_epoch=args.epoch_begin-1)

    ### record evertything
    if accelerator.is_main_process:
        logging.info(f"Args: {args}")
        logging.info(f"Model: {model}")
        logging.info(f"Optimizer: {optimizer}")
        logging.info(f"LR Scheduler: {vars(lr_scheduler)}")

    ### accelerator preparation
    model, optimizer, loader = accelerator.prepare(
        model, optimizer, loader
    )

    # if accelerator.num_processes > 1:
    #     model = model.module

    # writer = SummaryWriter('log/')
    log_losses = {
        'loss_xy': 0,
        'loss_pen': 0,
        'loss_nce': 0,
    } ; StepsPerEpoch = len(loader)

    pbar = tqdm(total=(args.epoch_end-args.epoch_begin)*len(loader), disable=not accelerator.is_main_process)
    for epoch in range(args.epoch_begin, args.epoch_end):
        pbar.set_description(f'Epoch: {epoch+1}/{args.epoch_end}')
        for batched_data in loader:
            writer_id = batched_data['writer_id']
            keys = batched_data['keys']
            hw_chars = batched_data['hw_chars'] ; hw_mask = batched_data['hw_mask']
            std_chars = batched_data['std_chars'] ; std_mask = batched_data['std_mask']
            sty_refs = batched_data['sty_refs'] ; sty_mask = batched_data['sty_mask']
            sty_stds = batched_data['sty_stds'] ; sty_stds_mask = batched_data['sty_stds_mask']

            ### arange input and output
            hw_input = hw_chars[:, :-1] ; mask_input = hw_mask[:, :-1]

            ###
            param_GMM, pen_state, f_sty, f_sty_mask = model(hw_input, mask_input, 
                                                            sty_refs, sty_mask, 
                                                            sty_stds, sty_stds_mask,
                                                            std_chars, std_mask
                                                            )
            
            loss = 0
            ### reconstruction loss
            loss_xy, loss_pen = reconstruction_loss(param_GMM, pen_state, hw_chars, hw_mask)
            loss += loss_xy + loss_pen*2
            ### inter-writer nce loss
            loss_nce = inter_writer_nce(f_sty, f_sty_mask, args.batch_user)
            loss += loss_nce

            # # gather losses from all gpus
            # loss = accelerator.gather(loss.repeat(hw_chars.shape[0])).mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                ## clip the gradients
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                pbar_report = {
                    'loss_xy': f"{loss_xy.item():.6f}",
                    'loss_pen': f"{loss_pen.item():.6f}",
                    'loss_nce': f"{loss_nce.item():.6f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                }
                pbar.update(1)
                pbar.set_postfix(pbar_report)

                ### logs losses
                log_losses['loss_xy'] += loss_xy.item() / StepsPerEpoch
                log_losses['loss_pen'] += loss_pen.item() / StepsPerEpoch
                log_losses['loss_nce'] += loss_nce.item() / StepsPerEpoch
        
        lr_scheduler.step()

        if accelerator.is_main_process:
            # print("hello, i'm main process")
            ### log the information
            log_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Epoch {epoch}, Step {pbar.n} => "
            for key, value in log_losses.items():
                log_message += f"{key}: {value:.6f} "
            log_message += f"lr: {optimizer.param_groups[0]['lr']:.2e} "
            # writer.add_scalar(key, value, epoch*len(loader)+pbar.n)
            logging.info(log_message)
            log_losses = {key: 0. for key in log_losses.keys()}   # reset the log_losses

            ### save model
            if (epoch+1) % args.save_epoch == 0:
                # save model
                torch.save(model.state_dict(), os.path.join('data_ckpt/ckpt', f'ar_generator_{epoch+1}.pth'))

            ### paint temporary results
            if (epoch+1) % args.vis_epoch == 0:
                n_examples = 5    # number of examples to visualize
                # vis_bar = tqdm(range(len(ddim.timesteps)), desc='Visualizing', leave=False)
                with torch.no_grad():
                    syn_seq = model.module.inference(
                        hw_chars.size(1),
                        sty_refs[:n_examples], sty_mask[:n_examples], 
                        sty_stds[:n_examples], sty_stds_mask[:n_examples],
                        std_chars[:n_examples], std_mask[:n_examples], 
                        )
                # vis_bar.update(1)

                canvas_size = 256 ; line_width = 2 ; point_radius = 1
                fig, axs = plt.subplots(n_examples, 4, figsize=(3*4, 3*n_examples))
                for idx_paint in range(n_examples):
                    ### paint standard
                    show = std_chars[idx_paint]
                    show = v2x(show.detach().cpu().numpy())
                    image_std = draw_n_save(show, None, canvas_size, line_width, point_radius, color=True, draw_pads=False)
                    axs[idx_paint, 0].imshow(image_std)
                    axs[idx_paint, 0].set_title('Standard Char')
                    ### paint synthetic handwritting
                    ## w/ padding
                    show = syn_seq[idx_paint]
                    show = v2x(show.detach().cpu().numpy())
                    image_syn = draw_n_save(show, None, canvas_size, line_width, point_radius, color=True, draw_pads=True)
                    axs[idx_paint, 1].imshow(image_syn)
                    axs[idx_paint, 1].set_title('Synthetic Char w/ Padding')
                    image_syn = draw_n_save(show, None, canvas_size, line_width, point_radius, color=True, draw_pads=False)
                    axs[idx_paint, 2].imshow(image_syn)
                    axs[idx_paint, 2].set_title('Synthetic Char w/o Padding')
                    ### paint handwritting
                    show = hw_chars[idx_paint]
                    show = v2x(show.detach().cpu().numpy())
                    image_hw = draw_n_save(show, None, canvas_size, line_width, point_radius, color=True, draw_pads=False )
                    axs[idx_paint, 3].imshow(image_hw)
                    axs[idx_paint, 3].set_title('Handwritting Char')
                for ax in axs.flatten():
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(log_img, f'{epoch+1}.png'))
                plt.close()
        else:
            # print("hello, i'm not main process")
            pass
        pass
    # writer.close()


def get_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-name', type=str, default=None)

    parser.add_argument('--batch-user', type=int, default=4)
    parser.add_argument('--sample-per-user', type=int, default=10)
    parser.add_argument('--n-reference', type=int, default=8)

    parser.add_argument('--format', type=int, default=3)
    parser.add_argument('--hw-path', type=str, nargs='+', default=None)
    parser.add_argument('--std-path', type=str, default=None)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch-begin', type=int, default=0)
    parser.add_argument('--epoch-end', type=int, default=5000)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--save-epoch', type=int, default=100)
    parser.add_argument('--vis-epoch', type=int, default=1)
    args = parser.parse_args()

    if args.hw_path is None:
        args.hw_path = ['data_ckpt/datas/train/Pot1.0Train_pp1.npy']
    if args.std_path is None:
        args.std_path = 'data_ckpt/datas/standard/stand_zi.npy'

    return args

def init(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


if __name__ == '__main__':
    args = get_args_train()
    init(args.seed)

    train(args)
    # try:
    #     train(args)
    # except Exception as e:
    #     logging.basicConfig(
    #         filename='error.log',
    #         filemode='a',
    #         format='%(asctime)s - %(levelname)s - %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S',
    #         level=logging.ERROR
    #     )
    #     logging.error("An error occurred during training", exc_info=True)


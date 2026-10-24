import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .ResNet1d import *


class LayerNorm_insert(nn.Module):
    def __init__(self, 
                 in_dim,
                 ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
    
    def forward(self, x):
        return self.norm(x.permute(0,2,1)).permute(0,2,1)


class An_Encoder(nn.Module):
    def __init__(self,
                 *args
                 ) -> None:
        super().__init__()
        self.enc = resnet18_1d(*args)
        self.pooling = att_pooling(512)

    def forward(self, x, mask):
        # 1. encode
        x = self.enc(x.permute(0, 2, 1)).permute(0, 2, 1)
        mask = nn.Upsample(size=x.size(1), mode='nearest')(mask.unsqueeze(1)).squeeze(1)

        # 2. pooling
        x_p = self.pooling(x, mask)
        return (x, x_p)



class std2sty_Encoder(nn.Module):
    def __init__(self,
                 in_dim
                 ) -> None:
        super().__init__()
        self.in_layer = nn.Conv1d(in_dim, 64, 7, 1, 3, bias=False)
        self.Diff_block1 = self.sandwich(64, 64)
        self.Diff_block2 = self.sandwich(64, 128)
        self.Diff_block3 = self.sandwich(128, 256)
        self.Diff_block4 = self.sandwich(256, 512)
        # self.end2 = BasicBlock(128, 128)
        # self.end3 = BasicBlock(256, 256)
        self.end4 = BasicBlock(512, 512)

    class sandwich(nn.Module):
        def __init__(self, 
                     in_dim, 
                     out_dim, 
                     ) -> None:
            super().__init__()
            if in_dim != out_dim:
                downsample = conv1x1(in_dim, out_dim, 2)
                stride = 2
            else:
                downsample = None
                stride = 1

            self.conv1_sty = BasicBlock(in_dim, out_dim, stride, downsample)
            self.conv1_std = BasicBlock(in_dim, out_dim, stride, downsample)
            self.sub = sub_layer(out_dim)
            self.conv2_sty = BasicBlock(out_dim, out_dim)
            self.conv2_std = BasicBlock(out_dim, out_dim)
        
        def forward(self, x_sty, x_std):
            x_sty = self.conv1_sty(x_sty)
            x_std = self.conv1_std(x_std)
            x_sty, x_std = self.sub(x_sty, x_std)
            x_sty = self.conv2_sty(x_sty)
            x_std = self.conv2_std(x_std)
            return x_sty, x_std

    def get_mask(self, mask_a, mask_b, L_now):
        mask_a = nn.Upsample(size=L_now, mode='nearest')(mask_a.unsqueeze(1)).squeeze(1)
        mask_b = nn.Upsample(size=L_now, mode='nearest')(mask_b.unsqueeze(1)).squeeze(1)
        x_mask = (mask_a.bool() | mask_b.bool()).to(mask_a.dtype)
        return x_mask

    def forward(self, 
                sty_chars, sty_mask, 
                sty_stds, sty_stds_mask, 
                channel_last = True
                ):
        if channel_last:
            sty_chars = sty_chars.permute(0, 2, 1)
            sty_stds = sty_stds.permute(0, 2, 1)
        
        x_sty = self.in_layer(sty_chars)
        x_std = self.in_layer(sty_stds)

        x_diff = [] ; x_mask = []

        x_sty, x_std = self.Diff_block1(x_sty, x_std)

        x_sty, x_std = self.Diff_block2(x_sty, x_std)
        # x_this = self.end2(torch.abs(x_sty - x_std))
        # x_diff.append(x_this)
        # x_mask.append(self.get_mask(sty_mask, sty_stds_mask, x_this.size(-1)))

        x_sty, x_std = self.Diff_block3(x_sty, x_std)
        # x_this = self.end3(torch.abs(x_sty - x_std))
        # x_diff.append(x_this)
        # x_mask.append(self.get_mask(sty_mask, sty_stds_mask, x_this.size(-1)))

        x_sty, x_std = self.Diff_block4(x_sty, x_std)
        x_this = self.end4(torch.abs(x_sty - x_std))    # 也许减完后加一个layernorm？
        x_diff.append(x_this)
        x_mask.append(self.get_mask(sty_mask, sty_stds_mask, x_this.size(-1)))

        if channel_last:
            x_diff = [x.permute(0, 2, 1) for x in x_diff]

        return x_diff, x_mask


class sub_layer(nn.Module):
    def __init__(self, 
                 in_dim, 
                 left_out=True, 
                 right_out=True
                 ) -> None:
        super().__init__()
        self.left_out = left_out
        self.right_out = right_out
        self.ln_diff = LayerNorm_insert(in_dim)
        if left_out:
            self.conv1 = nn.Conv1d(in_dim, in_dim, 1, 1, 0)
        if right_out:
            self.conv2 = nn.Conv1d(in_dim, in_dim, 1, 1, 0)

    def forward(self, a, b):
        diff_ab = torch.abs(a - b)
        diff_ab = self.ln_diff(diff_ab)
        if self.left_out:
            a = self.conv1(diff_ab) * a
        if self.right_out:
            b = self.conv2(diff_ab) * b
        
        if self.left_out and self.right_out:    return a, b
        elif self.left_out:     return a
        elif self.right_out:    return b


def positional_encoding(pos, d_model):
    pe = torch.zeros(pos, d_model)
    position = torch.arange(0, pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Transformer_Encoder(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, causal=False, pos=None):
        super(Transformer_Encoder, self).__init__()
        self.causal = causal
        self.pos = pos

        if pos: # positional encoding
            self.register_buffer('pos_emb', positional_encoding(pos, d_model))

        self.emb = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, src, src_mask):
        '''
        src: torch.Tensor, shape=(B, L, in_dim), input data
        src_mask: torch.Tensor, shape=(B, L), mask for input data
        '''
        if self.causal:
            # 生成一个上三角矩阵作为mask，用于掩盖未来的token
            mask = torch.triu(torch.ones(src.size(1), src.size(1)), 1).to(src.device)
        else:
            mask = None

        src = self.emb(src)
        if self.pos:
            src = src + self.pos_emb[:src.size(1)]

        output = self.transformer_encoder(src, mask=mask, src_key_padding_mask=src_mask.permute(1, 0))
        return output

class RNN_Encoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 inner_dim, 
                 layers=3,
                 bidirectional=True,
                 type='gru',
                 ) -> None:
        super().__init__()
        if type == 'lstm':
            self.rnn = nn.LSTM(in_dim, inner_dim, layers, batch_first=True, bidirectional=bidirectional)
        elif type == 'gru':
            self.rnn = nn.GRU(in_dim, inner_dim, layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError('Unsupported RNN type')
        
        if bidirectional:
            self.linear = nn.Linear(inner_dim*2, inner_dim)
        
    def forward(self, x, mask):
        '''x: torch.Tensor, shape=(B, L, in_dim), input data
        mask: torch.Tensor, shape=(B, L), mask for input data'''
        lengths = mask.sum(1).to('cpu', torch.int64)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x_packed, _ = self.rnn(x_packed)
        hs, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)

        if self.rnn.bidirectional:
            hs = self.linear(hs)

        return x
    
    
class att_pooling(nn.Module):
    def __init__(self, 
                 in_dim,
                 ) -> None:
        super().__init__()
        self.V = nn.Linear(in_dim, in_dim, bias=False)
    
    def forward(self, x, mask):
        '''x: (B, L, in_dim), mask: (B, L)'''
        x_bar = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        score = torch.einsum('bd, bld->bl', x_bar, self.V(x))
        mask = mask.to(torch.bool)
        score = score.masked_fill(~mask, -1e9)
        score = torch.softmax(score, dim=-1)
        x = torch.einsum('bl, bld->bd', score, x)
        return x


    

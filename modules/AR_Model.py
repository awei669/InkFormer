import torch
import torch.nn as nn
import torchvision.models as models
import math

from .MobileNet1d import MobileNetV3_Small_1D
from .ResNet1d import resnet18_1d
from .encoders import std2sty_Encoder

'''A variational autoencoder with an autoregressive decoder'''


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_head, max_len=1024):
        super(RelPositionalEncoding, self).__init__()
        self.d_head = d_head
        self.max_len = max_len
        
        # Create relative position embedding
        self.rel_pos_emb = torch.zeros(2 * max_len - 1, d_head)
        
        # Initialize using sine/cosine
        position = torch.arange(-(max_len-1), max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_head, 2).float() * (-math.log(10000.0) / d_head))
        
        self.rel_pos_emb[:, 0::2] = torch.sin(position * div_term)
        self.rel_pos_emb[:, 1::2] = torch.cos(position * div_term)

        self.rel_pos_emb = nn.Parameter(self.rel_pos_emb)
    
    def forward(self, q, seq_len_k):
        """
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            seq_len_k: Length of key sequence
        Returns:
            Relative positional bias to be added to attention scores
        """
        batch_size, seq_len_q, _ = q.shape
        q = q.reshape(*q.shape[:2], self.d_head, -1)
        q = q.permute(0, 3, 1, 2).reshape(-1, seq_len_q, self.d_head)
        q = q / (self.d_head ** 0.5)
        
        # Generate all relative position indices
        q_pos = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=q.device).unsqueeze(0)
        relative_pos = k_pos - q_pos  # shape: (seq_len_q, seq_len_k)
        
        # Shift indices to be non-negative
        relative_pos = relative_pos + (self.max_len - 1)
        
        # Clip indices that might go out of bounds
        relative_pos = torch.clamp(relative_pos, 0, 2 * self.max_len - 2)
        
        # Get embeddings for each relative position
        rel_pos_bias = self.rel_pos_emb[relative_pos]  # shape: (seq_len_q, seq_len_k, d_model)
        
        # Project to obtain attention bias
        pos_attn_bias = torch.einsum('bqd,qkd->bqk', q, rel_pos_bias)  # shape: (batch_size*N_head, seq_len_q, seq_len_k)
        
        return pos_attn_bias


def get_q(x:torch.tensor, attn_layer:nn.MultiheadAttention):
    if attn_layer.in_proj_weight != None:
        q = x @ attn_layer.in_proj_weight.T
        q += attn_layer.in_proj_bias
        q = q[..., :x.size(-1)]
    else:
        q = x @ attn_layer.q_proj_weight.T
        q += attn_layer.in_proj_bias[..., :x.size(-1)]
    return q


class Self_Attn_Layer(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 n_heads: int,
                 dropout: float=0.1,
                 ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffw = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout),
        ) 
        self.ln2 = nn.LayerNorm(embed_dim)

        self.rel_pos = RelPositionalEncoding(embed_dim // n_heads)

    def forward(self, 
                x: torch.FloatTensor, 
                mask: torch.FloatTensor,
                causal: bool = True,
                ):
        r"""
        Args:
            x (`torch.FloatTensor`): Input tensor. shape of (B,L,C)
            mask (`torch.FloatTensor`): Mask. shape of (B,L)
        """
        B, L, C = x.size()
        ### relative position encoding
        relpos = self.rel_pos(
            q = get_q(x, self.attn_layer), 
            seq_len_k = L
            )
        ### causal mask
        if causal: 
            causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.float), diagonal=1) * -1e9
            causal_mask = causal_mask.unsqueeze(0).expand(B*self.n_heads, -1, -1)
            relpos = relpos + causal_mask
        ### self-attention
        mask_v = mask.bool().logical_not()    # In nn.MultiheadAttention, True for padding, False for non-padding, and float will be directly added to key
        x = self.ln1(x + self.attn_layer(x, x, x, attn_mask=relpos, key_padding_mask=mask_v)[0])
        x = self.ln2(x + self.ffw(x))
        return x


class Cross_Attn_Layer(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 condi_dim: int,
                 n_heads: int,
                 dropout: float=0.1,
                 ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.cross_attn_cont = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            kdim=condi_dim,
            vdim=condi_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn_sty = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            kdim=condi_dim,
            vdim=condi_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ffw = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout),
        ) 
        self.ln4 = nn.LayerNorm(embed_dim)

        self.rel_pos1 = RelPositionalEncoding(embed_dim // n_heads)
        self.rel_pos2 = RelPositionalEncoding(embed_dim // n_heads)
        self.rel_pos3 = RelPositionalEncoding(embed_dim // n_heads)
    
    def forward(self,
                x: torch.FloatTensor,
                cont: torch.FloatTensor,
                cont_mask: torch.FloatTensor,
                sty: torch.FloatTensor, 
                sty_mask: torch.FloatTensor,
                causal: bool = True,
                ):
        r"""
        Args:
            x (`torch.FloatTensor`): Input tensor. shape of (B,L,C)
            mask (`torch.FloatTensor`): Mask. shape of (B,L)
            condi (`torch.FloatTensor`): Conditional tensor. shape of (B,nL,C')
            condi_mask (`torch.FloatTensor`): Conditional mask. shape of (B,nL)
            causal (`bool`, *optional*, defaults to `True`): whether to use causal mask.
        """
        B, L, C = x.size()

        ### relative position encoding
        relpos1 = self.rel_pos1(
            q = get_q(x, self.self_attn), 
            seq_len_k = L
            )
        ### causal mask
        if causal: 
            causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.float), diagonal=1) * -1e9
            causal_mask = causal_mask.unsqueeze(0).expand(B*self.n_heads, -1, -1)
            relpos1 = relpos1 + causal_mask
        ### self-attention
        x = self.ln1(x + self.self_attn(x, x, x, attn_mask=relpos1)[0])

        ### relative position encoding
        relpos2 = self.rel_pos2(
            q = get_q(x, self.cross_attn_cont),
            seq_len_k = cont.size(1)
            )
        ### cross-attention
        cont_mask_v = cont_mask.bool().logical_not()
        x = self.ln2(x + self.cross_attn_cont(x, cont, cont, attn_mask=relpos2, key_padding_mask=cont_mask_v)[0])

        ### relative position encoding
        relpos3 = self.rel_pos3(
            q = get_q(x, self.cross_attn_sty),
            seq_len_k = sty.size(1)
            )
        ### cross-attention
        sty_mask_v = sty_mask.bool().logical_not()
        x = self.ln3(x + self.cross_attn_sty(x, sty, sty, attn_mask=relpos3, key_padding_mask=sty_mask_v)[0])
        
        ### feed-forward
        x = self.ln4(x + self.ffw(x))
        return x


class sos_generater(nn.Module):
    def __init__(self, 
                    embed_dim: int,
                    n_heads: int,
                    dropout: float=0.1,
                    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.cross1 = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            kdim=embed_dim,
            vdim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.cross2 = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            kdim=embed_dim,
            vdim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffw = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln3 = nn.LayerNorm(embed_dim)

        self.rel_pos1 = RelPositionalEncoding(embed_dim // n_heads)
    
    def forward(self,
                sty: torch.FloatTensor,
                sty_mask: torch.FloatTensor,
                cont: torch.FloatTensor,
                cont_mask: torch.FloatTensor,
                ):
        r"""
        Args:
            sty (`torch.FloatTensor`): Style tensor. shape of (B,nL,C)
            sty_mask (`torch.FloatTensor`): Style mask. shape of (B,nL)
            cont (`torch.FloatTensor`): Content tensor. shape of (B,L,C)
            cont_mask (`torch.FloatTensor`): Content mask. shape of (B,L)
        """
        B, nL, C = sty.size()
        L = cont.size(1)

        ### relative position encoding
        relpos1 = self.rel_pos1(
            q = get_q(sty, self.cross1), 
            seq_len_k = L
            )
        ### cross-attention
        cont_mask_v = cont_mask.bool().logical_not()
        sty2cont = self.cross1(sty, cont, cont, attn_mask=relpos1, key_padding_mask=cont_mask_v)[0]
        sty2cont = self.ln1(sty2cont)

        ### average content
        mean_cont = (cont * cont_mask.unsqueeze(2)).sum(1) / cont_mask.sum(1, keepdim=True)  # (B,C)
        mean_cont = mean_cont.unsqueeze(1)  # (B,1,C)

        ### cross-attention
        sty_mask_v = sty_mask.bool().logical_not()
        sos = self.cross2(mean_cont, sty2cont, sty2cont, key_padding_mask=sty_mask_v)[0]
        sos = self.ln2(sos)
        sos = self.ln3(sos + self.ffw(sos))

        return sos



class Transformer_AR(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 embed_dim: int=512, 
                 n_heads: int=16,
                 n_layers: int=2, 
                 dropout: float=0.1,
                 M: int=20, 
                 ) -> None:
        super().__init__()
        self.M = M

        self.embedder = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim)
        )
        self.decoder = nn.ModuleList([
            Cross_Attn_Layer(embed_dim, embed_dim, n_heads, dropout) for _ in range(n_layers)
        ])

        self.cont_enc = MobileNetV3_Small_1D(in_dim=in_dim)

        self.sty_enc = std2sty_Encoder(in_dim=in_dim)
        # self.sty_enc = MobileNetV3_Small_1D(in_dim=in_dim)
        # self.sty_enc = resnet18_1d(n_in=in_dim, require_multi=False)

        self.proj2param = nn.Linear(embed_dim, 6*M+1)
        # 6 parameters are pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy. 
        # The rest parameters are pen state.

        self.sos_gen = sos_generater(embed_dim, n_heads, dropout)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, 
                hw: torch.FloatTensor, hw_mask: torch.FloatTensor,
                sty: torch.FloatTensor, sty_mask: torch.FloatTensor,
                sty_std: torch.FloatTensor, sty_std_mask: torch.FloatTensor,
                cont: torch.FloatTensor, cont_mask: torch.FloatTensor,
                ):
        r'''
        Args:
            hw (`torch.FloatTensor`): Input tensor. shape of (B,L,C)
            hw_mask (`torch.FloatTensor`): Mask. shape of (B,L)
            sty (`torch.FloatTensor`): Style tensor. shape of (B,n,L,C)
            sty_mask (`torch.FloatTensor`): Style mask. shape of (B,n,L)
            cont (`torch.FloatTensor`): Content tensor. shape of (B,L,C)
            cont_mask (`torch.FloatTensor`): Content mask. shape of (B,L)
        '''
        B, n, L = sty.shape[:3]
        sty = sty.reshape(-1, *sty.shape[2:])                   # (B*n,L,3)
        sty_mask = sty_mask.reshape(-1, *sty_mask.shape[2:])    # (B*n,L)
        sty_std = sty_std.reshape(-1, *sty_std.shape[2:])       # (B*n,L,3)
        sty_std_mask = sty_std_mask.reshape(-1, *sty_std_mask.shape[2:])
        ## Our std2sty encoder
        sty, sty_mask = self.sty_enc(
            sty, sty_mask, 
            sty_std, sty_std_mask, 
            channel_last=True
        )
        ## only the last scale feature
        sty = sty[-1]               # (B*n,L1,C)
        sty_mask = sty_mask[-1]     # (B*n,L1)
        L1 = sty.size(1)                # L1 = L/8
        sty = sty.reshape(B, n*L1, -1)          # (B,n*L1,C)
        sty_mask = sty_mask.reshape(B, n*L1)    # (B,n*L1)

        cont = self.cont_enc(cont.permute(0,2,1)).permute(0,2,1)
        cont_mask = nn.Upsample(size=cont.size(1), mode='nearest')(cont_mask.unsqueeze(1)).squeeze(1)   # (B,L1)

        sos = self.sos_gen(
            sty, sty_mask, 
            cont, cont_mask
        )

        x = self.embedder(hw)   # (B,L,C)
        x = torch.concat([sos, x], dim=1)    # (B,1+L,C)
        for layer in self.decoder:
            x = layer(x, cont, cont_mask, sty, sty_mask)

        x = self.proj2param(x)
        param_GMM = x[..., :6*self.M]
        pen_state = x[..., 6*self.M:]

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(param_GMM, self.M, dim=-1)
        pi = torch.softmax(pi, dim=-1)
        sigma_x = torch.minimum(torch.exp(sigma_x), torch.tensor(500.0, device=sigma_x.device))
        sigma_y = torch.minimum(torch.exp(sigma_y), torch.tensor(500.0, device=sigma_y.device))
        rho_xy = torch.tanh(rho_xy)

        param_GMM = (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

        # pen_state = nn.Sigmoid()(pen_state)    # 如果用nn.BCEWithLogitsLoss的话，不需要sigmoid

        return param_GMM, pen_state, sty, sty_mask

    def sample(self, 
               GMM_param: torch.FloatTensor,
               pen_state: torch.FloatTensor,
               greedy: bool = False,
               ) -> torch.FloatTensor:
        r"""
        Args:
            GMM_param (`torch.FloatTensor`): Parameters of GMM. tuple of (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
            pen_state (`torch.FloatTensor`): Pen state. shape of (B,L,1)
            greedy (`bool`, *optional*, defaults to `False`):
                Whether to sample greedily.
        """
        pen_state = torch.sign(pen_state)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = GMM_param

        B, L, M = pi.size()
        pi = pi.view(B*L, M)
        idx = torch.multinomial(pi, 1)
        idx = idx.view(B, L, 1)

        mu_x = torch.gather(mu_x, -1, idx)
        mu_y = torch.gather(mu_y, -1, idx)
        sigma_x = torch.gather(sigma_x, -1, idx)
        sigma_y = torch.gather(sigma_y, -1, idx)
        rho_xy = torch.gather(rho_xy, -1, idx)

        if greedy:
            x = mu_x ; y = mu_y
        else:
            z_x = torch.randn_like(mu_x)
            z_y = torch.randn_like(mu_y)
            x = mu_x + sigma_x * z_x
            y = mu_y + sigma_y * (rho_xy * z_x + torch.sqrt(1 - rho_xy ** 2) * z_y)

        return torch.cat([x, y, pen_state], dim=-1)

    def inference(self, 
                  max_len: int,
                  sty: torch.FloatTensor, sty_mask: torch.FloatTensor, 
                  sty_std: torch.FloatTensor, sty_std_mask: torch.FloatTensor,
                  cont: torch.FloatTensor, cont_mask: torch.FloatTensor,
                  ):
        r'''
        Args:
            sty (`torch.FloatTensor`): Style tensor. shape of (B,n,L,C)
            sty_mask (`torch.FloatTensor`): Style mask. shape of (B,n,L)
            cont (`torch.FloatTensor`): Content tensor. shape of (B,L,C)
            cont_mask (`torch.FloatTensor`): Content mask. shape of (B,L)
        '''
        B, n, L = sty.shape[:3]
        sty = sty.reshape(-1, *sty.shape[2:])                   # (B*n,L,3)
        sty_mask = sty_mask.reshape(-1, *sty_mask.shape[2:])    # (B*n,L)
        sty_std = sty_std.reshape(-1, *sty_std.shape[2:])       # (B*n,L,3)
        sty_std_mask = sty_std_mask.reshape(-1, *sty_std_mask.shape[2:])
        ## Our std2sty encoder
        sty, sty_mask = self.sty_enc(
            sty, sty_mask, 
            sty_std, sty_std_mask, 
            channel_last=True
        )
        ## only the last scale feature
        sty = sty[-1]               # (B*n,L1,C)
        sty_mask = sty_mask[-1]     # (B*n,L1)
        L1 = sty.size(1)                # L1 = L/8
        sty = sty.reshape(B, n*L1, -1)          # (B,n*L1,C)
        sty_mask = sty_mask.reshape(B, n*L1)    # (B,n*L1)

        cont = self.cont_enc(cont.permute(0,2,1)).permute(0,2,1)
        cont_mask = nn.Upsample(size=cont.size(1), mode='nearest')(cont_mask.unsqueeze(1)).squeeze(1)   # (B,L1)

        sos = self.sos_gen(
            sty, sty_mask, 
            cont, cont_mask
        )

        syn_seq = None
        for _ in range(max_len):
            if syn_seq is None:
                x = sos    # (B,1,C)
            else:
                x = self.embedder(syn_seq)   # (B,L,C)
                x = torch.concat([sos, x], dim=1)
            
            for layer in self.decoder:
                x = layer(x, cont, cont_mask, sty, sty_mask)
            
            x = self.proj2param(x)
            param_GMM = x[:, -1:, :6*self.M]    # (B,1,6M)
            pen_state = x[:, -1:, 6*self.M:]    # (B,1,1)

            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(param_GMM, self.M, dim=-1)
            pi = torch.softmax(pi, dim=-1)
            sigma_x = torch.minimum(torch.exp(sigma_x), torch.tensor(500.0, device=sigma_x.device))
            sigma_y = torch.minimum(torch.exp(sigma_y), torch.tensor(500.0, device=sigma_y.device))
            rho_xy = torch.tanh(rho_xy)

            param_GMM = (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

            syn_now = self.sample(param_GMM, pen_state, greedy=True)    # (B,1,3)
            if syn_seq is None:
                syn_seq = syn_now
            else:
                syn_seq = torch.cat([syn_seq, syn_now], dim=1)
            
            if syn_now[:, 0, 2].sum() == -syn_now.size(0):  # 直到所有样本的pen_state都为-1
                break
        return syn_seq

    def extract_style_feature(
        self,
        sty,
        sty_mask,
        sty_std,
        sty_std_mask,
    ):

        B, n, L = sty.shape[:3]

        sty = sty.reshape(
            -1,
            *sty.shape[2:]
        )

        sty_mask = sty_mask.reshape(
            -1,
            *sty_mask.shape[2:]
        )

        sty_std = sty_std.reshape(
            -1,
            *sty_std.shape[2:]
        )

        sty_std_mask = sty_std_mask.reshape(
            -1,
            *sty_std_mask.shape[2:]
        )


        sty, sty_mask = self.sty_enc(
            sty,
            sty_mask,
            sty_std,
            sty_std_mask,
            channel_last=True
        )


        # 只取最后尺度
        sty = sty[-1]
        sty_mask = sty_mask[-1]


        L1 = sty.shape[1]


        # 恢复 reference 维度
        sty = sty.reshape(
            B,
            n,
            L1,
            -1
        )

        sty_mask = sty_mask.reshape(
            B,
            n,
            L1
        )


        return sty, sty_mask
            

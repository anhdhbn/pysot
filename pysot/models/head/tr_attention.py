import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from copy import deepcopy
from pysot.models.head.embedding import PositionEmbeddingSine
from pysot.models.head.rpn import RPN

from pysot.models.head.attention import with_pos_embed, getClones, clone_module

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads:int, dropout: float, dim_feedforward: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dims, num_heads, dropout=dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(hidden_dims)
        
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(hidden_dims)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dims, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dims),
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor, src: Tensor):
        attn_out, attn_output_weights = self.attention(query=q, key=k, value=v)
        x = self.norm_sa(self.dropout_sa(attn_out) + src) # q or kernel

        # Add skip connection, run through normalization and finally dropout
        forward = self.feed_forward(x)
        out = self.norm_ff(self.dropout_ff(forward) + x)
        return out

class EncodeDecodeLayer(nn.Module):
    def __init__(self, attention_layer: nn.Module):
        super(EncodeDecodeLayer, self).__init__()
        self.template_attn = clone_module(attention_layer)
        self.search_attn = clone_module(attention_layer)
        self.attn = clone_module(attention_layer)

    def forward(self,
                template: Tensor, 
                search: Tensor,
                pos_template: Tensor,
                pos_search: Tensor
            ):
        q_t = k_t = with_pos_embed(template, pos_template)
        out_template = self.template_attn(q_t, k_t, template, src=template)

        q_s = k_s = with_pos_embed(search, pos_search)
        out_search = self.search_attn(q_s, k_s, search, src=search)

        k_en_de = with_pos_embed(out_template, pos_template)
        out = self.attn(out_search, k_en_de, out_template, src=out_search)
        return out

class GlobalAttention(nn.Module):
    def __init__(self, attention_layer: nn.Module, num_layers):
        super(GlobalAttention, self).__init__()
        self.layers = getClones(attention_layer, num_layers)

    def forward(self, features, pos):
        out = features
        for layer in self.layers:
            q = k = with_pos_embed(out, pos)
            out = layer(q, k, features, src=out)
        return out

class Attention(nn.Module):
    def __init__(self, attn_layer: nn.Module, num_global_layers: int, hidden_dims: int, out_channels:int):
        super(Attention, self).__init__()

        self.ende_atten = EncodeDecodeLayer(attn_layer)
        # self.self_attn = GlobalAttention(attn_layer, num_global_layers)

        self.head = nn.Conv2d(hidden_dims, out_channels, kernel_size=1)
    
    def forward(self, 
                kernel: Tensor, 
                search: Tensor,
                pos_kernel: Tensor,
                pos_search: Tensor,
                size) -> Tensor:
        out = self.ende_atten(kernel, search, pos_kernel, pos_search)
        # out = self.self_attn(out, pos_search)
        # 49, 32, 256 => 32, 256, 7, 7
        hw, bs, c = out.shape
        h , w = size
        out = out.permute(1, 2, 0).view(bs, c, h, w)
        out = self.head(out)
        return out

    

class AttnRPN(RPN):
    def __init__(self, attention_layer, hidden_dims, num_global_layers, anchor_num=5):
        super(AttnRPN, self).__init__()
        self.cls = Attention(attention_layer, num_global_layers, hidden_dims, anchor_num*2)
        self.loc = Attention(attention_layer, num_global_layers, hidden_dims, anchor_num*4)
        self.embedding = PositionEmbeddingSine(hidden_dims // 2)
    
    def forward(self, z_f, x_f):
        # z_f [32, 256, 7, 7]
        # x_f [32, 256, 31, 31]
        bs, c, h, w = x_f.shape

        (pos_zf, _) = self.embedding(z_f)
        (pos_xf, _) = self.embedding(x_f)

        template = z_f.flatten(2).permute(2, 0, 1) # HWxNxC [49, 32, 256]
        search = x_f.flatten(2).permute(2, 0, 1) # HWxNxC [961, 32, 256]

        pos_template = pos_zf.flatten(2).permute(2, 0, 1) # HWxNxC
        pos_search = pos_xf.flatten(2).permute(2, 0, 1) # HWxNxC

        cls = self.cls(template, search, pos_template, pos_search, [h, w])
        loc = self.loc(template, search, pos_template, pos_search, [h, w])

        return cls, loc

class MultiAttnRPNv2(RPN):
    def __init__(self, 
                anchor_num, 
                in_channels, 
                weighted=False, 
                hidden_dims = 256, 
                num_heads=8,
                num_layers=6,
                dim_feed_forward=1024,
                dropout=0.1,
                ):
        super(MultiAttnRPNv2, self).__init__()
        attention_layer = AttentionLayer(hidden_dims, num_heads, dropout, dim_feed_forward)

        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('attnrpn'+str(i+2),
                    AttnRPN(attention_layer, hidden_dims, num_layers, anchor_num))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'attnrpn'+str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)
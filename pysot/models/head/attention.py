import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from copy import deepcopy
from pysot.models.head.embedding import PositionEmbeddingSine
from pysot.models.head.rpn import RPN

def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

def getClones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads:int, dropout: float, dim_feedforward: int, template_as_query: True):
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
        self.template_as_query = template_as_query

    def forward(self, kernel, search, pos_kernel, pos_search):
        if self.template_as_query:
            return self.forward_template_as_query(kernel, search, pos_kernel, pos_search)
        else:
            return self.forward_search_as_query(kernel, search, pos_kernel, pos_search)

    def forward_template_as_query(self, kernel, search, pos_kernel, pos_search):
        q = with_pos_embed(kernel, pos_kernel)
        k = with_pos_embed(search, pos_search)
        v = search

        attn_out, attn_output_weights = self.attention(query=q, key=k, value=v)
        x = self.norm_sa(self.dropout_sa(attn_out) + kernel) # q or kernel

        # Add skip connection, run through normalization and finally dropout
        forward = self.feed_forward(x)
        out = self.norm_ff(self.dropout_ff(forward) + x)
        return out

    def forward_search_as_query(self, kernel, search, pos_kernel, pos_search):
        q = with_pos_embed(search, pos_search)
        k = with_pos_embed(kernel, pos_kernel)
        v = kernel

        attn_out, attn_output_weights = self.attention(query=q, key=k, value=v)
        x = self.norm_sa(self.dropout_sa(attn_out) + search) # q or kernel

        # Add skip connection, run through normalization and finally dropout
        forward = self.feed_forward(x)
        out = self.norm_ff(self.dropout_ff(forward) + x)
        return out

class Attention(nn.Module):
    def __init__(self, attn_layer: nn.Module, num_layers: int, hidden_dims: int, out_channels:int, template_as_query: bool):
        super(Attention, self).__init__()
        self.layers = getClones(attn_layer, num_layers)

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, out_channels, kernel_size=1)
        )
        self.template_as_query = template_as_query
        
    def forward(self, 
                kernel: Tensor, 
                search: Tensor,
                pos_kernel: Tensor,
                pos_search: Tensor,
                size) -> Tensor:
        out = kernel if self.template_as_query else search
        for layer in self.layers:
            if self.template_as_query:
                out = layer(out, search, pos_kernel, pos_search)
            else:
                out = layer(kernel, out, pos_kernel, pos_search)
        
        # 49, 32, 256 => 32, 256, 7, 7
        hw, bs, c = out.shape
        h , w = size
        out = out.permute(1, 2, 0).view(bs, c, h, w)
        out = self.head(out)
        return out

class AttnRPN(RPN):
    def __init__(self, attention_layer, hidden_dims, num_layers, anchor_num=5, template_as_query=True):
        super(AttnRPN, self).__init__()
        self.cls = Attention(attention_layer, num_layers, hidden_dims, anchor_num*2, template_as_query)
        self.loc = Attention(attention_layer, num_layers, hidden_dims, anchor_num*4, template_as_query)

        self.embedding = PositionEmbeddingSine(hidden_dims // 2)
        self.template_as_query = template_as_query

    def forward(self, z_f, x_f):
        # z_f [32, 256, 7, 7]
        # x_f [32, 256, 31, 31]
        bs, c, h, w = z_f.shape if self.template_as_query else x_f.shape

        (pos_zf, _) = self.embedding(z_f)
        (pos_xf, _) = self.embedding(x_f)

        template = z_f.flatten(2).permute(2, 0, 1) # HWxNxC [49, 32, 256]
        search = x_f.flatten(2).permute(2, 0, 1) # HWxNxC [961, 32, 256]

        pos_template = pos_zf.flatten(2).permute(2, 0, 1) # HWxNxC
        pos_search = pos_xf.flatten(2).permute(2, 0, 1) # HWxNxC

        cls = self.cls(template, search, pos_template, pos_search, [h, w])
        loc = self.loc(template, search, pos_template, pos_search, [h, w])

        return cls, loc

class MultiAttnRPN(RPN):
    def __init__(self, 
                anchor_num, 
                in_channels, 
                weighted=False, 
                hidden_dims = 256, 
                num_heads=8,
                num_layers=6,
                dim_feed_forward=1024,
                dropout=0.1,
                template_as_query=True):
        super(MultiAttnRPN, self).__init__()
        attention_layer = AttentionLayer(hidden_dims, num_heads, dropout, dim_feed_forward, template_as_query)

        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('attnrpn'+str(i+2),
                    AttnRPN(attention_layer, hidden_dims, num_layers, anchor_num, template_as_query))
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
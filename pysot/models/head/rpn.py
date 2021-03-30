# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from copy import deepcopy

from pysot.core.xcorr import xcorr_fast, xcorr_depthwise
from pysot.models.init_weight import init_weights
from pysot.models.head.embedding import PositionEmbeddingSine
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
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

def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

def getClones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])
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

    def forward(self, kernel, search, pos_kernel, pos_search):
        q = with_pos_embed(kernel, pos_kernel)
        k = with_pos_embed(search, pos_search)
        v = search

        attn_out, attn_output_weights = self.self_attention(query=q, key=k, value=v)
        x = self.norm_sa(self.dropout_sa(attn_out) + kernel) # q or kernel

        # Add skip connection, run through normalization and finally dropout
        forward = self.feed_forward(x)
        out = self.norm_ff(self.dropout_ff(forward) + x)
        return out

class Attention(nn.Module):
    def __init__(self, attn_layer: nn.Module, num_layers: int):
        super(Attention, self).__init__()
        self.layers = getClones(attn_layer, num_layers)

    def forward(self, 
                kernel: Tensor, 
                search: Tensor,
                pos_kernel: Tensor,
                pos_search: Tensor) -> Tensor:
        out = kernel
        for layer in self.layers:
            out = layer(out, search, pos_kernel, pos_search)
        return out

class Tr2RPN(RPN):
    def __init__(self, attention_layer, hidden_dims, num_layers, anchor_num=5, in_channels=256, out_channels=256):
        super(Tr2RPN, self).__init__()
        self.cls = Attention(attention_layer, num_layers)
        self.loc = Attention(attention_layer, num_layers)

        self.embedding = PositionEmbeddingSine(hidden_dims // 2)

    def forward(self, z_f, x_f):
        (pos_zf, _) = self.embedding(z_f)
        (pos_xf, _) = self.embedding(z_f)
        cls = self.cls(z_f, x_f, pos_zf, pos_xf)
        loc = self.loc(z_f, x_f, pos_zf, pos_xf)
        return cls, loc

class MultiTrRPN(RPN):
    def __init__(self, 
                anchor_num, 
                in_channels, 
                weighted=False, 
                hidden_dims = 256, 
                num_heads=8,
                num_layers=6,
                dim_feed_forward=1024,
                dropout=0.1):
        super(MultiTrRPN, self).__init__()
        attention_layer = AttentionLayer(hidden_dims, num_heads, dropout, dim_feed_forward)

        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('tr2rpn'+str(i+2),
                    Tr2RPN(attention_layer, hidden_dims, num_layers, anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'tr2rpn'+str(idx))
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
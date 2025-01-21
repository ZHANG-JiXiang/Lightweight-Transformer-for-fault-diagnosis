import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, LongTensor
from typing import Tuple, Optional
from models.backbone.Signal.vit import Mlp, DropPath
import math


class PyramidConvPatchEmbedding(nn.Module):
    def __init__(self,
                 in_c,
                 kernel_sizes,
                 strides,
                 out_channels):
        super().__init__()
        self.in_c = in_c
        self.norm = nn.BatchNorm1d(in_c)
        self.layers = []
        for idx, (kernel_size, stride, out_channel) in enumerate(zip(kernel_sizes, strides, out_channels)):
            if idx == 0:
                layer = nn.Conv1d(in_channels=in_c, out_channels=out_channels[idx],
                                  kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
            else:
                layer = nn.Conv1d(in_channels=out_channels[idx - 1], out_channels=out_channel,
                                  kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))
            self.layers += [layer, nn.BatchNorm1d(out_channel), nn.ReLU(True)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.norm(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 0:
                f = x
        f = f.transpose(-1, -2)
        f = F.interpolate(f, size=(x.shape[1]))
        x = x + f.transpose(-1, -2)
        return x


class PatchEmerging(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size,
                 stride):
        super().__init__()
        self.path_embedding = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.path_embedding(x)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BRATBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=0.5,
                 drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(BRATBlock, self).__init__()
        self.norm1 = norm_layer(dim) if norm_layer else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) if norm_layer else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.BRA = Attention(dim, num_heads=16)

    def forward(self, x):
        x1 = x
        x = self.norm1(x)
        x = self.BRA(x)
        x = self.drop_path(x)
        x = x+x1
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MCSAT(nn.Module):
    # Multi-Scale Convolutional Sparse Attention Transformer
    def __init__(self,
                 in_c,
                 num_cls,
                 h_args,
                 dim,
                 kernel_sizes,
                 strides,
                 out_channels):
        super().__init__()

        self.PCPatchEmbedding = PyramidConvPatchEmbedding(in_c=in_c,
                                                      kernel_sizes=kernel_sizes,
                                                      strides=strides,
                                                      out_channels=out_channels)
        self.DWSConv = PatchEmerging(in_channels=out_channels[-1], out_channels=dim,
                                             patch_size=8, stride=8)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = dim
        self.block = BRATBlock(dim=dim)
        if not h_args:
            self.classifier.append(nn.Linear(self.last_channels, num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(self.last_channels, h_args[i]))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            self.classifier.append(nn.Softmax(dim=-1))

    def forward(self, x):
        b = x.shape[0]
        x = self.PCPatchEmbedding(x)
        x = self.DWSConv(x)
        x = x.transpose(-1, -2)
        x = self.block(x)
        x = self.avg_pool(x.transpose(-1, -2))
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x


def Transformer(in_c, h_args, num_cls):
    model = MCSAT(in_c=in_c, h_args=h_args, num_cls=num_cls, dim=128,
                    kernel_sizes=[16],
                    strides=[2],
                    out_channels=[64])
    return model

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.Signal.vit import Mlp, DropPath
from models.backbone.Signal.maxvit import window_partition, window_reverse
from models.backbone.Signal.nat import ConvDownsampler
from models.backbone.Signal.localvit import h_swish


class MultiScalePatching(nn.Module):
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


class GroupBatchnorm1d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm1d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num))
        self.bias = nn.Parameter(torch.zeros(c_num))
        self.eps = eps

    def forward(self, x):
        N, C, L = x.size()
        x = x.reshape(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.reshape(N, C, L)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5, torch_gn: bool = False):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm1d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1)
        reweights = self.sigmoid(gn_x * w_gamma)
        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, torch.div(x_1.size(1), 2), dim=1)
        x_21, x_22 = torch.split(x_2, torch.div(x_2.size(1), 2), dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv1d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv1d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv1d(up_channel // squeeze_radio, op_channel, kernel_size=3, padding=1, groups=group_size)
        self.PWC1 = nn.Conv1d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv1d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.avgpool(out), dim=1) * out
        out1, out2 = torch.split(out, torch.div(out.size(1), 2), dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 4, gate_treshold: float = 0.5, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2):
        super().__init__()
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
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


class conv_projection(nn.Module):
    def __init__(self, in_channels, conv_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.conv_ratio = conv_ratio
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=in_channels * conv_ratio,
                      kernel_size=1, stride=1, padding=0),
            h_swish())
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels * conv_ratio,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0),
            h_swish())
        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + self.norm(out)


class SparseAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 kv_ratio=1,
                 conv_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.heads_dim = dim // num_heads
        self.scale = qk_scale or self.heads_dim ** -0.5
        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, 2 * self.dim, bias=qkv_bias)
        self.resize = nn.Linear(self.dim, self.dim)
        self.mlp = conv_projection(in_channels=dim, conv_ratio=conv_ratio)
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        if kv_ratio > 1:
            self.reduce = nn.Sequential(nn.Conv1d(in_channels=2 * dim, out_channels=2 * dim,
                                                  kernel_size=kv_ratio, stride=kv_ratio),
                                        h_swish(),
                                        nn.BatchNorm1d(2 * dim))
        else:
            self.reduce = nn.Identity()

    def forward(self, x):
        b, n, c = x.shape
        # [b, n, c] ==> [b, num_heads, n, heads_dim]
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        # [b, num_heads, n, heads_dim]
        # [b, n, 2*c]
        kv = self.kv(x)
        # [b, 2 * c, n]
        kv = kv.transpose(-1, -2)
        # [b, 2 * c, m] ==> [b, m, 2*c] ==> [b, m, 2, num_heads, heads_dim] ==> [2, b, num_heads, m, heads_dim]
        kv = self.reduce(kv).transpose(-1, -2).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3,
                                                                                                              1, 4)
        # [b, num_heads, m, heads_dim]
        k, v = kv[0], kv[1]
        # [b, num_heads, n, m]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        # [b, num_heads, n, heads_dim] ==> [b, n, num_heads, heads_dim] ==> [b, n, dim]
        value = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        value = self.resize(value)
        x_mlp = (x + value)
        x_mlp = self.norm1(x_mlp)
        x_out = self.mlp(x_mlp.transpose(-1, -2)).transpose(-1, -2)
        out = self.norm2(x_mlp + x_out)
        return out.transpose(-1, -2)


class MCSwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_sacle=None,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attention = SparseAttention(dim=dim,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_sacle)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        # use layer scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, attn_mask):

        L = self.L
        B, N, C = x.shape

        shortcut = x
        x = self.norm1(x)

        pad_r = pad_l = 0
        _, Np, _ = x.shape

        # [B, Np, C] ==> [B, N, C]
        if pad_r > 0 or pad_l > 0:
            x = x[:, :N, :].contiguous()

        if not self.layer_scale:
            # ! 没有进行layer_scale
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = shortcut + self.drop_path(self.gamma1 * x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x


class MCSwinlayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 downsample=False,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_sacle=None,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        # build stages
        self.blocks = nn.Sequential(*[
            MCSwinTransformerBlock(dim=dim,
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_sacle=qk_sacle,
                                   drop=drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   act_layer=act_layer,
                                   norm_layer=norm_layer,
                                   layer_scale=layer_scale)
            for i in range(depth)])
        self.downsample = downsample
        self.downsample_layer = ConvDownsampler(dim=dim) if downsample else nn.Identity()


    def forward(self, x, N):
        x = self.downsample_layer(x)
        if self.downsample:
            N = (N + 1) // 2
        return x, N


class MSSA_T(nn.Module):
    def __init__(self,
                 in_c,
                 num_cls,
                 h_args,
                 kernel_sizes,
                 strides,
                 out_channels,
                 dim,
                 depth,
                 num_heads,
                 downscale=False,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=0.5):
        super().__init__()

        self.conv_embedding = MultiScalePatching(in_c=in_c,
                                                      kernel_sizes=kernel_sizes,
                                                      strides=strides,
                                                      out_channels=out_channels)

        self.patch_embedding = PatchEmerging(in_channels=out_channels[-1], out_channels=dim,
                                             patch_size=8, stride=8)

        self.SwinTransformerBlock = MCSwinlayer(dim=dim,
                                                depth=depth,
                                                num_heads=num_heads,
                                                downsample=downscale,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_sacle=qk_scale,
                                                drop=drop,
                                                act_layer=act_layer,
                                                norm_layer=norm_layer,
                                                layer_scale=layer_scale)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = dim
        self.scconv = ScConv(64)
        self.sacm = SparseAttention(dim=128)
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
        x = self.conv_embedding(x)
        x = self.patch_embedding(x).transpose(-1, -2)
        x = self.sacm(x)
        _, N, _ = x.shape
        x, N = self.SwinTransformerBlock(x, N)
        x = self.avg_pool(x.transpose(-1, -2))
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x


def SATFM(in_c, h_args, num_cls):
    model = MSSA_T(in_c=in_c, h_args=h_args, num_cls=num_cls,
                     kernel_sizes=[15, 9, 5, 3],
                     strides=[2, 1, 1, 1],
                     out_channels=[8, 16, 32, 64],
                     dim=128, depth=12, num_heads=8)
    return model


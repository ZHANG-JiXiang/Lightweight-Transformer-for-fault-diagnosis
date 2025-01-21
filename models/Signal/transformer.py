import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, LongTensor
from typing import Tuple, Optional
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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=8):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 逐点卷积层，使用1x1卷积核进行卷积，以改变通道数
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):  # 构造函数，初始化模块
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing

        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()

        self.routing_act = nn.Softmax(dim=-1)

    # 前向传播函数
    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):

        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)

        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),

                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)

                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):

    def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=True):
        super().__init__()

        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:
            mul_weight = 'soft'
        elif self.diff_routing:
            mul_weight = 'hard'
        else:
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':

            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':

            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):

        x = rearrange(x, "n c h w -> n h w c")

        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0

        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        q, kv = self.qkv(x)

        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])

        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        attn_weight = (
                              q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe

        out = self.wo(out)

        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        _, _, H, W = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)
        return x


class AttentionLePE(nn.Module):
    """
    vanilla attention
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

    def forward(self, x):
        _, _, H, W = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)
        return x


def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int]):
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead * head_dim,
                                                    region_h * region_size[0], region_w * region_size[1])
    return x


def find_closest_factors(n):
    sqrt_n = math.sqrt(n)
    sqrt_n = int(math.ceil(sqrt_n))  # 取上整
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            H = i
            W = torch.div(n, i, rounding_mode='trunc')
            return H, W


def regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: LongTensor, region_size: Tuple[int],
        kv_region_size: Optional[Tuple[int]] = None,
        auto_pad=True) -> Tensor:
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))

    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                         expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                         index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                           expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                           index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)

    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)

    output = attn @ value_g.flatten(-3, -2)

    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


class BiLevelRoutingAttention_nchw(nn.Module):

    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4, side_dwconv=3, auto_pad=False,
                 attn_backend='torch'):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5

        # 侧面深度卷积，用于局部上下文增强
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        self.topk = topk
        self.n_win = n_win

        # 线性层用于生成查询（q）、键（k）和值（v）
        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        # 选择后端实现注意力机制
        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):

        N, C, H, W = x.size()
        region_size = (H // self.n_win, W // self.n_win)

        # 第一步：线性投影到q、k、v空间
        qkv = self.qkv_linear.forward(x)  # ncHW
        q, k, v = qkv.chunk(3, dim=1)  # ncHW

        # 第二步：区域到区域路由
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
        q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
        k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # 第三步：非参数化的token-to-token注意力
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size
                                        )

        output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW

        if ret_attn_mask:
            return output, attn_mat

        return output


class MSSA_T(nn.Module):
    # Multi-scale Sparse Attention Transformer
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
        self.DWSConv = DepthwiseSeparableConv(in_channels=out_channels[-1], out_channels=dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = dim
        self.BiFormer = BiLevelRoutingAttention_nchw(64).cuda()
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
        print(x.shape)
        B, C, N = x.size()
        H, W = find_closest_factors(N)
        x_r = x.view(-1, C, H, W)
        output = self.BiFormer(x_r)
        B, C, H, W = output.size()
        x = x.view(B, C, N)
        x = self.avg_pool(x.transpose(-1, -2))
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    1D data Patch Embedding
    """

    def __init__(self, data_size, in_c, patch_size, norm_layer):
        super().__init__()
        self.data_size = data_size
        self.patch_size = patch_size
        self.grid_size = data_size // patch_size
        self.num_patches = self.grid_size
        self.embed_dim = in_c * patch_size
        self.projection = nn.Conv1d(in_c, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, N = x.shape
        assert N == self.data_size, "the Input data size {} does not math the model {}".format(N, self.data_size)

        # [B, C, N] ==> [B, embed_dim, grid_size]
        x = self.projection(x)
        # [B,embed_dim, grid_size] ==> [B, grid_size, embed_dim]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


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


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim) if norm_layer else nn.Identity()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) if norm_layer else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.BiFormer = BiLevelRoutingAttention_nchw(64).cuda()
    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x1 = x
        x = self.norm1(x)
        print(x.shape)
        x = self.attn(x)
        B, C, N = x.size()
        H, W = find_closest_factors(N)
        x_r = x.view(-1, C, H, W)
        output = self.BiFormer(x_r)
        B, C, H, W = output.size()
        x = x.view(B, C, N)
        x = self.drop_path(x)
        x = x + x1
        x2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x2
        return x


class VisionTransformer(nn.Module):
    cfgs = {'base': [12, 8, 16],
            "middle_1": [12, 16, 16],
            'middle_2': [12, 16, 32],
            'large_1': [24, 16, 16],
            'large_2': [24, 16, 32]}

    def __init__(self, data_size, in_c, num_cls,
                 cfgs, h_args, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 norm_layer=None, act_layer=None, use_init=False):
        super(VisionTransformer, self).__init__()
        self.num_cls = num_cls
        depth, num_heads, patch_size = cfgs[0], cfgs[1], cfgs[2]
        self.last_channels = self.embed_dim = patch_size * in_c
        self.num_tokens = 1
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act = act_layer or nn.GELU
        self.patch_embed = PatchEmbed(data_size=data_size,
                                      patch_size=patch_size,
                                      in_c=in_c,
                                      norm_layer=norm_layer)
        self.num_patches = data_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()
        self.classifier = nn.ModuleList([])
        if not h_args:
            self.classifier.append(nn.Linear(self.last_channels, self.num_cls))
            self.classifier.append((nn.Softmax(dim=-1)))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(self.last_channels, h_args[i]))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        if use_init:
            self._initialize_weights()

    def forward_features(self, x):
        # taking example: data: [B, 3, 1024], patch_size:16
        # [B, 3, 1024] ==> [B, 64, 48]
        x = self.patch_embed(x)
        # cls_token: [1, 1, 48] ==> [B, 1, 48]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # concat [B, 64, 48] abd [B, 1, 48] ==> [B, 65, 48]
        x = torch.cat((cls_token, x), dim=1)
        # x + position encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, _ = self.forward_features(x)
        for module in self.classifier:
            x = module(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# depth, num heads, patch size
cfgs = {'base': [4, 8, 16],
        "middle_1": [12, 16, 16],
        'middle_2': [12, 16, 32],
        'large_1': [24, 16, 16],
        'large_2': [24, 16, 32]}


def vit_base(data_size, in_c, num_cls, h_args, cfgs=cfgs['base']):
    model = VisionTransformer(data_size=data_size,
                              in_c=in_c,
                              num_cls=num_cls,
                              h_args=h_args,
                              cfgs=cfgs)
    return model


def vit_middle_patch16(data_size, in_c, num_cls, h_args, cfgs=cfgs["middle_1"]):
    model = VisionTransformer(data_size=data_size,
                              in_c=in_c,
                              num_cls=num_cls,
                              h_args=h_args,
                              cfgs=cfgs)
    return model


def vit_middle_patch32(data_size, in_c, num_cls, h_args, cfgs=cfgs["middle_2"]):
    model = VisionTransformer(data_size=data_size,
                              in_c=in_c,
                              num_cls=num_cls,
                              h_args=h_args,
                              cfgs=cfgs)
    return model


if __name__ == "__main__":
    model = vit_middle_patch16(data_size=1024, in_c=2, num_cls=8, h_args=[256, 128, 64, 32])
    tensor = torch.rand(1, 2, 1024)
    print(model(tensor).shape)

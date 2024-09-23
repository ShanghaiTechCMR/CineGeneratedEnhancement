"""
Conditional GAN, Spatial Temporal UNet

This code is adapted from the following repository:
  - [IGITUGraz/WeatherDiffusion: Code for "Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models" [TPAMI 2023]](https://github.com/IGITUGraz/WeatherDiffusion)
  - [sagan-pytorch/model.py at master · rosinality/sagan-pytorch](https://github.com/rosinality/sagan-pytorch/blob/master/model.py)
  
Special thanks to the authors for their contributions.
"""
import itertools
import math
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import rearrange

from models.fused_clstm import BiHOTTCLSTMCell

DEFAULT_ACT = partial(nn.LeakyReLU, 0.1)
SpectralNorm = torch.nn.utils.spectral_norm


def MyGroupNorm(in_channels, max_groups=32) -> nn.Module:
    num_groups = min(in_channels, max_groups)
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-4, affine=True)


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_class, NormLayer=nn.BatchNorm2d):
        super().__init__()

        self.norm_layer = NormLayer(in_channel, affine=False)
        self.embed = nn.Embedding(n_class, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.norm_layer(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out


def _switch_norm_layer(name: str):
    if name == 'group_norm':
        return MyGroupNorm
    elif name == 'instance_norm_2d':
        return partial(nn.InstanceNorm2d, eps=1e-05)
    elif name == 'instance_norm_3d':
        return partial(nn.InstanceNorm3d, eps=1e-05)
    elif name == 'batch_norm_2d':
        return partial(nn.BatchNorm2d, eps=1e-05)
    elif name == 'batch_norm_3d':
        return partial(nn.BatchNorm3d, eps=1e-05)
    elif name == 'cc_batch_norm_2d':
        return partial(ConditionalNorm, NormLayer=partial(nn.BatchNorm2d, eps=1e-05))
    elif name == 'cc_batch_norm_3d':
        return partial(ConditionalNorm, NormLayer=partial(nn.BatchNorm3d, eps=1e-05))
    elif name == 'cc_instance_norm_2d':
        return partial(ConditionalNorm, NormLayer=partial(nn.InstanceNorm2d, eps=1e-05))
    elif name == 'cc_instance_norm_3d':
        return partial(ConditionalNorm, NormLayer=partial(nn.InstanceNorm3d, eps=1e-05))
    else:
        raise NotImplemented


class DeepResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *,
                 num_classes=None,
                 bottleneck_factor=4,
                 p_dropout=-1.0,
                 NormLayer=ConditionalNorm,
                 is_spectral_norm=True,
                 ActLayer=DEFAULT_ACT):
        super().__init__()
        self.num_classes = num_classes

        def ActDropConv(c_in, c_out, kernel_size, *, is_last_layer):
            """pre-activation
            Norm -> Act -> Dropout -> Conv
            """
            layers = []
            layers.append(ActLayer())
            if is_last_layer and p_dropout >= 0:
                layers.append(nn.Dropout(p_dropout))
            conv = nn.Conv2d(c_in, c_out, kernel_size, 1, kernel_size // 2)
            if is_spectral_norm:
                conv = SpectralNorm(conv)
            layers.append(conv)
            return nn.Sequential(*layers)

        mid_channels = out_channels // bottleneck_factor
        norm_init_args = tuple() if num_classes is None else (num_classes,)
        self.norms = nn.ModuleList([
            NormLayer(in_channels, *norm_init_args),
            NormLayer(mid_channels, *norm_init_args),
            NormLayer(mid_channels, *norm_init_args),
            NormLayer(mid_channels, *norm_init_args),
        ])
        self.convs = nn.ModuleList([
            ActDropConv(in_channels, mid_channels, 1, is_last_layer=False),
            ActDropConv(mid_channels, mid_channels, 3, is_last_layer=False),
            ActDropConv(mid_channels, mid_channels, 3, is_last_layer=False),
            ActDropConv(mid_channels, out_channels, 1, is_last_layer=True)
        ])

        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, class_id=None):
        """
        :param x: [B, C, H, W]
        :param class_id: [B,]Integer
        """
        skip = self.proj(x)
        h = x
        for norm_layer, conv_layer in zip(self.norms, self.convs):
            if self.num_classes is not None:
                h = norm_layer(h, class_id)
            else:
                h = norm_layer(h)
            h = conv_layer(h)
        return h + skip


class DeepResidualBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *,
                 bottleneck_factor=4,
                 p_dropout=-1.0,
                 NormLayer=MyGroupNorm,
                 is_spectral_norm=True,
                 ActLayer=DEFAULT_ACT):
        super().__init__()

        def NormActDropConv(c_in, c_out, kernel_size, *, is_last_layer):
            """pre-activation
            Norm -> Act -> Dropout -> Conv
            """
            layers = []
            if NormLayer is not None:
                layers.append(NormLayer(c_in))
            layers.append(ActLayer())
            if is_last_layer and p_dropout >= 0:
                layers.append(nn.Dropout(p_dropout))
            conv = nn.Conv3d(c_in, c_out, kernel_size, 1, kernel_size // 2)
            if is_spectral_norm:
                conv = SpectralNorm(conv)
            layers.append(conv)
            return nn.Sequential(*layers)

        mid_channels = out_channels // bottleneck_factor
        self.convs = nn.Sequential(
            NormActDropConv(in_channels, mid_channels, 1, is_last_layer=False),
            NormActDropConv(mid_channels, mid_channels, 3, is_last_layer=False),
            NormActDropConv(mid_channels, mid_channels, 3, is_last_layer=False),
            NormActDropConv(mid_channels, out_channels, 1, is_last_layer=True),
        )

        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: [B, C_in, T, H, W]
        """
        skip = self.proj(x)
        h = x
        h = self.convs(h)
        return h + skip


class CrossAttn2d(nn.Module):
    def __init__(self, q_channels, kv_channels, qk_reduction=8):
        super().__init__()
        self.in_channels = q_channels

        self.conv_q = nn.Conv2d(q_channels, q_channels // qk_reduction, kernel_size=1, stride=1, padding=0)
        self.conv_k = nn.Conv2d(kv_channels, q_channels // qk_reduction, kernel_size=1, stride=1, padding=0)
        self.conv_v = nn.Conv2d(kv_channels, q_channels, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(q_channels, q_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x4q, x4kv):
        """
        :param x4q: [B, C=q_channels, H, W]
        :param x4kv: [B, C=kv_channels, H, W]
        """
        q = self.conv_q(x4q)
        k = self.conv_k(x4kv)
        v = self.conv_v(x4kv)

        # compute attention
        B, dim, H, W = q.shape
        q = q.reshape(B, dim, H * W)  # B, C_qk, N=(HW)
        k = k.reshape(B, dim, H * W)  # B, C_qk, N=(HW)
        score = torch.bmm(q.permute(0, 2, 1), k)  # B, N(q), N(k)
        score_ = score * (int(dim) ** (-0.5))
        attention = torch.softmax(score_, dim=2)  # B, N(q), N(k)

        # attend to values
        v = v.reshape(B, -1, H * W)  # B, C, N(v)
        h = torch.bmm(v, attention.permute(0, 2, 1))  # B, C, N(q)
        h = h.reshape(B, -1, H, W)

        # out projection
        h = self.conv_out(h)

        # trick for scaling
        h = self.gamma * h

        # residual
        out = h + x4q

        return out


class ChannelFusedCrossAttn(nn.Module):
    def __init__(self, in_channels: int, context_channels: int, num_context: int, *, ActLayer=DEFAULT_ACT):
        super().__init__()
        self.fusion_mapping = nn.Sequential(
            nn.Conv2d(context_channels * num_context, context_channels, kernel_size=1, stride=1, padding=0),
            ActLayer()
        )
        self.cross_attn = CrossAttn2d(in_channels, context_channels)

    def forward(self, x, context):
        """
        :param x: [B, C, H, W]
        :param context: [B, D=num_context, C, H, W]
        """
        context = self.fusion_mapping(rearrange(context, 'b d c h w -> b (d c) h w'))  # [B, C, H, W]
        x = self.cross_attn(x, context)
        return x


class UpsampleWorkaround(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.up = nn.Upsample(*args, **kwargs)

    def forward(self, x):
        if x.dtype == torch.bfloat16:
            # torch does not support bfloat16 upsample for now (possible is a pytorch's bug, github issues can be found)
            y = self.up(x.to(torch.float32)).type_as(x)
        else:
            y = self.up(x)
        return y


class MultisliceContextSpatioTemporalMixedSpatialUNet(nn.Module):
    """
    Spatio Temporal Encoder + Self Attention Decoder with Multi-slice context Fusion by Cross Attention
        Spatio Temporal Encoder
            ConvRNN embedding
            Residual Conv3d Blocks
            ConvRNN 3D to 2D

        Self Attention Decoder:
            Vanilla UNet Decoder
            + Multiple Block Per Level
            + Multiple residual connections per Level
            + Cross Attention Layer used to multi-slices fusion
    """

    def __init__(self, num_classes: int, in_channels_st: int, in_channels_spatial: int, out_channels: int,
                 resolution: int,
                 base_channels_encoder3d: int, base_channels_encoder2d: int, base_channels_decoder: int,
                 depth_window_size: int = 3,
                 level_ch_mult: Sequence[int] = (1, 1, 2, 2), temporal_reduction_ratio: Sequence[int] = (1, 1, 2),
                 attn_fusion_resolutions: Sequence[int] = (16,), high_order_lstm_rank_ratio: int = 4,
                 n_blk_per_level: int = 2, p_dropout_encoder: float = -1.0, ActLayer=DEFAULT_ACT, is_spectral_norm=True,
                 is_rnn_spectral_norm=False,
                 encoder2d_norm='batch_norm_2d',
                 encoder3d_norm='batch_norm_3d',
                 decoder2d_norm='cc_batch_norm_2d',
                 multislice_fusion_type='channel_projection'
                 ):
        super().__init__()
        assert len(level_ch_mult) == len(temporal_reduction_ratio) + 1, \
            f"N level only have (N-1) down/up sampling"

        Enc2dNormLayer = _switch_norm_layer(encoder2d_norm)
        Enc3dNormLayer = _switch_norm_layer(encoder3d_norm)
        Dec2dNormLayer = _switch_norm_layer(decoder2d_norm)

        level_ch_mult = list(level_ch_mult)
        attn_fusion_resolutions = list(attn_fusion_resolutions)
        stage_in_ch_mult = [1, ] + level_ch_mult  # the first stage_ch_mult may not be 1*base_channels

        self.num_levels = len(level_ch_mult)
        self.n_blk_per_level = n_blk_per_level
        self.resolution = resolution
        self.depth_window_size = depth_window_size
        self.max_context_level = max(attn_fusion_resolutions)

        def _DeepResBlk2d(in_channels, out_channels, NormLayer, *, num_classes=None, p_dropout=-1.0):
            return DeepResidualBlock2d(
                num_classes=num_classes, in_channels=in_channels, out_channels=out_channels,
                p_dropout=p_dropout, NormLayer=NormLayer, is_spectral_norm=is_spectral_norm,
                ActLayer=ActLayer
            )

        def _DeepResBlk3d(in_channels, out_channels, NormLayer, *, p_dropout=-1.0):
            return DeepResidualBlock3d(
                in_channels=in_channels, out_channels=out_channels,
                p_dropout=p_dropout, NormLayer=NormLayer, is_spectral_norm=is_spectral_norm,
                ActLayer=ActLayer
            )

        def _ConvDownsample2d(channels: int):
            conv = nn.Conv2d(channels, channels, 4, 2, 1)
            if is_spectral_norm:
                conv = SpectralNorm(conv)
            return conv

        def _ConvDownsample3d(channels: int, temporal_ratio: int):
            if temporal_ratio == 1:
                conv = nn.Conv3d(channels, channels, (3, 4, 4), (1, 2, 2), (1, 1, 1))
            elif temporal_ratio == 2:
                conv = nn.Conv3d(channels, channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))
            else:
                raise NotImplemented
            if is_spectral_norm:
                conv = SpectralNorm(conv)
            return conv

        def _ConvRNN(in_channels, hidden_channels):
            return BiHOTTCLSTMCell(in_channels, hidden_channels,
                                   order=3, steps=5, ranks=hidden_channels // high_order_lstm_rank_ratio,
                                   kernel_size=3,
                                   is_spectral_norm=is_rnn_spectral_norm,
                                   NormLayer=None,
                                   conv_block_type='conv')

        self.stem3d = nn.Conv3d(in_channels_st, base_channels_encoder3d, 3, 1, 1)
        # self.stem3d = SpectralNorm(self.stem3d) if is_spectral_norm else self.stem3d

        self.stem2d = nn.Conv2d(in_channels_spatial, base_channels_encoder3d, 3, 1, 1)
        # self.stem2d = SpectralNorm(self.stem2d) if is_spectral_norm else self.stem2d

        # Spatial-Temporal downsampling (3D -> 2D)
        curr_res = resolution
        self.downs3d = nn.ModuleList()
        for i_level in range(self.num_levels):
            block_in = base_channels_encoder3d * stage_in_ch_mult[i_level]
            level_out = base_channels_encoder3d * level_ch_mult[i_level]

            down = nn.Module()
            down.blocks = nn.ModuleList()
            down.is_downsample = True if i_level != self.num_levels - 1 else False
            for i_block in range(self.n_blk_per_level):
                down.blocks.append(_DeepResBlk3d(block_in, level_out, Enc3dNormLayer, p_dropout=p_dropout_encoder))
                block_in = level_out
            down.clstm = _ConvRNN(level_out, level_out)
            if down.is_downsample:
                # bottleneck process the feature map with the resolution of last level
                down.downsample = _ConvDownsample3d(level_out, temporal_reduction_ratio[i_level])
                curr_res = curr_res // 2
            self.downs3d.append(down)

        # bottleneck (for Spatial-Temporal branch)
        bottleneck_channels = level_out
        self.bottleneck = nn.Module()
        self.bottleneck.block1 = _DeepResBlk2d(level_out * 3, bottleneck_channels, Enc2dNormLayer,
                                               p_dropout=p_dropout_encoder)
        self.bottleneck.block2 = _DeepResBlk2d(bottleneck_channels, bottleneck_channels, Enc2dNormLayer,
                                               p_dropout=p_dropout_encoder)

        # Spatial downsampling (2D -> 2D)
        curr_res = resolution
        self.downs2d = nn.ModuleList()
        for i_level in range(self.num_levels):
            block_in = base_channels_encoder2d * stage_in_ch_mult[i_level]
            level_out = base_channels_encoder2d * level_ch_mult[i_level]

            down = nn.Module()
            down.blocks = nn.ModuleList()
            down.attns = nn.ModuleList()
            down.is_downsample = True if i_level != self.num_levels - 1 else False
            for i_block in range(self.n_blk_per_level):
                down.blocks.append(_DeepResBlk2d(block_in, level_out, Enc2dNormLayer))
                block_in = level_out
            if down.is_downsample:
                # bottleneck process the feature map with the resolution of last level
                down.downsample = _ConvDownsample2d(level_out)
                curr_res = curr_res // 2
            self.downs2d.append(down)

        # upsampling
        self.ups = nn.ModuleList()
        block_in = bottleneck_channels
        for i_level in reversed(range(self.num_levels)):
            level_out = base_channels_decoder * level_ch_mult[i_level]

            up = nn.Module()
            up.blocks = nn.ModuleList()
            up.attns = nn.ModuleList()
            up.is_upsample = True if i_level != 0 else False
            up.is_attn = (curr_res in attn_fusion_resolutions)
            for i_block in range(self.n_blk_per_level):
                skip3d_in = (base_channels_encoder3d * level_ch_mult[i_level]) * 3
                skip2d_in = (base_channels_encoder2d * level_ch_mult[i_level])
                up.blocks.append(_DeepResBlk2d(block_in + ((skip3d_in + skip2d_in) if i_block == 0 else 0),
                                               level_out, Dec2dNormLayer, num_classes=num_classes))
                block_in = level_out
                if up.is_attn:
                    up.attns.append(ChannelFusedCrossAttn(level_out, skip3d_in, self.depth_window_size))
            if up.is_upsample:
                # the first level won't upsample, it is already the input resolution
                # up.upsample = UpsampleWorkaround(scale_factor=2, mode='nearest')
                up.upsample = UpsampleWorkaround(scale_factor=2, mode='bilinear')
                curr_res = curr_res * 2
            self.ups.insert(0, up)  # prepend to get consistent order

        # end
        out_conv = nn.Conv2d(level_out, out_channels, kernel_size=3, stride=1, padding=1)
        out_conv = SpectralNorm(out_conv) if is_spectral_norm else out_conv
        self.out_conv = out_conv

    def forward(self, x4d, x2d=None, *, class_id=None):
        """
        :param x3d: [B, C_in_st, T, D, H, W]
        :param x2d: [B, C_in_s, H, W]
        :param class_id: [B,]Integer
        :return [B, C_out, H, W]
        """
        B, C, T, D, H, W = x4d.shape
        assert H == W == self.resolution
        assert x2d is not None and class_id is not None
        middle_slice_indices = torch.tensor([(D // 2 + i * D) for i in range(B)], dtype=torch.int64, device=x4d.device)

        """Multislice Spatial-Temporal Context Extractor Branch"""
        # Spatial-Temporal downsampling
        hs_enc3d = []  # List[[B, C, T, H, W]]
        hs_enc3d_context = []  # List[[B*D, C, T, H, W]]

        x3d = rearrange(x4d, 'b c t d h w -> (b d) c t h w')
        h = self.stem3d(x3d)
        for i_level in range(self.num_levels):
            down3d = self.downs3d[i_level]
            for i_block in range(self.n_blk_per_level):
                h = down3d.blocks[i_block](h)

            # LSTM forward
            outputs_f, outputs_b, hidden_list_f, hidden_list_b, memory_f, memory_b = down3d.clstm(h)
            h = outputs_f + outputs_b  # B, C, T, H, W
            hidden_pooling = torch.stack(hidden_list_f + hidden_list_b, dim=0).mean(dim=0)
            spatio_skip = torch.cat([memory_f, hidden_pooling, memory_b], dim=1)  # B, C, H, W
            hs_enc3d.append(spatio_skip[middle_slice_indices, ...])
            hs_enc3d_context.append(spatio_skip)

            if down3d.is_downsample:
                h = down3d.downsample(h)

        # middle
        h = self.bottleneck.block1(hs_enc3d[-1])
        h = self.bottleneck.block2(h)
        h_bottleneck3d = h

        """Style Translator Branch"""
        # Spatial downsampling
        hs_enc2d = []
        h = self.stem2d(x2d)
        for i_level in range(self.num_levels):
            down2d = self.downs2d[i_level]
            for i_block in range(self.n_blk_per_level):
                h = down2d.blocks[i_block](h)
            hs_enc2d.append(h)

            if down2d.is_downsample:
                h = down2d.downsample(h)

        # upsampling
        h = h_bottleneck3d
        for i_level in reversed(range(self.num_levels)):
            up = self.ups[i_level]
            if up.is_attn:
                context = hs_enc3d_context[i_level]
                context = rearrange(context, '(b d) c h w -> b d c h w', b=B, d=D)
            for i_block in range(self.n_blk_per_level):
                if i_block == 0:  # the first block will accept skip from encoder
                    h = torch.cat([h, hs_enc3d.pop(), hs_enc2d.pop()], dim=1)
                h = up.blocks[i_block](h, class_id)
                if up.is_attn:
                    h = up.attns[i_block](h, context)
            if up.is_upsample:
                h = up.upsample(h)

        # end
        h = self.out_conv(h)
        return h, h_bottleneck3d

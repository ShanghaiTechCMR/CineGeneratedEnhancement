import enum
import itertools
from functools import partial
from typing import Sequence, List
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as nnf
from einops import rearrange
from einops.layers.torch import Rearrange
from models.cgan_stenc_sdec_windowed import MultisliceContextSpatioTemporalMixedSpatialUNet

class Generator(nn.Module):
    def __init__(self, *,
                 num_classes=5,
                 latent_dim=256,
                 image_size=128,
                 base_channels_encoder3d=32,
                 base_channels_encoder2d=32,
                 base_channels_decoder=64,
                 depth_window_size=3,
                 level_ch_mult=(1, 2, 4, 4, 4),
                 temporal_reduction_ratio=(1, 1, 1, 1),
                 attn_fusion_resolutions=(64, 32, 16),
                 encoder2d_norm='instance_norm_2d',
                 encoder3d_norm='batch_norm_3d',
                 decoder2d_norm='cc_instance_norm_2d',
                 p_dropout_encoder=0.0
                 ):
        super().__init__()
        self.unet = MultisliceContextSpatioTemporalMixedSpatialUNet(
            num_classes=num_classes,
            in_channels_st=1,
            in_channels_spatial=1,
            out_channels=1,
            resolution=image_size,
            base_channels_encoder3d=base_channels_encoder3d,
            base_channels_encoder2d=base_channels_encoder2d,
            base_channels_decoder=base_channels_decoder,
            depth_window_size=depth_window_size,
            level_ch_mult=level_ch_mult,
            temporal_reduction_ratio=temporal_reduction_ratio,
            attn_fusion_resolutions=attn_fusion_resolutions,
            high_order_lstm_rank_ratio=4,
            n_blk_per_level=1,
            p_dropout_encoder=p_dropout_encoder,
            ActLayer=partial(nn.LeakyReLU, 0.1),
            is_spectral_norm=False,
            is_rnn_spectral_norm=False,
            encoder2d_norm=encoder2d_norm,
            encoder3d_norm=encoder3d_norm,
            decoder2d_norm=decoder2d_norm
        )

        hidden_dim = level_ch_mult[-1] * base_channels_encoder3d
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)', h=1, w=1),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def z_score(self, x, eps=1e-5):
        B = x.shape[0]
        xhat = x.reshape(B, -1)
        mu = xhat.mean(dim=1, keepdim=True)
        sigma = xhat.std(dim=1, keepdim=True)
        xhat = (xhat - mu) / (sigma + eps)
        return xhat.reshape(x.shape)

    def forward(self, closest_cine_image, cine4d, *, class_id):
        """
        :param closest_cine_image: [B, 1, H, W]
        :param cine4d: [B, 1, T_cine, D, H, W], D//2 is the slice of closest_cine_images located at
        :param class_id: [B,]
        """
        B, _, T, D, H, W = cine4d.shape
        # variable preparation, normalize sample wise
        x4d = self.z_score(cine4d)
        x2d = self.z_score(closest_cine_image)

        y, h_bottleneck = self.unet(x4d, x2d, class_id=class_id)
        latent_mapped = self.head(h_bottleneck)

        # using float32 to compute tanh, avoid low precision stability issues. And output float32 results
        y = torch.sigmoid(y.to(torch.float32))  # [B, 1, H, W], scale to [0, 1]
        return y, latent_mapped
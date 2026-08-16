"""Conditional projection discriminator used for CGE training."""

from typing import Sequence

import torch
import torch.nn as nn


SpectralNorm = torch.nn.utils.spectral_norm


def MyGroupNorm(in_channels, num_groups=32) -> nn.Module:
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class SelfAttn2d(nn.Module):
    def __init__(self, in_channels, NormLayer=MyGroupNorm):
        super().__init__()
        self.in_channels = in_channels

        self.norm = NormLayer(in_channels) if NormLayer is not None else nn.Identity()
        self.conv_q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = x
        h = self.norm(h)
        q = self.conv_q(h)
        k = self.conv_k(h)
        v = self.conv_v(h)

        # compute attention
        B, dim, H, W = q.shape
        q = q.reshape(B, dim, H * W)  # B, C, N=(HW)
        k = k.reshape(B, dim, H * W)  # B, C, N=(HW)
        score = torch.bmm(q.permute(0, 2, 1), k)  # B, N(q), N(k)
        score_ = score * (int(dim) ** (-0.5))
        attention = torch.softmax(score_, dim=2)  # B, N(q), N(k)

        # attend to values
        v = v.reshape(B, dim, H * W)  # B, C, N(v)
        h = torch.bmm(v, attention.permute(0, 2, 1))  # B, C, N(q)
        h = h.reshape(B, dim, H, W)

        # out projection
        h = self.conv_out(h)

        # trick for scaling
        out = self.gamma * h + x

        return out


class CGANSADiscriminator(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, image_size: int = 128, base_channels=64,
                 level_ch_multi: Sequence[int] = (1, 2, 4, 8),
                 attn_at_resolutions: Sequence[int] = (64,)
                 ):
        super().__init__()

        def _LevelBlock(in_channels: int, out_channels: int, resolution: int):
            layers = []
            layers.append(nn.Sequential(
                SpectralNorm(nn.Conv2d(in_channels, out_channels, 4, 2, 1)),
                nn.LeakyReLU(0.1)
            ))
            if resolution in attn_at_resolutions:
                layers.append(SelfAttn2d(out_channels, NormLayer=None))
            return nn.Sequential(*layers)

        blocks = []
        for i in range(len(level_ch_multi)):
            level_in = base_channels * level_ch_multi[i - 1] if i > 0 else in_channels
            level_out = base_channels * level_ch_multi[i]
            resolution = image_size // 2 ** i
            blocks.append(_LevelBlock(level_in, level_out, resolution))

        self.convs = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = SpectralNorm(nn.Linear(level_out, 1))

        embed = nn.Embedding(num_classes, level_out)
        embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = SpectralNorm(embed)

    def forward(self, x, *, class_id):
        """
        :param x: [B, C, H, W]
        :param class_id: [B,]Integer
        :returns [B, 1]
        """
        B, _, _, _ = x.shape
        y = self.pool(self.convs(x))  # [B, C, 1, 1]
        y = y.reshape(B, -1)  # [B, C]

        out_linear = self.linear(y)  # [B, 1]
        out_class = (y * self.embed(class_id)).sum(1, keepdims=True)  # [B, 1]
        out = out_linear + out_class
        return out


# Public aliases; the implementation above keeps the original class names and
# body so it can be compared directly with CGE_128.
Discriminator = CGANSADiscriminator
SelfAttention2d = SelfAttn2d


__all__ = [
    "CGANSADiscriminator",
    "Discriminator",
    "MyGroupNorm",
    "SelfAttn2d",
    "SelfAttention2d",
]

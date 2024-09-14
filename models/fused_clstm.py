from typing import Sequence
import enum
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

DEFAULT_ACT = nn.SiLU
SpectralNorm = torch.nn.utils.spectral_norm


def MyGroupNorm(in_channels, num_groups=32) -> nn.Module:
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Conv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 is_spectral_norm=False):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        if is_spectral_norm:
            conv = SpectralNorm(conv)
        self.conv = conv

    def forward(self, x):
        """
        :param x: [B, C, H, W]
        """
        out = self.conv(x)
        return out


class ConvNorm(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 is_spectral_norm=False,
                 NormLayer=MyGroupNorm):
        super().__init__()
        layers = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        if is_spectral_norm:
            conv = SpectralNorm(conv)
        layers.append(conv)
        if NormLayer is not None:
            layers.append(NormLayer(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: [B, C, H, W]
        """
        out = self.block(x)
        return out


class ConvBlockType(enum.Enum):
    conv = 'conv'
    conv_norm = 'conv_norm'


@torch.jit.script
def fuse_mul_add_mul(F, C, I, C_tilde):
    return F * C + I * C_tilde


def lstm_internal_op(I, F, O, C_tilde, C):
    I = torch.sigmoid(I)
    F = torch.sigmoid(F)
    O = torch.sigmoid(O)
    C_tilde = torch.tanh(C_tilde)

    C = fuse_mul_add_mul(F, C, I, C_tilde)
    H = O * torch.tanh(C)

    return H, C


class CLSTMCell(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 is_spectral_norm=False,
                 NormLayer=MyGroupNorm,
                 conv_block_type: str = 'conv'):
        super().__init__()
        conv_block_type = ConvBlockType(conv_block_type)

        def _Conv(in_channels, out_channels):
            if conv_block_type == ConvBlockType.conv:
                return Conv(in_channels, out_channels, 3)
            elif conv_block_type == ConvBlockType.conv_norm:
                return ConvNorm(in_channels, out_channels, 3, is_spectral_norm=is_spectral_norm, NormLayer=NormLayer)
            else:
                raise NotImplemented

        self.conv_x2ifoc = _Conv(in_channels, hidden_channels * 4)
        self.conv_h2ifoc = _Conv(hidden_channels, hidden_channels * 4)
        self.hidden_channels = hidden_channels

    def forward(self, inputs, h_0=None, c_0=None):
        """
        :param inputs: [B, C * n_parallel, T, H, W]
        :param h_0: [B, Ch * n_parallel, H, W]
        :param c_0: [B, Ch * n_parallel, H, W]
        """
        nb_B, nb_C, nb_T, nb_H, nb_W = inputs.shape
        if h_0 is None:
            h_0 = torch.zeros((nb_B, self.hidden_channels, nb_H, nb_W),
                              device=inputs.device).type_as(inputs)
        if c_0 is None:
            c_0 = torch.zeros((nb_B, self.hidden_channels, nb_H, nb_W),
                              device=inputs.device).type_as(inputs)

        outputs = []
        H = h_0
        C = c_0
        for i in range(nb_T):
            X = inputs[:, :, i, :, :]
            xI, xF, xO, xC_tilde = torch.split(self.conv_x2ifoc(X), self.hidden_channels, dim=1)
            hI, hF, hO, hC_tilde = torch.split(self.conv_h2ifoc(H), self.hidden_channels, dim=1)
            I, F, O, C_tilde = (xI + hI), (xF + hF), (xO + hO), (xC_tilde + hC_tilde)
            # H, C = lstm_internal_op(I, F, O, C_tilde, C)
            H, C = checkpoint(lstm_internal_op, I, F, O, C_tilde, C, use_reentrant=False)
            outputs.append(H)
        outputs = torch.stack(outputs, dim=2)  # T*[B, Ch * n_parallel, H, W] -> [B, Ch * n_parallel, T, H, W]
        return outputs, H, C


class HOTTCLSTMCell(nn.Module):
    """Higher Order Tensor-Train Convolutional LSTM"""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 order: int = 3,
                 steps: int = 5,
                 ranks: int = 8,
                 kernel_size: int = 3,
                 is_spectral_norm=False,
                 NormLayer=nn.BatchNorm2d,
                 conv_block_type: str = 'conv'):
        super().__init__()
        conv_block_type = ConvBlockType(conv_block_type)

        def _Conv(in_channels, out_channels, is_feature_norm=False):
            if conv_block_type == ConvBlockType.conv:
                return Conv(in_channels, out_channels, kernel_size)
            elif conv_block_type == ConvBlockType.conv_norm:
                return ConvNorm(in_channels, out_channels, kernel_size,
                                is_spectral_norm=is_spectral_norm,
                                NormLayer=NormLayer if is_feature_norm else None)
            else:
                raise NotImplemented

        window_size = steps - order + 1
        assert window_size > 0
        assert order >= 1

        # parameterize 2nd, 3rd, 4th, ... orders, 1st order parametrization contains in conv_xh2ifoc
        self.tt_conv = nn.ModuleList([
            _Conv(ranks, ranks, is_feature_norm=False) for _ in range(order - 1)
        ])
        self.pre_conv = nn.ModuleList([
            _Conv(window_size * hidden_channels, ranks, is_feature_norm=False) for _ in range(order)
        ])
        self.conv_xh2ifoc = _Conv(in_channels + ranks, hidden_channels * 4, is_feature_norm=True)

        self.order = order
        self.steps = steps
        self.ranks = ranks
        self.hidden_channels = hidden_channels
        self.window_size = window_size

    def _step(self, X_prev, H_states, C_prev):
        """
        :param X_prev: [B, C, H, W]
        :param H_states: List[[B, C, H, W] * steps]
        :param C_prev: [B, C, H, W]
        """
        # Preprocessing and Tensor-Train Module
        h_acc = None
        for i in reversed(range(self.order)):
            h_states = torch.cat(H_states[i:i + self.window_size], dim=1)  # [B, C * window_size, H, W]
            h_states = self.pre_conv[i](h_states)  # [B, C, H, W]

            if h_acc is None:
                h_acc = h_states  # V(N) initialized as 0
            else:
                h_acc = h_states + self.tt_conv[i - 1](h_acc)  # V(0) 's convolution contains in conv_xh2ifoc

        # standard ConvLSTM
        X_i = torch.concatenate([X_prev, h_acc], dim=1)
        I_i, F_i, O_i, G_i = torch.split(self.conv_xh2ifoc(X_i), self.hidden_channels, dim=1)
        H_i, C_i = checkpoint(lstm_internal_op, I_i, F_i, O_i, G_i, C_prev, use_reentrant=False)
        # H_i, C_i = lstm_internal_op(I_i, F_i, O_i, G_i, C_prev)
        return H_i, C_i

    def forward(self, inputs, Hs=None, c_0=None):
        """
        :param inputs: [B, C, T, H, W]
        :param Hs: List[[B, Ch, H, W] * steps]
        :param c_0: [B, Ch, H, W]
        """
        nb_B, nb_C, nb_T, nb_H, nb_W = inputs.shape
        if Hs is None:
            Hs = [torch.zeros((nb_B, self.hidden_channels, nb_H, nb_W)).type_as(inputs) for _ in range(self.steps)]
        if c_0 is None:
            c_0 = torch.zeros((nb_B, self.hidden_channels, nb_H, nb_W)).type_as(inputs)

        X = []
        C_i = c_0
        for i in range(nb_T):
            X_p = inputs[:, :, i, :, :]
            H_ps = Hs[- self.steps:]
            H_i, C_i = self._step(X_p, H_ps, C_i)
            X.append(H_i)
            Hs.append(H_i)

        X = torch.stack(X, dim=2)  # T*[B, Ch, H, W] -> [B, Ch, T, H, W]
        Hs = Hs[-self.steps:]  # steps * [B, Ch, H, W]
        return X, Hs, C_i


class BiCLSTMCell(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 is_spectral_norm=False,
                 NormLayer=MyGroupNorm,
                 conv_block_type: str = 'conv'):
        super().__init__()

        def _build_clstm():
            return CLSTMCell(in_channels=in_channels, hidden_channels=hidden_channels,
                             is_spectral_norm=is_spectral_norm, NormLayer=NormLayer, conv_block_type=conv_block_type)

        self.clstm_f = _build_clstm()
        self.clstm_b = _build_clstm()

    def forward(self,
                inputs: torch.Tensor,
                hidden_f: torch.Tensor = None,
                hidden_b: torch.Tensor = None,
                memory_f: torch.Tensor = None,
                memory_b: torch.Tensor = None):
        """
        :param inputs: [B, C, T, H, W]
        :param hidden_f: [B, Ch, H, W]
        :param hidden_b: [B, Ch, H, W]
        :param memory_f: [B, Ch, H, W]
        :param memory_b: [B, Ch, H, W]
        """
        outputs_f, hidden_f, memory_f = self.clstm_f(inputs, hidden_f, memory_f)
        outputs_b, hidden_b, memory_b = self.clstm_b(inputs.flip([2]), hidden_b, memory_b)
        outputs_b = outputs_b.flip([2])
        return outputs_f, outputs_b, hidden_f, hidden_b, memory_f, memory_b


class BiHOTTCLSTMCell(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 order: int = 3,
                 steps: int = 5,
                 ranks: int = 8,
                 kernel_size: int = 3,
                 is_spectral_norm=False,
                 NormLayer=nn.BatchNorm2d,
                 conv_block_type: str = 'conv'):
        super().__init__()

        def _build_hottclstm():
            return HOTTCLSTMCell(
                in_channels=in_channels, hidden_channels=hidden_channels, order=order, steps=steps, ranks=ranks,
                kernel_size=kernel_size, is_spectral_norm=is_spectral_norm,
                NormLayer=NormLayer, conv_block_type=conv_block_type
            )

        self.clstm_f = _build_hottclstm()
        self.clstm_b = _build_hottclstm()

    def forward(self,
                inputs: torch.Tensor,
                hidden_list_f: torch.Tensor = None,
                hidden_list_b: torch.Tensor = None,
                memory_f: torch.Tensor = None,
                memory_b: torch.Tensor = None):
        """
        :param inputs: [B, C, T, H, W]
        :param hidden_list_f: List[[B, Ch, H, W] * steps]
        :param hidden_list_b: List[[B, Ch, H, W] * steps]
        :param memory_f: [B, Ch, H, W]
        :param memory_b: [B, Ch, H, W]
        """
        outputs_f, hidden_list_f, memory_f = self.clstm_f(inputs, hidden_list_f, memory_f)
        outputs_b, hidden_list_b, memory_b = self.clstm_b(inputs.flip([2]), hidden_list_b, memory_b)
        outputs_b = outputs_b.flip([2])
        return outputs_f, outputs_b, hidden_list_f, hidden_list_b, memory_f, memory_b

#!/usr/bin/env python3

import torch as th
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class WavenetBlock(nn.Module):
    def __init__(self, layers, channels, kernel_size, groups=1):
        super(WavenetBlock, self).__init__()
        self.layers = layers
        self.filters = nn.ModuleList(
            [
                CausalConv1d(
                    channels, channels, kernel_size, dilation=2 ** i, groups=groups
                )
                # CausalConv1d(channels, channels, kernel_size, dilation=1)
                for i in range(layers)
            ]
        )
        self.gates = nn.ModuleList(
            [
                CausalConv1d(
                    channels, channels, kernel_size, dilation=2 ** i, groups=groups
                )
                # CausalConv1d(channels, channels, kernel_size, dilation=1)
                for i in range(layers)
            ]
        )
        # self.biases = nn.Parameter(th.randn(self.layers))

    def forward(self, ys):
        Zs = [ys]
        for l in range(self.layers):
            # _F = F.relu(self.filters[l](Zs[-1]) + self.biases[l])
            _F = th.tanh(self.filters[l](Zs[-1]))
            # _G = th.sigmoid(self.gates[l](Zs[-1]))
            # _F = _F * _G
            assert _F.size(-1) == ys.size(-1), (_F.size(), ys.size())
            # Zs.append(_F + Zs[-1])
            Zs.append(_F)
        # Z = sum(Zs).squeeze(1)
        # FIXME: make end to end multichannel work
        Z = Zs[-1]
        return Z


class Wavenet(nn.Module):
    def __init__(self, blocks, layers, channels, kernel_size, groups=1):
        super(Wavenet, self).__init__()
        self.first = CausalConv1d(1, channels, 1)
        self.last = CausalConv1d(channels, 1, 1)
        self.blocks = nn.ModuleList(
            [
                WavenetBlock(layers, channels, kernel_size, groups=groups)
                for i in range(blocks)
            ]
        )

    def forward(self, ys):
        # Z = self.first(ys.unsqueeze(1))
        Z = ys
        for block in self.blocks:
            Z = block(Z)
        # Z = self.last(Z).squeeze(1)
        return Z

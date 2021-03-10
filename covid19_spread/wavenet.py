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
    def __init__(
        self, layers, channels, kernel_size, groups=1, embeddings=None, nlin=th.tanh
    ):
        super(WavenetBlock, self).__init__()
        self.layers = layers
        self.nlin = nlin
        self.embeddings = embeddings
        self.filters = nn.ModuleList(
            [
                CausalConv1d(
                    channels, channels, kernel_size, dilation=2 ** i, groups=groups
                )
                for i in range(layers)
            ]
        )
        self.gates = nn.ModuleList(
            [
                CausalConv1d(
                    channels, channels, kernel_size, dilation=2 ** i, groups=groups
                )
                for i in range(layers)
            ]
        )
        if embeddings is not None:
            dim = embeddings.size(1)
            self.weights_f = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
            self.weights_g = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])

    def forward(self, ys):
        Zs = ys
        Zout = th.zeros_like(ys)
        for l in range(self.layers):
            _F = self.filters[l](Zs)
            _G = self.gates[l](Zs)
            if self.embeddings is not None:
                v_f = self.weights_f[l](self.embeddings)
                v_g = self.weights_g[l](self.embeddings)
                _F = _F + v_f.t().unsqueeze(0)
                _G = _G + v_g.t().unsqueeze(0)
            _F = self.nlin(_F)
            _G = th.sigmoid(_G)
            _F = _F * _G
            assert _F.size(-1) == ys.size(-1), (_F.size(), ys.size())
            Zout.add_(_F)
            Zs = _F  # + Zs
        return Zout


class Wavenet(nn.Module):
    def __init__(
        self,
        blocks,
        layers,
        channels,
        kernel_size,
        groups=1,
        embeddings=None,
        nlin=th.tanh,
    ):
        super(Wavenet, self).__init__()
        self.first = CausalConv1d(1, channels, 1)
        self.last = CausalConv1d(channels, 1, 1)
        self.blocks = nn.ModuleList(
            [
                WavenetBlock(
                    layers,
                    channels,
                    kernel_size,
                    groups=groups,
                    embeddings=embeddings,
                    nlin=nlin,
                )
                for i in range(blocks)
            ]
        )

    def forward(self, ys):
        # Z = self.first(ys)
        Z = ys
        for block in self.blocks:
            Z = block(Z)
        # Z = self.last(Z)
        return Z

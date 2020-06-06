#!/usr/bin/env python3

import torch as th
from torch import nn
import torch.nn.functional as F


class CausalConv2d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=(1, 1),
        groups=1,
        bias=True,
    ):

        super(CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size[1] - 1) * dilation[1]

    def forward(self, input):
        return super(CausalConv2d, self).forward(F.pad(input, (self.__padding, 0)))


class CoronaBlock(nn.Module):
    def __init__(self, layers, channels, kernel_size, groups=1):
        super(CoronaBlock, self).__init__()
        self.layers = layers
        self.filters = nn.ModuleList(
            [
                CausalConv2d(
                    channels, channels, kernel_size, dilation=(1, 2 ** i), groups=groups
                )
                # CausalConv1d(channels, channels, kernel_size, dilation=1)
                for i in range(layers)
            ]
        )
        self.gates = nn.ModuleList(
            [
                CausalConv2d(
                    channels, channels, kernel_size, dilation=(1, 2 ** i), groups=groups
                )
                # CausalConv1d(channels, channels, kernel_size, dilation=1)
                for i in range(layers)
            ]
        )
        # self.biases = nn.Parameter(th.randn(self.layers))

    def forward(self, ys, ts):
        ts = ts.unsqueeze(0)
        _zs = th.stack([ys, ts], dim=2)
        Zs = [_zs]
        for l in range(self.layers):
            # _F = F.relu(self.filters[l](Zs[-1]) + self.biases[l])
            # print(Zs[-1].size())
            _F = th.tanh(self.filters[l](Zs[-1]))
            # _G = th.sigmoid(self.gates[l](Zs[-1]))
            # _F = _F * _G
            assert _F.size(-1) == ys.size(-1), (_F.size(), ys.size())
            # Zs.append(_F + Zs[-1])
            # print(_F.size(), ts.size())
            _zs = th.cat([_F, ts.unsqueeze(2)], dim=2)
            Zs.append(_zs)
        # Z = sum(Zs).squeeze(1)
        # FIXME: make end to end multichannel work
        Z = Zs[-1]  # + ys
        return Z


class CoronaNet(nn.Module):
    def __init__(self, blocks, layers, channels, n_features, kernel_size, groups=1):
        super(CoronaNet, self).__init__()
        self.first = CausalConv2d(1, channels, (1, 1))
        self.last = CausalConv2d(channels, 1, (1, 1))
        self.kernel_size = (n_features, kernel_size)
        self.blocks = nn.ModuleList(
            [
                CoronaBlock(layers, channels, self.kernel_size, groups=groups)
                for i in range(blocks)
            ]
        )

    def forward(self, ys, ts):
        # Z = self.first(ys.unsqueeze(1))
        Z = ys.unsqueeze(0)
        for block in self.blocks:
            Z = block(Z, ts)[:, :, 0, :]
        # Z = self.last(Z).squeeze(1)
        return Z

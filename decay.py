#!/usr/bin/env python3
import torch as th
from torch import nn


class BetaConst(nn.Module):
    def __init__(self, regions):
        super(BetaConst, self).__init__()
        self.M = len(regions)
        self.b = nn.Parameter(th.ones(self.M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        return self.fpos(self.b).expand(self.M, t.size(-1))


class BetaExpDecay(nn.Module):
    def __init__(self, regions):
        super(BetaExpDecay, self).__init__()
        M = len(regions)
        self.a = nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.b = nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.c = nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        t = t.unsqueeze(0)
        beta = self.fpos(self.a) * th.exp(-self.fpos(self.b) * t + self.fpos(self.c))
        return beta

    def __repr__(self):
        with th.no_grad():
            return f"Exp = ({self.fpos(self.a).mean().item():.3f}, {self.fpos(self.b).mean().item():.3f})"


class BetaLogistic(nn.Module):
    def __init__(self, regions):
        super(BetaLogistic, self).__init__()
        M = len(regions)
        self.C = nn.Parameter(th.ones(M, 1, dtype=th.float))
        self.k = nn.Parameter(th.ones(M, 1, dtype=th.float))
        self.m = nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        return self.fpos(self.C) / (
            1 + th.exp(self.fpos(self.k) * (t - self.fpos(self.m)))
        )


class BetaPowerLawDecay(nn.Module):
    def __init__(self, regions):
        super(BetaPowerLawDecay, self).__init__()
        M = len(regions)
        # self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        # self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.a = nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.b = nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.c = nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        t = t.unsqueeze(0).float()
        a = self.fpos(self.a)
        m = self.fpos(self.b)
        beta = (a * m).pow_(a) / t ** (a + 1) + self.fpos(self.c)  # pareto
        # beta = (a * t).pow_(-m) + self.fpos(self.c)
        return beta

    def __repr__(self):
        with th.no_grad():
            return f"Power law = ({self.fpos(self.a).mean().item():.3f}, {self.fpos(self.b).mean().item():.3f})"


class BetaRBF(nn.Module):
    def __init__(self, regions, dim, kernel, tmax):
        super(BetaRBF, self).__init__()
        self.M = len(regions)
        self.dim = dim
        self.tmax = tmax
        # self.bs = nn.Parameter(th.ones(self.M, dim, dtype=th.float).fill_(-4))
        # self.bs = nn.Parameter(th.randn(self.M, dim, dtype=th.float))
        self.w = nn.Parameter(th.randn(self.M, dim, dtype=th.float))
        self.c = nn.Parameter(th.randn(self.M, dim, dtype=th.float))
        self.b = nn.Parameter(th.ones(self.M, 1, dtype=th.float))
        self.v = nn.Parameter(th.randn(self.M, 1, dtype=th.float))
        self.temp = nn.Parameter(th.randn(self.M, 1, dtype=th.float))
        self.fpos = F.softplus
        self.kernel = kernel

    def gaussian(self, t):
        t = t.float().unsqueeze(0).unsqueeze(0)
        temp = self.temp.unsqueeze(-1)
        c = self.c.unsqueeze(-1)
        d = (t - c) ** 2  # / self.fpos(temp)
        return th.exp(-d)

    def forward(self, t):
        # reshape tensors
        w = self.w.unsqueeze(-1)
        if self.kernel == "gaussian":
            scores = self.gaussian(t)
        elif self.kernel == "polyharmonic":
            scores = self.polyharmonic(t)
        beta = self.fpos(th.sum(w * scores, dim=1) + self.v * t + self.b)
        # beta = self.fpos(self.bs.narrow(-1, int(t), 1))
        return beta.squeeze()

    def __repr__(self):
        return f"RBF | {self.c.data.mean(dim=0)}"


class BetaPolynomial(nn.Module):
    def __init__(self, regions, degree, tmax):
        super(BetaPolynomial, self).__init__()
        self.M = len(regions)
        self.degree = degree
        self.tmax = tmax
        self.w = nn.Parameter(th.ones(self.M, degree + 1, dtype=th.float))
        self.fpos = F.softplus

    def forward(self, t):
        # reshape tensors
        t = t.float() / self.tmax
        t = [th.pow(t, d).unsqueeze(0) for d in range(self.degree + 1)]
        t = th.cat(t, dim=0)
        beta = self.fpos(th.mm(self.w, t))
        return beta

    def __repr__(self):
        return f"Poly | {self.w.data}"

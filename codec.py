import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from functools import partial

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import NegativeBinomial, Normal, Poisson
import load
import cv
from wavenet import Wavenet


class ODEEncoder(nn.Module):
    def __init__(self, M, dim, input_dim, odeint, method="euler", step_size=1):
        super(ODEEncoder, self).__init__()
        self.weight1 = nn.Linear(input_dim + 2 * dim + 1, dim)
        self.weight2 = nn.Linear(dim + 1, dim)
        self.z0 = nn.Parameter(th.randn(M, dim))
        self.odeint = odeint
        self.method = method
        self.step_size = step_size
        nn.init.xavier_normal_(self.weight1.weight)
        nn.init.xavier_normal_(self.weight2.weight)

    def ode_func(self, x, vs, z0, t, z):
        t_ = t.unsqueeze(0).unsqueeze(0).expand(z.size(0), 1)
        tmp = th.cat([z, z0, x[t.long()], t_], dim=1)
        tmp = self.weight1(tmp)
        tmp = th.cat([tmp, t_], dim=1)
        tmp = th.tanh(tmp)
        tmp = self.weight2(tmp)
        return tmp

    def forward(self, xs, vs, t):
        f_encode = partial(self.ode_func, xs, vs, self.z0)
        zs = self.odeint(
            f_encode,
            self.z0,
            t,
            method=self.method,
            options={"step_size": self.step_size},
        )
        return zs


class RNNEncoder(nn.Module):
    def __init__(self, M, dim, input_dim, layers):
        super(RNNEncoder, self).__init__()
        self.h0 = nn.Parameter(th.zeros(layers, M, dim))
        self.rnn = nn.RNN(input_dim + 1, dim, layers)

        # initialize weights
        for p in self.rnn.parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)

    def forward(self, xs, vs, t):
        t_ = t.unsqueeze(-1).unsqueeze(-1).expand(xs.size(0), xs.size(1), 1)
        x = th.cat([xs, t_], dim=2)
        ht, hn = self.rnn(x, self.h0)
        return ht


class Decoder(nn.Module):
    def __init__(self, M, dim, layers):
        super(Decoder, self).__init__()
        self.M = M
        self.weight_out = nn.Linear(dim, 1)
        self.cross = nn.Parameter(th.zeros(M, M))
        self.wave = Wavenet(1, layers, 1, 2)
        nn.init.xavier_normal_(self.weight_out.weight)

    def forward(self, v, xs, z):
        ys = xs.narrow(2, 0, 1)
        # tmp = th.tanh(z)
        tmp = (z * v.unsqueeze(0)).sum(axis=2)
        tmp = th.mm(tmp, th.sigmoid(self.cross))
        tmp = self.wave(tmp.unsqueeze(1)).squeeze(1)
        # tmp = F.softplus(tmp)
        tmp = th.exp(tmp)
        return tmp.t()


class CODEC(nn.Module):
    """
    CODEC are Ordinary Differential Equations for COVID-19
    """

    def __init__(self, regions, dim, xs, dist, layers, encoder):
        super(CODEC, self).__init__()
        self.M = len(regions)
        self._dist = dist
        self.regions = nn.Parameter(th.randn(self.M, dim))
        self.encoder = encoder
        self.decoder = Decoder(self.M, dim, layers)
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(8))
        self.xs = xs

    def dist(self, scores, ix):
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu)
        elif self._dist == "normal":
            # return Normal(scores, th.exp(self.nu))
            return Normal(scores, 1)
        else:
            raise RuntimeError(f"Unknown loss")

    def forward(self, ix, t):
        vs = self.regions
        zs = self.encoder(self.xs, vs, t)
        score = self.decoder(vs, self.xs, zs)
        assert (score == score).all()
        return self.dist(score.narrow(1, 0, t.size(0) - 1), ix), zs

    def simulate(self, tobs, ys, days, deterministic=True):
        t = th.arange(tobs + days).float().to(ys.device)
        _xs = self.xs
        self.xs = th.zeros(tobs + days, self.M, self.xs.size(2)).to(ys.device)
        self.xs.narrow(0, 0, tobs).copy_(_xs)
        predicted, _ = self.forward(None, t)
        if deterministic:
            predicted = predicted.mean
        else:
            predicted = predicted.sample()

        self.xs = _xs
        predicted = predicted.narrow(1, -days, days)
        return predicted

    def __repr__(self):
        with th.no_grad():
            return f"CODEC"


def train(model, new_cases, optimizer, checkpoint, args):
    device = new_cases.device
    tmax = new_cases.size(1)
    t = th.arange(tmax).float().to(device)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        ix = th.randint(0, new_cases.size(0), (8,), device=new_cases.device)
        ys = new_cases.narrow(1, 1, tmax - 1)
        predicted, zs = model(ix, t)
        reg = args.kinetic * th.pow(zs, 2).sum()

        # compute loss
        _loss = -predicted.log_prob(ys)
        loss = _loss.sum()

        # back prop
        (loss + reg).backward()
        optimizer.step()

        # sometimes infected goes below 0 - prevent that
        # check if initial betas are large enough ...
        # perhaps start from large beta and minimize it??

        # control
        if itr % 100 == 0 or loss == 0:
            # target betas and estimated ones
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                maes = th.abs(ys - predicted.mean)
            print(
                f"Iter {itr:04d} | Loss {loss.item():.2f} | Kin {reg.item():.2f} | MAE {maes.mean():.2f} | {model} "
            )
            th.save(model.state_dict(), checkpoint)
    return model


def _get_dict(args, v, device, regions):
    if hasattr(args, v):
        _feats = []
        if getattr(args, v) is None:
            return None
        for _file in getattr(args, v):
            print(f"Loading {_file}")
            d = th.load(_file)
            feats = None
            for i, r in enumerate(regions):
                if r not in d:
                    print(r)
                    continue
                _f = d[r]
                if feats is None:
                    feats = th.zeros(len(regions), d[r].size(0), _f.size(1))
                feats[i, :, : _f.size(1)] = _f
            # feats.div_(feats.abs().max())
            _feats.append(feats.to(device).float())
        return th.cat(_feats, dim=2)
    else:
        return None


class CodecCV(cv.CV):
    def initialize(self, args):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        cases, regions, basedate = load.load_confirmed_csv(args.fdat)
        assert (cases == cases).all(), th.where(cases != cases)
        new_cases = cases[:, 1:] - cases[:, :-1]
        new_cases = new_cases.to(device)
        assert (new_cases >= 0).all(), th.where(new_cases < 0)

        print("Number of Regions =", new_cases.size(0))
        print("Timeseries length =", new_cases.size(1))
        print("Max increase =", new_cases.max().item())
        tmax = new_cases.size(1) + 1

        # construct input features
        xs = new_cases.clone().t().unsqueeze(-1).to(device).float()
        time_features = _get_dict(args, "time_features", device, regions)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            time_features = time_features.narrow(0, args.t0, new_cases.size(1))
            xs = th.cat([xs, time_features], dim=2)
        print("Feature size = {} x {} x {}".format(*xs.size()))

        if args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        # encoder = ODEEncoder(len(regions), args.dim, xs.size(2), odeint, args.method, 1)
        encoder = RNNEncoder(len(regions), args.dim, xs.size(2), 1)
        self.func = CODEC(regions, args.dim, xs, args.loss, args.layers, encoder).to(
            device
        )
        return new_cases, regions, basedate, device

    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        new_cases, regions, _, device = self.initialize(args)

        optimizer = optim.AdamW(
            self.func.parameters(),
            lr=args.lr,
            betas=[args.momentum, 0.999],
            weight_decay=args.weight_decay,
        )

        # optimization is unstable, quickly it tends to explode
        # check norm_grad weight norm etc...
        # optimizer = optim.RMSprop(func.parameters(), lr=args.lr, weight_decay=weight_decay)

        model = train(self.func, new_cases, optimizer, checkpoint, args)
        return model


CV_CLS = CodecCV

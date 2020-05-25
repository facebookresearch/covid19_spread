import argparse
import numpy as np
import pandas as pd
from datetime import timedelta

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import NegativeBinomial, Normal, Poisson
import load
import cv


class WeightsRNN(nn.Module):
    def __init__(self, regions, dim, layers, tmax, time_features, regularizer):
        super(WeightsRNN, self).__init__()
        self.M = len(regions)
        self.tmax = tmax
        self.fpos = th.sigmoid
        self.time_features = time_features
        self.dim = dim
        input_dim = 1

        self.u0 = nn.Parameter(th.zeros(layers, self.M, dim))
        self.v0 = nn.Parameter(th.zeros(layers, self.M, dim))
        self.s0 = nn.Parameter(th.zeros(layers, self.M, dim))
        self.w_u = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_s = nn.Linear(dim, 1)
        if time_features is not None:
            self.w_time = nn.Linear(time_features.size(2), dim)
            nn.init.xavier_normal_(self.w_time.weight)
            # input_dim += dim
            input_dim += time_features.size(2)
        self.rnn_u = nn.RNN(input_dim, self.dim, layers)
        self.rnn_v = nn.RNN(input_dim, self.dim, layers)
        self.rnn_s = nn.RNN(input_dim, self.dim, layers)
        for p in [self.rnn_u, self.rnn_v, self.rnn_s, self.w_u, self.w_v, self.w_s]:
            self.init_params(p)

    def init_params(self, w):
        # initialize weights
        for p in w.parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)

    def forward(self, t):
        t = t.unsqueeze(-1).unsqueeze(-1).float().div_(self.tmax)
        t = t.expand(t.size(0), self.M, 1)
        if self.time_features is not None:
            # f = self.w_time(self.time_features).narrow(0, 0, t.size(0))
            f = self.time_features.narrow(0, 0, t.size(0))
            # f = f.unsqueeze(0).expand(t.size(0), self.M, self.dim)
            t = th.cat([t, f], dim=2)
        ut, _ = self.rnn_u(t, self.u0)
        vt, _ = self.rnn_v(t, self.v0)

        # print(vt[-1])
        ut = self.w_u(ut)
        vt = self.w_v(vt).transpose(1, 2)
        zs = th.bmm(ut, vt)
        return th.sigmoid(zs)

    def __repr__(self):
        return f"{self.rnn_u}"


class RescalRNN(nn.Module):
    def __init__(self, regions, dim, layers, tmax, time_features, regularizer):
        super(RescalRNN, self).__init__()
        self.M = len(regions)
        self.tmax = tmax
        self.fpos = th.sigmoid
        self.time_features = time_features
        self.dim = dim
        input_dim = 1

        self.u = nn.Parameter(th.randn(self.M, dim))
        self.v = nn.Parameter(th.randn(self.M, dim))
        self.s0 = nn.Parameter(th.zeros(layers, self.M, dim))
        self.w_s = nn.Linear(dim, dim)
        self.regularizer = regularizer
        if time_features is not None:
            self.w_time = nn.Linear(time_features.size(2), dim)
            nn.init.xavier_normal_(self.w_time.weight)
            # input_dim += dim
            input_dim += time_features.size(2)
        self.rnn_s = nn.RNN(input_dim, self.dim, layers)
        for p in [self.rnn_s, self.w_s]:
            self.init_params(p)
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.v)

    def init_params(self, w):
        # initialize weights
        for p in w.parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)

    def forward(self, t):
        t = t.unsqueeze(-1).unsqueeze(-1).float().div_(self.tmax)
        t = t.expand(t.size(0), self.M, 1)
        if self.time_features is not None:
            # f = self.w_time(self.time_features).narrow(0, 0, t.size(0))
            f = self.time_features.narrow(0, 0, t.size(0))
            # f = f.unsqueeze(0).expand(t.size(0), self.M, self.dim)
            t = th.cat([t, f], dim=2)
        st, _ = self.rnn_s(t, self.s0)

        # print(vt[-1])
        ut = self.u
        vt = self.v.t()
        st = self.w_s(st)
        ut = ut.unsqueeze(0).expand(t.size(0), ut.size(0), ut.size(1))
        vt = vt.unsqueeze(0).expand(t.size(0), vt.size(0), vt.size(1))
        zs = th.bmm(ut * st, vt)
        # zs = F.softplus(zs)
        # zs = zs / zs.max()
        zs = th.sigmoid(zs)
        with th.no_grad():
            self.w_stats = (zs.min().item(), zs.mean().item(), zs.max().item())
        return zs

    def __repr__(self):
        return f"{self.rnn_s} | W({self.w_stats[0]:.2f}, {self.w_stats[1]:.2f}, {self.w_stats[2]:.2f})"


class DeepAR(nn.Module):
    def __init__(self, regions, w_net, dist, window_size, graph, features):
        super(DeepAR, self).__init__()
        self.M = len(regions)
        self.repro = nn.Parameter(th.ones((self.M, window_size)))
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(10))
        self.w_net = w_net
        self._dist = dist
        self.window = window_size
        self.graph = graph
        self.features = features
        if graph is not None:
            assert graph.size(0) == self.M, graph.size()
            assert graph.size(1) == self.M, graph.size()
        if features is not None:
            self.w_feat = nn.Linear(features.size(1), 1)
            nn.init.xavier_normal_(self.w_feat.weight)

    def dist(self, scores):
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu)
        elif self._dist == "normal":
            return Normal(scores, F.softplus(self.nu))
        else:
            raise RuntimeError(f"Unknown loss")

    def _score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        offset = self.window - 1

        # beta evolution
        # ys = ys.narrow(1, offset, ys.size(1) - offset)
        Ys = self.w_net(t, ys, self.window)
        assert Ys.size(-1) == t.size(-1) - offset, (Ys.size(-1), t.size(-1), offset)
        return Ys

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        offset = self.window - 1
        length = ys.size(1) - self.window + 1

        # cross-correlation
        W = self.w_net(t)
        Ys = th.bmm(W, ys.t().unsqueeze(-1))
        Ys = Ys.squeeze(-1).t()
        # print(W.size(), ys.size(), Ys.size())

        Z = F.conv1d(
            Ys.unsqueeze(0),
            # F.softplus(self.repro).unsqueeze(1).expand(self.M, 1, self.window),
            F.softplus(self.repro).unsqueeze(1),
            groups=self.M,
        )
        Z = Z.squeeze(0)
        Z.div_(float(self.window))

        assert Z.size(-1) == t.size(-1) - offset, (Z.size(-1), t.size(-1), offset)
        return Z

    def simulate(self, tobs, ys, days, deterministic=True):
        preds = ys.clone()
        offset = self.window + 1
        for d in range(days):
            t = th.arange(tobs + d).to(ys.device) + 1
            s = self.score(t, preds)
            # assert s.size(1) == tobs + d, (s.size(), ys.size(), tobs, d)
            assert (s >= 0).all(), s.squeeze()
            s = s.narrow(1, -1, 1)
            if deterministic:
                y = self.dist(s).mean
            else:
                y = self.dist(s).sample()
            assert (y >= 0).all(), y.squeeze()
            preds = th.cat([preds, y], dim=1)
        preds = preds.narrow(1, -days, days)
        return preds

    def __repr__(self):
        return f"DAR | {self.w_net}"


def train(model, new_cases, regions, optimizer, checkpoint, args):
    print(args)
    print(f"max inc = {new_cases.max()}")
    M = len(regions)
    device = new_cases.device
    tmax = new_cases.size(1)
    t = th.arange(tmax).to(device) + 1

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores = model.score(t, new_cases).clamp(min=1e-8)

        # compute loss
        dist = model.dist(scores.narrow(1, 0, tmax - args.window))
        _loss = -dist.log_prob(
            new_cases.narrow(1, args.window, tmax - args.window)  # .clamp(min=1e-8)
        )
        loss = _loss.sum()

        assert loss == loss, (loss, scores, _loss)

        # back prop
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        # control
        if itr % 50 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                pred = dist.mean[:, -3:]
                gt = new_cases[:, -3:]
                maes = th.abs(gt - pred)
            print(
                f"[{itr:04d}] Loss {loss.item() / M:.2f} | MAE {maes.mean():.2f} | {model} | {args.loss} ({scores[:, -1].min().item():.2f}, {scores[:, -1].max().item():.2f})"
            )
            th.save(model.state_dict(), checkpoint)
    print(f"Train MAE,{maes.mean():.2f}")
    return model


def _get_arg(args, v, device):
    if hasattr(args, v):
        return th.load(getattr(args, v)).to(device).float()
    else:
        return None


class DeepARCV(cv.CV):
    def initialize(self, args):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        cases, regions, basedate = load.load_confirmed_csv(args.fdat)
        new_cases = cases[:, 1:] - cases[:, :-1]
        assert (new_cases >= 0).all()

        new_cases = new_cases.float().to(device)[:, args.t0 :]
        print("Timeseries length", new_cases.size(1))

        args.window = min(args.window, cases.size(1) - 4)
        return new_cases, regions, basedate, device

    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        new_cases, regions, _, device = self.initialize(args)
        tmax = new_cases.size(1) + 1

        # setup optional features
        graph = _get_arg(args, "graph", device)
        features = _get_arg(args, "features", device)
        time_features = _get_arg(args, "time_features", device)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            time_features = time_features.narrow(0, args.t0, cases.size(1))
            print(time_features.size(), new_cases.size())

        # setup beta function
        if args.decay.startswith("latent"):
            dim, layers = args.decay[6:].split("_")
            beta_net = RescalRNN(
                regions, int(dim), int(layers), tmax, time_features, args.weight_decay
            )
        else:
            raise ValueError("Unknown beta function")

        func = DeepAR(regions, beta_net, args.loss, args.window, graph, features).to(
            device
        )
        params = []
        # exclude = {"nu", "beta.w_feat.weight", "beta.w_feat.bias"}
        exclude = {
            "nu",
            # "w_net.u0",
            # "w_net.v0",
            # "w_net.s0",
            # "w_net.w_u.weight",
            # "w_net.w_u.bias",
            # "w_net.w_v.weight",
            # "w_net.w_v.bias",
            # "w_net.w_s.weight",
            # "w_net.w_s.bias",
        }
        for name, p in dict(func.named_parameters()).items():
            wd = 0 if name in exclude else args.weight_decay
            print(name, wd)
            params.append({"params": p, "weight_decay": wd})
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            betas=[args.momentum, 0.999],
            weight_decay=args.weight_decay,
        )

        model = train(func, new_cases, regions, optimizer, checkpoint, args)
        return model


CV_CLS = DeepARCV


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ODE demo")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument("-lr", type=float, default=5e-2)
    parser.add_argument("-weight-decay", type=float, default=0)
    parser.add_argument("-niters", type=int, default=2000)
    parser.add_argument("-amsgrad", default=False, action="store_true")
    parser.add_argument("-loss", default="lsq", choices=["nb", "poisson"])
    parser.add_argument("-decay", default="exp", choices=["exp", "powerlaw", "latent"])
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=5, type=int)
    parser.add_argument("-test-on", default=5, type=int)
    parser.add_argument("-checkpoint", type=str, default="/tmp/ar_model.bin")
    parser.add_argument("-keep-counties", type=int, default=0)
    args = parser.parse_args()

    cv = DeepARCV()

    model = cv.run_train(args, args.checkpoint)

    with th.no_grad():
        forecast = cv.run_simulate(args, model)

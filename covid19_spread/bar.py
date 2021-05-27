#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import numpy as np
import pandas as pd
import warnings
from datetime import timedelta

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import NegativeBinomial, Normal, Poisson
from . import load
from .cross_val import CV
from .common import rebase_forecast_deltas
import yaml
from . import metrics
import click
import sys
from scipy.stats import nbinom, norm
from bisect import bisect_left, bisect_right
from tqdm import tqdm
import timeit
from typing import List
import os


warnings.filterwarnings("ignore", category=UserWarning)


class BetaRNN(nn.Module):
    def __init__(self, M, layers, dim, input_dim, dropout=0.0):
        # initialize parameters
        super(BetaRNN, self).__init__()
        self.h0 = nn.Parameter(th.zeros(layers, M, dim))
        self.rnn = nn.RNN(input_dim, dim, layers, dropout=dropout)
        self.v = nn.Linear(dim, 1, bias=False)
        self.fpos = th.sigmoid

        # initialize weights
        nn.init.xavier_normal_(self.v.weight)
        for p in self.rnn.parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        ht, hn = self.rnn(x, self.h0)
        beta = self.fpos(self.v(ht))
        return beta

    def __repr__(self):
        return str(self.rnn)


class BetaGRU(BetaRNN):
    def __init__(self, M, layers, dim, input_dim, dropout=0.0):
        super().__init__(M, layers, dim, input_dim, dropout)
        self.rnn = nn.GRU(input_dim, dim, layers, dropout=dropout)
        self.rnn.reset_parameters()
        self.h0 = nn.Parameter(th.randn(layers, M, dim))


class BetaLSTM(BetaRNN):
    def __init__(self, M, layers, dim, input_dim, dropout=0.0):
        super().__init__(M, layers, dim, input_dim, dropout)
        self.rnn = nn.LSTM(input_dim, dim, layers, dropout=dropout)
        self.rnn.reset_parameters()
        self.h0 = nn.Parameter(th.zeros(layers, M, dim))
        self.c0 = nn.Parameter(th.randn(layers, M, dim))

    def forward(self, x):
        ht, (hn, cn) = self.rnn(x, (self.h0, self.c0))
        beta = self.fpos(self.v(ht))
        return beta


class BetaLatent(nn.Module):
    def __init__(self, fbeta, regions, tmax, time_features):
        """
        Params
        ======
        - regions: names of regions (list)
        - dim: dimensionality of hidden vector (int)
        - layer: number of RNN layers (int)
        - tmax: maximum observation time (float)
        - time_features: tensor of temporal features (time x region x features)
        """
        super(BetaLatent, self).__init__()
        self.M = len(regions)
        self.tmax = tmax
        self.time_features = time_features
        input_dim = 0

        if time_features is not None:
            input_dim += time_features.size(2)

        self.fbeta = fbeta(self.M, input_dim)

    def forward(self, t, ys):
        x = []
        if self.time_features is not None:
            if self.time_features.size(0) > t.size(0):
                f = self.time_features.narrow(0, 0, t.size(0))
            else:
                f = th.zeros(
                    t.size(0), self.M, self.time_features.size(2), device=t.device
                )
                f.copy_(self.time_features.narrow(0, -1, 1))
                f.narrow(0, 0, self.time_features.size(0)).copy_(self.time_features)
            x.append(f)
        x = th.cat(x, dim=2)
        beta = self.fbeta(x)
        return beta.squeeze().t()

    def apply(self, x):
        ht, hn = self.rnn(x, self.h0)
        return self.fpos(self.v(ht))

    def __repr__(self):
        return str(self.fbeta)


class BAR(nn.Module):
    def __init__(
        self,
        regions,
        beta,
        window,
        dist,
        graph,
        features,
        self_correlation=True,
        cross_correlation=True,
        offset=None,
    ):
        super(BAR, self).__init__()
        self.regions = regions
        self.M = len(regions)
        self.beta = beta
        self.features = features
        self.self_correlation = self_correlation
        self.cross_correlation = cross_correlation
        self.window = window
        self.z = nn.Parameter(th.ones((self.M, 7)).fill_(1))
        self._alphas = nn.Parameter(th.zeros((self.M, self.M)).fill_(-3))
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(8))
        self.scale = nn.Parameter(th.ones((self.M, 1)))
        self._dist = dist
        self.graph = graph
        self.offset = offset
        self.neighbors = self.M
        self.adjdrop = nn.Dropout2d(0.1)
        if graph is not None:
            assert graph.size(0) == self.M, graph.size()
            assert graph.size(1) == self.M, graph.size()
            self.neighbors = graph.sum(axis=1)
        if features is not None:
            self.w_feat = nn.Linear(features.size(1), 1)
            nn.init.xavier_normal_(self.w_feat.weight)

    def dist(self, scores):
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu)
        elif self._dist == "normal":
            return Normal(scores, th.exp(self.nu))
        else:
            raise RuntimeError("Unknown loss")

    def alphas(self):
        alphas = self._alphas
        if self.self_correlation:
            with th.no_grad():
                alphas.fill_diagonal_(-1e10)
        return alphas

    def metapopulation_weights(self):
        alphas = self.alphas()
        W = th.sigmoid(alphas)
        W = W.squeeze(0).squeeze(-1).t()
        if self.graph is not None:
            W = W * self.graph
        return W

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        length = ys.size(-1) - self.window + 1

        # beta evolution
        beta = self.beta(t, ys)

        Z = th.zeros(0).sum()
        if self.self_correlation:
            ws = F.softplus(self.z)
            ws = ws.expand(self.M, self.z.size(1))
            # self-correlation
            Z = F.conv1d(
                F.pad(ys.unsqueeze(0) if ys.ndim == 2 else ys, (self.z.size(1) - 1, 0)),
                ws.unsqueeze(1),
                groups=self.M,
            )
            Z = Z.squeeze(0)
            Z = Z.div(float(self.z.size(1)))

        # cross-correlation
        Ys = th.zeros(0).sum(0)
        W = th.zeros(1, 1)
        if self.cross_correlation:
            W = self.metapopulation_weights()
            Ys = th.stack(
                [
                    F.pad(ys.narrow(-1, i, length), (self.window - 1, 0))
                    for i in range(self.window)
                ]
            )
            orig_shape = Ys.shape
            Ys = Ys.view(-1, Ys.size(-2), Ys.size(-1)) if Ys.ndim == 4 else Ys
            Ys = (
                th.bmm(W.unsqueeze(0).expand(Ys.size(0), self.M, self.M), Ys)
                .view(orig_shape)
                .mean(dim=0)
            )
        with th.no_grad():
            self.train_stats = (Z.mean().item(), Ys.mean().item())

        if self.features is not None:
            Ys = Ys + F.softplus(self.w_feat(self.features))

        Ys = beta * (Z + Ys) / self.neighbors
        return Ys, beta, W

    def simulate(self, tobs, ys, days, deterministic=True, return_stds=False):
        preds = ys.clone()
        self.eval()
        assert tobs == preds.size(-1), (tobs, preds.size())
        stds = []
        for d in range(days):
            t = th.arange(tobs + d, device=ys.device) + 1
            s, _, _ = self.score(t, preds)
            assert (s >= 0).all(), s.squeeze()
            if deterministic:
                y = self.dist(s).mean
            else:
                y = self.dist(s).sample()
            assert (y >= 0).all(), y.squeeze()
            y = y.narrow(-1, -1, 1).clamp(min=1e-8)
            preds = th.cat([preds, y], dim=-1)
            stds.append(self.dist(s).stddev)
        preds = preds.narrow(-1, -days, days)
        self.train()
        if return_stds:
            return preds, stds
        return preds

    def __repr__(self):
        return f"bAR({self.window}) | {self.beta} | EX ({self.train_stats[0]:.1e}, {self.train_stats[1]:.1e})"


def train(model, new_cases, regions, optimizer, checkpoint, args):
    print(args)
    days_ahead = getattr(args, "days_ahead", 1)
    M = len(regions)
    device = new_cases.device
    tmax = new_cases.size(1)
    t = th.arange(tmax, device=device) + 1
    size_pred = tmax - days_ahead
    reg = th.tensor([0.0], device=device)
    target = new_cases.narrow(1, days_ahead, size_pred)

    start_time = timeit.default_timer()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores, beta, W = model.score(t, new_cases)
        scores = scores.clamp(min=1e-8)
        assert scores.dim() == 2, scores.size()
        assert scores.size(1) == size_pred + 1
        assert beta.size(0) == M

        # compute loss
        dist = model.dist(scores.narrow(1, days_ahead - 1, size_pred))
        _loss = dist.log_prob(target)
        loss = -_loss.sum(axis=1).mean()

        stddev = model.dist(scores).stddev.mean()
        # loss += stddev * args.weight_decay

        # temporal smoothness
        if args.temporal > 0:
            reg = (
                args.temporal * th.pow(beta[:, 1:] - beta[:, :-1], 2).sum(axis=1).mean()
            )

        # back prop
        (loss + reg).backward()

        # do AdamW-like update for Granger regularization
        if args.granger > 0:
            with th.no_grad():
                mu = np.log(args.granger / (1 - args.granger))
                y = args.granger
                n = th.numel(model._alphas)
                ex = th.exp(-model._alphas)
                model._alphas.fill_diagonal_(mu)
                de = 2 * (model._alphas.sigmoid().mean() - y) * ex
                nu = n * (ex + 1) ** 2
                _grad = de / nu
                _grad.fill_diagonal_(0)
                r = args.lr * args.eta * n
                model._alphas.copy_(model._alphas - r * _grad)

        # make sure we have no NaNs
        assert loss == loss, (loss, scores, _loss)

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        # take gradient step
        optimizer.step()

        # control
        if itr % 500 == 0:
            time = timeit.default_timer() - start_time
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                length = scores.size(1) - 1
                maes = th.abs(dist.mean - new_cases.narrow(1, 1, length))
                z = model.z
                nu = th.sigmoid(model.nu)
                means = model.dist(scores).mean
                W_spread = (W * (1 - W)).mean()
                _err = W.mean() - args.granger
                print(
                    f"[{itr:04d}] Loss {loss.item():.2f} | "
                    f"Temporal {reg.item():.5f} | "
                    f"MAE {maes.mean():.2f} | "
                    f"{model} | "
                    f"{args.loss} ({means[:, -1].min().item():.2f}, {means[:, -1].max().item():.2f}) | "
                    f"z ({z.min().item():.2f}, {z.mean().item():.2f}, {z.max().item():.2f}) | "
                    f"W ({W.min().item():.2f}, {W.mean().item():.2f}, {W.max().item():.2f}) | "
                    f"W_spread {W_spread:.2f} | mu_err {_err:.3f} | "
                    f"nu ({nu.min().item():.2f}, {nu.mean().item():.2f}, {nu.max().item():.2f}) | "
                    f"nb_stddev ({stddev.data.mean().item():.2f}) | "
                    f"scale ({th.exp(model.scale).mean():.2f}) | "
                    f"time = {time:.2f}s"
                )
                th.save(model.state_dict(), checkpoint)
                start_time = timeit.default_timer()
    print(f"Train MAE,{maes.mean():.2f}")
    return model


def _get_arg(args, v, device, regions):
    if hasattr(args, v):
        print(getattr(args, v))
        fs = []
        for _file in getattr(args, v):
            d = th.load(_file)
            _fs = th.cat([d[r].unsqueeze(0) for r in regions], dim=0)
            fs.append(_fs)
        return th.cat(fs, dim=1).float().to(device)
    else:
        return None


def _get_dict(args, v, device, regions):
    if hasattr(args, v):
        _feats = []
        for _file in getattr(args, v):
            print(f"Loading {_file}")
            d = th.load(_file)
            feats = None
            for i, r in enumerate(regions):
                if r not in d:
                    continue
                _f = d[r]
                if feats is None:
                    feats = th.zeros(len(regions), d[r].size(0), _f.size(1))
                feats[i, :, : _f.size(1)] = _f
            _feats.append(feats.to(device).float())
        return th.cat(_feats, dim=2)
    else:
        return None


class BARCV(CV):
    def initialize(self, args):
        device = th.device(
            "cuda" if th.cuda.is_available() and getattr(args, "cuda", True) else "cpu"
        )
        cases, regions, basedate = load.load_confirmed_csv(args.fdat)
        assert (cases == cases).all(), th.where(cases != cases)

        # Cumulative max across time
        cases = np.maximum.accumulate(cases, axis=1)

        new_cases = th.zeros_like(cases)
        new_cases.narrow(1, 1, cases.size(1) - 1).copy_(cases[:, 1:] - cases[:, :-1])

        assert (new_cases >= 0).all(), new_cases[th.where(new_cases < 0)]
        new_cases = new_cases.float().to(device)[:, args.t0 :]

        print("Number of Regions =", new_cases.size(0))
        print("Timeseries length =", new_cases.size(1))
        print(
            "Increase: max all = {}, max last = {}, min last = {}".format(
                new_cases.max().item(),
                new_cases[:, -1].max().item(),
                new_cases[:, -1].min().item(),
            )
        )
        tmax = new_cases.size(1) + 1

        # adjust max window size to available data
        args.window = min(args.window, new_cases.size(1) - 4)

        # setup optional features
        graph = (
            th.load(args.graph).to(device).float() if hasattr(args, "graph") else None
        )
        features = _get_arg(args, "features", device, regions)
        time_features = _get_dict(args, "time_features", device, regions)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            time_features = time_features.narrow(0, args.t0, new_cases.size(1))
            print("Feature size = {} x {} x {}".format(*time_features.size()))
            print(time_features.min(), time_features.max())

        self.weight_decay = 0
        # setup beta function
        if args.decay.startswith("latent"):
            dim, layers = args.decay[6:].split("_")
            fbeta = lambda M, input_dim: BetaRNN(
                M,
                int(layers),
                int(dim),
                input_dim,
                dropout=getattr(args, "dropout", 0.0),
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        elif args.decay.startswith("lstm"):
            dim, layers = args.decay[len("lstm") :].split("_")
            fbeta = lambda M, input_dim: BetaLSTM(
                M,
                int(layers),
                int(dim),
                input_dim,
                dropout=getattr(args, "dropout", 0.0),
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        elif args.decay.startswith("gru"):
            dim, layers = args.decay[len("gru") :].split("_")
            fbeta = lambda M, input_dim: BetaGRU(
                M,
                int(layers),
                int(dim),
                input_dim,
                dropout=getattr(args, "dropout", 0.0),
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        else:
            raise ValueError("Unknown beta function")

        self.func = BAR(
            regions,
            beta_net,
            args.window,
            args.loss,
            graph,
            features,
            self_correlation=getattr(args, "self_correlation", True),
            cross_correlation=not getattr(args, "no_cross_correlation", False),
            offset=cases[:, 0].unsqueeze(1).to(device).float(),
        ).to(device)

        return new_cases, regions, basedate, device

    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        new_cases, regions, _, device = self.initialize(args)

        params = []
        exclude = {
            "z",
            "nu",
            "_alphas",
            "_alpha_weights",
            "beta.fbeta.h0",
            "beta.fbeta.c0",
            "beta.fbeta.conv.weight",
            "beta.fbeta.conv.bias",
            "scale",
        }
        for name, p in dict(self.func.named_parameters()).items():
            wd = 0 if name in exclude else args.weight_decay
            if wd != 0:
                print(f"Regularizing {name} = {wd}")
            params.append({"params": p, "weight_decay": wd})

        optimizer = optim.AdamW(params, lr=args.lr, betas=[args.momentum, 0.999])

        model = train(self.func, new_cases, regions, optimizer, checkpoint, args)
        return model

    def run_prediction_interval(
        self, means_pth: str, stds_pth: str, intervals: List[float],
    ):
        means = pd.read_csv(means_pth, index_col="date", parse_dates=["date"])
        stds = pd.read_csv(stds_pth, index_col="date", parse_dates=["date"])

        means_t = means.values
        stds_t = stds.values

        multipliers = np.array([norm.ppf(1 - (1 - x) / 2) for x in intervals])
        result = np.empty((means_t.shape[0], means_t.shape[1], len(intervals), 3))
        lower = means_t[:, :, None] - multipliers.reshape(1, 1, -1) * stds_t[:, :, None]
        upper = means_t[:, :, None] + multipliers.reshape(1, 1, -1) * stds_t[:, :, None]
        result = np.stack(
            [np.clip(lower, a_min=0, a_max=None), upper, np.ones(lower.shape)], axis=-1,
        )

        cols = pd.MultiIndex.from_product(
            [means.columns, intervals, ["lower", "upper", "fallback"]]
        )
        result_df = pd.DataFrame(result.reshape(result.shape[0], -1), columns=cols)
        result_df["date"] = means.index
        melted = result_df.melt(
            id_vars=["date"], var_name=["location", "interval", "lower/upper"]
        )
        pivot = melted.pivot(
            index=["date", "location", "interval"],
            columns="lower/upper",
            values="value",
        ).reset_index()
        return pivot.merge(
            means.reset_index().melt(
                id_vars=["date"], var_name="location", value_name="mean"
            ),
            on=["date", "location"],
        ).merge(
            stds.reset_index().melt(
                id_vars=["date"], var_name="location", value_name="std"
            ),
            on=["date", "location"],
        )


CV_CLS = BARCV


@click.group()
def cli():
    pass


@cli.command()
@click.argument("pth")
def simulate(pth):
    chkpnt = th.load(pth)
    mod = BARCV()
    prefix = ""
    if "final_model" in pth:
        prefix = "final_model_"
    cfg = yaml.safe_load(open(f"{os.path.dirname(pth)}/{prefix}bar.yml"))
    args = argparse.Namespace(**cfg["train"])
    new_cases, regions, basedate, device = mod.initialize(args)
    mod.func.load_state_dict(chkpnt)
    res = mod.func.simulate(new_cases.size(1), new_cases, args.test_on)
    df = pd.DataFrame(res.cpu().data.numpy().transpose(), columns=regions)
    df.index = pd.date_range(
        start=pd.to_datetime(basedate) + timedelta(days=1), periods=len(df)
    )
    df = rebase_forecast_deltas(cfg["data"], df)
    gt = pd.read_csv(cfg["data"], index_col="region").transpose()
    gt.index = pd.to_datetime(gt.index)
    print(metrics._compute_metrics(gt, df, nanfill=True))


def main(args):
    parser = argparse.ArgumentParser("beta-AR")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-lr", type=float, default=5e-2)
    parser.add_argument("-weight-decay", type=float, default=0)
    parser.add_argument("-niters", type=int, default=2000)
    parser.add_argument("-amsgrad", default=False, action="store_true")
    parser.add_argument("-loss", default="lsq", choices=["nb", "poisson"])
    parser.add_argument("-decay", default="exp")
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=5, type=int)
    parser.add_argument("-test-on", default=5, type=int)
    parser.add_argument("-checkpoint", type=str, default="/tmp/bar_model.bin")
    parser.add_argument("-window", type=int, default=25)
    parser.add_argument("-momentum", type=float, default=0.99)
    args = parser.parse_args()

    mod = BARCV()

    model = mod.run_train(args.fdat, args, args.checkpoint)

    with th.no_grad():
        forecast = mod.run_simulate(args, model)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in cli.commands:
        cli()
    else:
        main(sys.argv[1:])

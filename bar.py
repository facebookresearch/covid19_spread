import argparse
import copy
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
from cv import rebase_forecast_deltas
import yaml
import metrics
import click
import sys


class BetaLatent(nn.Module):
    def __init__(self, regions, dim, layers, tmax, time_features):
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
        self.fpos = th.sigmoid
        self.time_features = time_features
        self.dim = dim
        input_dim = 2

        if time_features is not None:
            self.w_time = nn.Linear(time_features.size(2), dim)
            nn.init.xavier_normal_(self.w_time.weight)
            input_dim += time_features.size(2)

        # initialize parameters
        self.h0 = nn.Parameter(th.zeros(layers, self.M, dim))
        self.rnn = nn.RNN(input_dim, self.dim, layers)
        self.v = nn.Linear(self.dim, 1, bias=False)

        # initialize weights
        nn.init.xavier_normal_(self.v.weight)
        for p in self.rnn.parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)

    def forward(self, t, ys):
        _ys = th.zeros_like(ys)
        # _ys.narrow(1, 1, ys.size(1) - 1).copy_(ys[:, 1:] - ys[:, :-1])
        _ys.narrow(1, 1, ys.size(1) - 1).copy_(
            th.log(ys[:, 1:] + 1) - th.log(ys[:, :-1] + 1)
        )
        # _ys.narrow(1, 1, ys.size(1) - 1).copy_(ys[:, 1:] / ys[:, :-1].clamp(min=1))
        t = t.unsqueeze(-1).unsqueeze(-1).float()  # .div_(self.tmax)
        t = t.expand(t.size(0), self.M, 1)
        x = [t, _ys.t().unsqueeze(-1)]
        if self.time_features is not None:
            f = th.zeros(t.size(0), self.M, self.time_features.size(2)).to(t.device)
            f.copy_(self.time_features.narrow(0, -1, 1))
            f.narrow(0, 0, self.time_features.size(0)).copy_(self.time_features)
            x.append(f)
        x = th.cat(x, dim=2)
        ht, hn = self.rnn(x, self.h0)
        beta = self.fpos(self.v(ht))
        # beta = beta.expand(beta.size(0), self.M, 1)
        return beta.squeeze().t()
        # return beta.permute(2, 1, 0)

    def __repr__(self):
        return f"{self.rnn}"


class BAR(nn.Module):
    def __init__(
        self, regions, beta, window, dist, graph, features, self_correlation=True
    ):
        super(BAR, self).__init__()
        self.regions = regions
        self.M = len(regions)
        self.beta = beta
        self.features = features
        self.self_correlation = self_correlation
        self.window = window
        self.z = nn.Parameter(th.zeros((1, window)).fill_(1))
        # self.z = nn.Parameter(th.ones((self.M, window)).fill_(1))
        self._alphas = nn.Parameter(th.zeros((self.M, self.M)).fill_(-5))
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(8))
        self._dist = dist
        self.graph = graph
        if graph is not None:
            assert graph.size(0) == self.M, graph.size()
            assert graph.size(1) == self.M, graph.size()
        if features is not None:
            self.w_feat = nn.Linear(features.size(1), 1)
            nn.init.xavier_normal_(self.w_feat.weight)

    # nn.init.xavier_normal_(self.z)
    # nn.init.xavier_normal_(self._alphas)

    def dist(self, scores):
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu)
        elif self._dist == "normal":
            # return Normal(scores, th.exp(self.nu))
            return Normal(scores, 1)
        else:
            raise RuntimeError(f"Unknown loss")

    def alphas(self):
        alphas = self._alphas
        if self.self_correlation:
            with th.no_grad():
                alphas.fill_diagonal_(-1e10)
        return alphas

    def metapopulation_weights(self):
        alphas = self.alphas()
        W = th.sigmoid(alphas)
        # W = F.softplus(self.alphas)
        if self.graph is not None:
            W = W * self.graph
        return W

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        offset = self.window - 1
        length = ys.size(1) - self.window + 1

        Z = th.zeros(0).sum()
        if self.self_correlation:
            ws = F.softplus(self.z)
            ws = ws.expand(self.M, self.window)
            # self-correlation
            Z = F.conv1d(ys.unsqueeze(0), ws.unsqueeze(1), groups=self.M)
            Z = Z.squeeze(0)
            Z = Z.div(float(self.window))

        # cross-correlation
        W = self.metapopulation_weights()
        Ys = th.stack([ys.narrow(1, i, length) for i in range(self.window)])
        Ys = th.bmm(W.unsqueeze(0).expand(self.window, self.M, self.M), Ys).mean(dim=0)
        # Ys = th.bmm(W, Ys).mean(dim=0)
        with th.no_grad():
            self.train_stats = (Z.sum().item(), Ys.sum().item())

        # beta evolution
        beta = self.beta(t, ys)
        ys = ys.narrow(1, offset, ys.size(1) - offset)
        beta = beta.narrow(-1, -ys.size(1), ys.size(1))
        if self.features is not None:
            Ys = Ys + F.softplus(self.w_feat(self.features))

        # Ys = F.softplus(self.out_proj(th.stack([beta, Z, Ys], dim=-1)).squeeze())
        Ys = beta * (Z + Ys)

        assert Ys.size(-1) == t.size(-1) - offset, (Ys.size(-1), t.size(-1), offset)
        return Ys, Z, beta, W

    def simulate(self, tobs, ys, days, deterministic=True):
        preds = ys.clone()
        self.eval()
        assert tobs == preds.size(1), (tobs, preds.size())
        for d in range(days):
            t = th.arange(tobs + d).to(ys.device) + 1
            s, _, _, _ = self.score(t, preds)
            assert (s >= 0).all(), s.squeeze()
            s = s.narrow(1, -1, 1).clamp(min=1e-8)
            if deterministic:
                y = self.dist(s).mean
            else:
                y = self.dist(s).sample()
            assert (y >= 0).all(), y.squeeze()
            preds = th.cat([preds, y], dim=1)
        preds = preds.narrow(1, -days, days)
        self.train()
        return preds

    def __repr__(self):
        return f"bAR({self.window}) | {self.beta} | EX ({self.train_stats[0]:.1e}, {self.train_stats[1]:.1e})"


def train(model, new_cases, regions, optimizer, checkpoint, args):
    print(args)
    M = len(regions)
    device = new_cases.device
    tmax = new_cases.size(1)
    t = th.arange(tmax).to(device) + 1
    size_pred = tmax - args.window
    reg = th.tensor([0]).to(device)
    target = new_cases.narrow(1, args.window, size_pred)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores, scores_restricted, beta, W = model.score(t, new_cases)
        scores = scores.clamp(min=1e-8)
        # scores_restricted = scores_restricted.clamp(min=1e-8)
        assert scores.size(1) == size_pred + 1
        assert size_pred + args.window == new_cases.size(1)
        assert beta.size(0) == M

        # compute loss
        length = scores.size(1) - 1
        dist = model.dist(scores.narrow(1, 0, size_pred))
        _loss = -dist.log_prob(target)
        loss = _loss.sum()

        # temporal smoothness
        if args.temporal > 0:
            reg = th.pow(beta[:, 1:] - beta[:, :-1], 2).sum()

        # back prop
        (loss + args.temporal * reg).backward()

        # do AdamW-like update for Granger regularization
        if args.granger > 0:
            with th.no_grad():
                mu = np.log(args.granger / (1 - args.granger))
                r = args.lr * args.eta
                err = model.alphas() - mu
                err.fill_diagonal_(0)
                model._alphas.copy_(model._alphas - r * err)

        # make sure we have no NaNs
        assert loss == loss, (loss, scores, _loss)

        # take gradient step
        optimizer.step()

        # control
        if itr % 100 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                maes = th.abs(dist.mean - new_cases.narrow(1, 1, length))
                z = F.softplus(model.z)
                nu = th.sigmoid(model.nu)
                print(
                    f"[{itr:04d}] Loss {loss.item():.2f} | "
                    f"Temporal {reg.item():.5f} | "
                    f"MAE {maes.mean():.2f} | "
                    f"{model} | "
                    f"{args.loss} ({scores[:, -1].min().item():.2f}, {scores[:, -1].max().item():.2f}) | "
                    f"z ({z.min().item():.2f}, {z.mean().item():.2f}, {z.max().item():.2f}) | "
                    f"alpha ({W.min().item():.2f}, {W.mean().item():.2f}, {W.max().item():.2f}) | "
                    f"nu ({nu.min().item():.2f}, {nu.mean().item():.2f}, {nu.max().item():.2f})"
                )
            th.save(model.state_dict(), checkpoint)
    print(f"Train MAE,{maes.mean():.2f}")
    return model


def _get_arg(args, v, device):
    if hasattr(args, v):
        arg = getattr(args, v)
        if isinstance(arg, list):
            ts = [th.load(a).to(device).float() for a in arg]
            return th.cat(ts, dim=2)
        return th.load(arg).to(device).float()
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
            # feats.div_(feats.abs().max())
            _feats.append(feats.to(device).float())
        return th.cat(_feats, dim=2)
    else:
        return None


from glob import glob
from typing import List
from cv import BestRun
import os


class BARCV(cv.CV):
    # def model_selection(self, basedir: str) -> List[BestRun]:
    #     """
    #     Evaluate a sweep returning a list of models to retrain on the full dataset.
    #     """
    #     runs = []
    #     for metrics_pth in glob(os.path.join(basedir, "*/metrics.csv")):
    #         metrics = pd.read_csv(metrics_pth, index_col="Measure")
    #         runs.append(
    #             {
    #                 "pth": os.path.dirname(metrics_pth),
    #                 "mae": metrics.loc["MAE"][-1]
    #             }
    #         )
    #     df = pd.DataFrame(runs)
    #     return [
    #         BestRun(row.pth, f"best_mae_{i}") for i, (_, row) in enumerate(df.iterrows())
    #     ]

    def initialize(self, args):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        cases, regions, basedate = load.load_confirmed_csv(args.fdat)
        assert (cases == cases).all(), th.where(cases != cases)
        new_cases = cases[:, 1:] - cases[:, :-1]
        assert (new_cases >= 0).all(), th.where(new_cases < 0)

        # prepare population
        # populations = load.load_populations_by_region(args.fpop, regions=regions)
        # populations = th.from_numpy(populations["population"].values).to(device)

        new_cases = new_cases.float().to(device)[:, args.t0 :]
        print("Number of Regions =", new_cases.size(0))
        print("Timeseries length =", new_cases.size(1))
        print("Max increase =", new_cases.max().item())
        tmax = new_cases.size(1) + 1

        # adjust max window size to available data
        args.window = min(args.window, new_cases.size(1) - 4)

        # setup optional features
        graph = _get_arg(args, "graph", device)
        features = _get_arg(args, "features", device)
        time_features = _get_dict(args, "time_features", device, regions)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            time_features = time_features.narrow(0, args.t0, new_cases.size(1))
            print("Feature size = {} x {} x {}".format(*time_features.size()))

        self.weight_decay = 0
        # setup beta function
        if args.decay == "const":
            beta_net = decay.BetaConst(regions)
        elif args.decay == "exp":
            beta_net = decay.BetaExpDecay(regions)
        elif args.decay == "logistic":
            beta_net = decay.BetaLogistic(regions)
        elif args.decay == "powerlaw":
            beta_net = decay.BetaPowerLawDecay(regions)
        elif args.decay.startswith("poly"):
            degree = int(args.decay[4:])
            beta_net = decay.BetaPolynomial(regions, degree, tmax)
        elif args.decay.startswith("rbf"):
            dim = int(args.decay[3:])
            beta_net = decay.BetaRBF(regions, dim, "gaussian", tmax)
        elif args.decay.startswith("latent"):
            dim, layers = args.decay[6:].split("_")
            beta_net = BetaLatent(regions, int(dim), int(layers), tmax, time_features)
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
        ).to(device)

        return new_cases, regions, basedate, device

    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        new_cases, regions, _, device = self.initialize(args)

        params = []
        # exclude = {"nu", "beta.w_feat.weight", "beta.w_feat.bias"}
        exclude = {"nu", "_alphas", "beta.h0"}
        for name, p in dict(self.func.named_parameters()).items():
            wd = 0 if name in exclude else args.weight_decay
            if wd != 0:
                print(f"Regularizing {name} = {wd}")
            params.append({"params": p, "weight_decay": wd})
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            betas=[args.momentum, 0.999],
            weight_decay=args.weight_decay,
            amsgrad=False,
        )

        model = train(self.func, new_cases, regions, optimizer, checkpoint, args)

        return model


CV_CLS = BARCV


@click.group()
def cli():
    pass


@cli.command()
@click.argument("pth")
def simulate(pth):
    chkpnt = th.load(pth)
    cv = BARCV()
    if "final_model" in pth:
        prefix = "final_model_"
    cfg = yaml.safe_load(open(f"{os.path.dirname(pth)}/{prefix}bar.yml"))
    args = argparse.Namespace(**cfg["train"])
    new_cases, regions, basedate, device = cv.initialize(args)
    cv.func.load_state_dict(chkpnt)
    res = cv.func.simulate(new_cases.size(1), new_cases, args.test_on)
    df = pd.DataFrame(res.cpu().data.numpy().transpose(), columns=regions)
    df.index = pd.date_range(
        start=pd.to_datetime(basedate) + timedelta(days=1), periods=len(df)
    )
    df = rebase_forecast_deltas(cfg["data"], df)
    gt = pd.read_csv(cfg["data"], index_col="region").transpose()
    gt.index = pd.to_datetime(gt.index)
    print(metrics._compute_metrics(gt, df))


def main(args):
    parser = argparse.ArgumentParser("ODE demo")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
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

    cv = BARCV()

    model = cv.run_train(args.fdat, args, args.checkpoint)

    with th.no_grad():
        forecast = cv.run_simulate(args, model)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in cli.commands:
        cli()
    else:
        main(sys.argv[1:])

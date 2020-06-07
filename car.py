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
from wavenet import Wavenet
from coronanet import CoronaNet


class CAR(nn.Module):
    def __init__(
        self, regions, blocks, layers, kernel_size, dist, graph, features, time_features
    ):
        super(CAR, self).__init__()
        self.M = len(regions)
        self.blocks = blocks
        self.layers = layers
        self.kernel_size = kernel_size
        self.features = features
        self.time_features = None
        self.alphas = nn.Parameter(th.ones((self.M, self.M)).fill_(0))
        self.decoder_head = nn.Parameter(th.ones((self.M, self.M)).fill_(1.0 / self.M))
        # self.alphas = nn.Parameter(th.ones((self.M, self.M)).fill_(1.0 / self.M))
        n_features = 2
        if time_features is not None:
            self.time_features = time_features.permute(1, 2, 0)
            n_features += self.time_features.size(1)
        self.encoder = CoronaNet(
            blocks, layers, self.M, n_features, kernel_size, groups=self.M
        )
        self.decoder = Wavenet(blocks, layers, self.M, kernel_size, groups=self.M)
        # self.encoder = CoronaNet(blocks, layers, self.M, 2, kernel_size, groups=self.M)
        # self.decoder = Wavenet(blocks, layers, self.M, 2, kernel_size, groups=self.M)
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(8))
        self._dist = dist
        self.graph = graph
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

    def metapopulation_weights(self):
        with th.no_grad():
            self.alphas.fill_diagonal_(-1e10)
        # self.alphas.fill_diagonal_(0)
        # W = F.softmax(self.alphas, dim=1)
        W = th.sigmoid(self.alphas)
        # W = F.softplus(self.alphas)
        # W = self.alphas
        # W = W / W.sum()
        if self.graph is not None:
            W = W * self.graph
        return W

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())

        # feature rep
        t = t.unsqueeze(0).expand_as(ys).float()
        if self.time_features is not None:
            # f = self.w_time(self.time_features).narrow(0, 0, t.size(0))
            f = self.time_features.narrow(0, 0, t.size(0))
            # f = f.unsqueeze(0).expand(t.size(0), self.M, self.dim)
            t = th.cat([t.unsqueeze(1), f], dim=1)

        # encoder
        Z = self.encoder(ys, t)
        Z = Z.squeeze(0)

        # coupling
        W = self.metapopulation_weights()
        Z = th.mm(W, Z)

        # decoder
        ws = self.decoder(Z.unsqueeze(0))
        ws = ws.squeeze(0)
        ws = th.mm(self.decoder_head, ws)

        # cross-correlation
        # Ys = F.pad(Z, (self.window, 0))
        # Ys = th.stack([Ys.narrow(1, i, length) for i in range(self.window)])
        # Ys = th.bmm(W.unsqueeze(0).expand(self.window, self.M, self.M), Ys).mean(dim=0)
        # Ys = th.bmm(W, Ys).mean(dim=0)
        with th.no_grad():
            self.train_stats = (ws.min().item(), ws.max().item())

        # link function
        Ys = F.softplus(ws) * ys

        # assert Ys.size(-1) == t.size(-1) - offset, (Ys.size(-1), t.size(-1), offset)
        return Ys, W, None

    def simulate(self, tobs, ys, days, deterministic=True):
        preds = ys.clone()
        assert tobs == preds.size(1), (tobs, preds.size())
        for d in range(days):
            t = th.arange(tobs + d).to(ys.device) + 1
            s, _, _ = self.score(t, preds)
            s = s.narrow(1, -1, 1)
            assert (s >= 0).all(), s.squeeze()
            if deterministic:
                y = self.dist(s).mean
            else:
                y = self.dist(s).sample()
            assert (y >= 0).all(), y.squeeze()
            preds = th.cat([preds, y], dim=1)
        preds = preds.narrow(1, -days, days)
        return preds

    def __repr__(self):
        return f"CAR({self.blocks}, {self.layers}, {self.kernel_size}) | EX ({self.train_stats[0]:.1e}, {self.train_stats[1]:.1e})"


def regularize(A, beta, ys):
    ys = ys.narrow(1, -beta.size(1), beta.size(1))
    q = beta * ys
    Aq = th.mm(A, q)
    return 0.1 * (
        th.mm(q.t(), q).sum() - 2 * th.mm(q.t(), Aq).sum() + th.mm(Aq.t(), Aq).sum()
    )


def train(model, new_cases, regions, optimizer, checkpoint, args):
    print(args)
    print(f"max inc = {new_cases.max()}")
    M = len(regions)
    device = new_cases.device
    tmax = new_cases.size(1)
    t = th.arange(tmax).to(device) + 1

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores, A, beta = model.score(t, new_cases)
        scores = scores.clamp(min=1e-8)
        assert scores.size(1) == new_cases.size(1), (scores.size(), new_cases.size())

        # compute loss
        length = scores.size(1) - 1
        dist = model.dist(scores.narrow(1, 0, length))
        _loss = -dist.log_prob(new_cases.narrow(1, 1, length))
        loss = _loss.sum()
        reg = 0  # regularize(A, beta, new_cases)
        # if args.weight_decay > 0:
        #    reg = args.weight_decay * (
        #        model.metapopulation_weights().sum()  # + F.softplus(model.repro).sum()
        #    )

        assert loss == loss, (loss, scores, _loss)

        # back prop
        (loss + reg).backward()
        optimizer.step()

        # control
        if itr % 100 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                pred = dist.mean[:, -3:]
                gt = new_cases[:, -3:]
                maes = th.abs(gt - pred)
                _a = model.metapopulation_weights()
                print(
                    f"[{itr:04d}] Loss {loss.item():.2f} | "
                    # f"Coupling {reg.item() / M:.2f} | "
                    f"MAE {maes.mean():.2f} | "
                    f"{model} | "
                    f"{args.loss} ({scores[:, -1].min().item():.2f}, {scores[:, -1].max().item():.2f}) | "
                    f"alpha ({_a.min().item():.2f}, {_a.mean().item():.2f}, {_a.max().item():.2f})"
                )
            th.save(model.state_dict(), checkpoint)
    print(f"Train MAE,{maes.mean():.2f}")
    return model


def _get_arg(args, v, device):
    if hasattr(args, v):
        return th.load(getattr(args, v)).to(device).float()
    else:
        return None


class CARCV(cv.CV):
    def initialize(self, args):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        cases, regions, basedate = load.load_confirmed_csv(args.fdat)
        assert (cases == cases).all(), th.where(cases != cases)
        new_cases = cases[:, 1:] - cases[:, :-1]
        assert (new_cases >= 0).all(), th.where(new_cases < 0)

        new_cases = new_cases.float().to(device)[:, args.t0 :]
        print("Timeseries length", new_cases.size(1))

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
            time_features = time_features.narrow(0, args.t0, new_cases.size(1))
            print(time_features.size(), new_cases.size())

        func = CAR(
            regions,
            args.blocks,
            args.layers,
            args.kernel_size,
            args.loss,
            graph,
            features,
            time_features,
        ).to(device)
        params = []
        # exclude = {"nu", "beta.w_feat.weight", "beta.w_feat.bias"}
        # exclude = {"nu", "alphas"}
        exclude = {"nu"}
        for name, p in dict(func.named_parameters()).items():
            wd = (
                0
                if name in exclude
                or name.startswith("encoder")
                or name.startswith("decoder")
                else args.weight_decay
            )
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


CV_CLS = CARCV


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

    cv = ARCV()

    model = cv.run_train(args, args.checkpoint)

    with th.no_grad():
        forecast = cv.run_simulate(args, model)

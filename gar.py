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


class GAR(nn.Module):
    def __init__(self, regions, dist, blocks, layers, kernel_size, graph, features):
        super(GAR, self).__init__()
        self.M = len(regions)
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(10))
        self.dim = 16
        self.wavenet = Wavenet(blocks, layers, self.dim, kernel_size)
        self._dist = dist
        self.graph = graph
        self.features = features
        self.window = kernel_size
        self.W = nn.Parameter(th.randn((self.dim, self.dim)))
        self.E = nn.Parameter(th.randn((self.M, self.dim)))
        if graph is not None:
            assert graph.size(0) == self.M, graph.size()
            assert graph.size(1) == self.M, graph.size()
        if features is not None:
            self.w_feat = nn.Linear(features.size(1), 1)
            nn.init.xavier_normal_(self.w_feat.weight)
        nn.init.xavier_normal_(self.W)

    def dist(self, scores):
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu)
        elif self._dist == "normal":
            return Normal(scores, F.softplus(self.nu))
        else:
            raise RuntimeError(f"Unknown loss")

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())

        length = ys.size(1) - self.window + 1
        # Z = th.tanh(th.mm(self.Vt, ys))
        # Z = th.bmm(self.W, Z).mean(dim=0)

        _W = self.W.unsqueeze(0).expand(self.M, self.dim, self.dim)
        _E = self.E.unsqueeze(-1)
        Z = self.wavenet(ys)
        # Z = th.bmm(th.bmm(_W, Z).transpose(1, 2), _E).squeeze()
        # Z = th.stack([Z.narrow(-1, i, length) for i in range(self.window)])
        # Y = th.stack([ys.narrow(-1, i, length) for i in range(self.window)])
        # Z = (Z * Y).mean(dim=0)
        Z = F.softplus(Z)
        # print(Z.size(), Y.size(), t.size())
        # Z = th.mm(self.U, Z)
        # if self.features is not None:
        #    beta = beta + th.sigmoid(self.w_feat(self.features))

        # beta evolution
        # Z = F.softplus(Z)

        # assert Ys.size(-1) == t.size(-1) - offset, (Ys.size(-1), t.size(-1), offset)
        return Z

    def simulate(self, tobs, ys, days, deterministic=True):
        preds = ys.clone()
        offset = self.window + 1
        # t = th.arange(offset).to(ys.device) + (tobs + 1 - offset)
        for d in range(days):
            t = th.arange(tobs + d).to(ys.device) + 1
            s = self.score(t, preds)
            # print(s.size(), ys.size(), tobs, d)
            # assert s.size(1) == tobs + d, (s.size(), ys.size(), tobs, d)
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
        return f"GAR"


def train(model, new_cases, regions, optimizer, checkpoint, args, tb_writer):
    print(args)
    M = len(regions)
    device = new_cases.device
    print(f"max inc = {new_cases.max()}")
    tmax = new_cases.size(1)
    t = th.arange(tmax).to(device) + 1
    size_pred = tmax - args.window

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores = model.score(t, new_cases).clamp(min=1e-8)
        # print(scores.size(), new_cases.size(), size_pred)
        # fail

        # compute loss
        dist = model.dist(scores.narrow(1, 0, tmax - 1))
        _loss = -dist.log_prob(new_cases.narrow(1, 1, tmax - 1))  # .clamp(min=1e-8)
        # dist = model.dist(scores.narrow(1, 0, size_pred))
        # _loss = -dist.log_prob(new_cases.narrow(1, args.window, size_pred))
        loss = _loss.sum()
        reg = 0

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
                # _a = model.metapopulation_weights()
                tb_writer.add_scalar("MAE", maes.mean(), itr)
                tb_writer.add_scalar("Loss", loss.item(), itr)
                tb_writer.add_scalar("NP_min", scores[:, -1].min())
                print(
                    f"[{itr:04d}] Loss {loss.item() / M:.2f} | "
                    f"MAE {maes.mean():.2f} | "
                    f"{model} | "
                    f"{args.loss} ({scores[:, -1].min().item():.2f}, {scores[:, -1].max().item():.2f}) | "
                    # f"alpha ({_a.min().item():.2f}, {_a.mean().item():.2f}, {_a.max().item():.2f})"
                )
            th.save(model.state_dict(), checkpoint)
    print(f"Train MAE,{maes.mean():.2f}")
    return model


def _get_arg(args, v, device):
    if hasattr(args, v):
        return th.load(getattr(args, v)).to(device).float()
    else:
        return None


class GARCV(cv.CV):
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
            time_features = time_features.narrow(0, args.t0, new_cases.size(1))
            print(time_features.size(), new_cases.size())

        weight_decay = 0
        # setup beta function

        func = GAR(
            regions, args.loss, args.blocks, args.layers, args.window, graph, features
        ).to(device)
        params = []
        # exclude = {"nu", "beta.w_feat.weight", "beta.w_feat.bias"}
        # exclude = {"nu", "repro"}
        exclude = {"nu"}
        for name, p in dict(func.named_parameters()).items():
            wd = 0 if name in exclude else weight_decay
            print(name, wd)
            params.append({"params": p, "weight_decay": wd})
        optimizer = optim.AdamW(
            params, lr=args.lr, betas=[args.momentum, 0.999], weight_decay=weight_decay
        )

        model = train(
            func, new_cases, regions, optimizer, checkpoint, args, self.tb_writer
        )

        return model


CV_CLS = GARCV


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

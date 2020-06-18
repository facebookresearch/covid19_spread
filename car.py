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
from wavenet import Wavenet
from coronanet import CoronaNet


class BetaLatent(nn.Module):
    def __init__(self, regions, dim, layers, tmax, time_features):
        super(BetaLatent, self).__init__()
        self.M = len(regions)
        self.tmax = tmax
        self.fpos = th.sigmoid
        self.time_features = time_features
        self.dim = dim
        input_dim = 1

        self.h0 = nn.Parameter(th.zeros(layers, self.M, dim))
        # self.h0 = nn.Parameter(th.zeros(layers, 1, dim))
        if time_features is not None:
            self.w_time = nn.Linear(time_features.size(2), dim)
            nn.init.xavier_normal_(self.w_time.weight)
            # input_dim += dim
            input_dim += time_features.size(2)
        self.rnn = nn.RNN(input_dim, self.dim, layers)
        self.v = nn.Linear(self.dim, 1, bias=False)

        # initialize weights
        nn.init.xavier_normal_(self.v.weight)
        for p in self.rnn.parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)

    def forward(self, t):
        t = t.unsqueeze(-1).unsqueeze(-1).float()  # .div_(self.tmax)
        t = t.expand(t.size(0), self.M, 1)
        if self.time_features is not None:
            # f = self.w_time(self.time_features).narrow(0, 0, t.size(0))
            f = self.time_features.narrow(0, 0, t.size(0))
            # f = f.unsqueeze(0).expand(t.size(0), self.M, self.dim)
            t = th.cat([t, f], dim=2)
        ht, hn = self.rnn(t, self.h0)
        beta = self.fpos(self.v(ht))
        # beta = beta.expand(beta.size(0), self.M, 1)
        return beta.squeeze().t()
        # return beta.permute(2, 1, 0)

    def __repr__(self):
        return f"{self.rnn}"


class CAR(nn.Module):
    def __init__(self, regions, beta, window, dist, graph, features):
        super(CAR, self).__init__()
        self.regions = regions
        self.M = len(regions)
        self.beta = beta
        # self.beta_restricted = copy.deepcopy(beta)
        self.features = features
        self.window = window
        # ``self.z = nn.Parameter(th.ones((1, window)).fill_(1))
        self.z = nn.Parameter(th.ones((self.M, window)).fill_(1))
        self.alphas = nn.Parameter(th.ones((self.M, self.M)).fill_(0))
        # self.alphas = nn.Parameter(th.ones((self.M, self.M)).fill_(1.0 / self.M))
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(10))
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
        #    self.alphas.fill_diagonal_(0)
        # W = F.softmax(self.alphas, dim=1)
        # W = th.sigmoid(self.alphas)
        # W = th.tanh(self.alphas)
        W = F.softplus(self.alphas)
        # W = self.alphas
        # W = W / W.sum()
        if self.graph is not None:
            W = W * self.graph
        return W

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        offset = self.window - 1
        length = ys.size(1) - self.window + 1

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
        ys = ys.narrow(1, offset, ys.size(1) - offset)
        beta = self.beta(t)
        beta = beta.narrow(-1, -ys.size(1), ys.size(1))
        if self.features is not None:
            fail
        Ys = beta * (Z + Ys)

        assert Ys.size(-1) == t.size(-1) - offset, (Ys.size(-1), t.size(-1), offset)
        return Ys, Z, beta, W

    def simulate(self, tobs, ys, days, deterministic=True):
        preds = ys.clone()
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
        return preds

    def __repr__(self):
        return f"CAR({self.window}) | {self.beta} | EX ({self.train_stats[0]:.1e}, {self.train_stats[1]:.1e})"


def train(model, new_cases, regions, optimizer, checkpoint, args):
    print(args)
    print(f"max inc = {new_cases.max()}")
    print(f"number of regions = {len(regions)}")
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
        # dist_restricted = model.dist(scores_restricted.narrow(1, 0, size_pred))
        _loss = -dist.log_prob(target)
        loss = _loss.sum()
        if args.granger > 0:
            # granger = args.granger * -dist_restricted.log_prob(target).sum()
            # w_mean = W.mean()
            # reg = M * M * args.granger * (th.mean((W - w_mean) ** 2) / w_mean)
            # reg = M * M * args.granger * (th.mean((W - w_mean) ** 2))
            # reg -= M * args.granger * w_mean
            # reg = M * M * args.granger * (th.std(W) / th.mean(W))
            # reg = (
            #    args.granger - th.clamp(W, max=args.granger)
            # ).sum() + args.weight_decay * W.sum()
            # reg = args.weight_decay * th.clamp(args.granger - W, max=0).sum()
            reg = th.clamp(args.granger - W, min=0).sum()
            # + args.weight_decay * W.sum()
            # + M * M * args.granger * (th.mean((W - w_mean) ** 2) / w_mean)
            # + th.clamp(W, min=0.02).sum()

        assert loss == loss, (loss, scores, _loss)

        # back prop
        (loss + reg).backward()
        optimizer.step()

        # control
        if itr % 100 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                maes = th.abs(dist.mean - new_cases.narrow(1, 1, length))
                z = F.softplus(model.z)
                print(
                    f"[{itr:04d}] Loss {loss.item():.2f} | "
                    f"Granger {reg.item() / M:.2f} | "
                    f"MAE {maes.mean():.2f} | "
                    f"{model} | "
                    f"{args.loss} ({scores[:, -1].min().item():.2f}, {scores[:, -1].max().item():.2f}) | "
                    f"z ({z.min().item():.2f}, {z.mean().item():.2f}, {z.max().item():.2f}) | "
                    f"alpha ({W.min().item():.2f}, {W.mean().item():.2f}, {W.max().item():.2f})"
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
        tmax = new_cases.size(1) + 1

        # setup optional features
        graph = _get_arg(args, "graph", device)
        features = _get_arg(args, "features", device)
        time_features = _get_arg(args, "time_features", device)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            time_features = time_features.narrow(0, args.t0, new_cases.size(1))
            print(time_features.size(), new_cases.size())

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
        elif args.decay.startswith("wave"):
            blocks, layers, dim = args.decay[4:].split("_")
            beta_net = BetaWavenet(regions, int(blocks), int(layers), int(2))
            self.weight_decay = args.weight_decay
        else:
            raise ValueError("Unknown beta function")

        self.func = CAR(regions, beta_net, args.window, args.loss, graph, features).to(
            device
        )

        return new_cases, regions, basedate, device

    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        new_cases, regions, _, device = self.initialize(args)

        params = []
        # exclude = {"nu", "beta.w_feat.weight", "beta.w_feat.bias"}
        exclude = {"nu", "alphas"}
        # exclude = {"nu"}
        for name, p in dict(self.func.named_parameters()).items():
            wd = 0 if name in exclude else args.weight_decay
            if wd != 0:
                print(name, wd)
            params.append({"params": p, "weight_decay": wd})
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            betas=[args.momentum, 0.999],
            weight_decay=args.weight_decay,
        )

        model = train(self.func, new_cases, regions, optimizer, checkpoint, args)

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

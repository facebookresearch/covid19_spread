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


class BetaConst(nn.Module):
    def __init__(self, regions):
        super(BetaConst, self).__init__()
        self.M = len(regions)
        self.b = th.nn.Parameter(th.ones(self.M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        return self.fpos(self.b).expand(self.M, t.size(-1))


class BetaExpDecay(nn.Module):
    def __init__(self, regions):
        super(BetaExpDecay, self).__init__()
        M = len(regions)
        self.a = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.c = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
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
        self.C = th.nn.Parameter(th.ones(M, 1, dtype=th.float))
        self.k = th.nn.Parameter(th.ones(M, 1, dtype=th.float))
        self.m = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
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
        self.a = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.c = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
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
        return beta.squeeze().t()

    def __repr__(self):
        return f"{self.rnn}"


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


class AR(nn.Module):
    def __init__(self, regions, beta_net, dist, window_size, graph, features):
        super(AR, self).__init__()
        self.M = len(regions)
        self.alphas = nn.Parameter(th.zeros((self.M, self.M)).fill_(-5))
        self.repro = nn.Parameter(th.ones((self.M, window_size)))
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(10))
        self.beta = beta_net
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

    def metapopulation_weights(self):
        with th.no_grad():
            self.alphas.fill_diagonal_(-1e10)
        # W = F.softmax(self.alphas, dim=1)
        W = th.sigmoid(self.alphas)
        # W = F.softplus(self.alphas)
        # W = W / W.sum()
        if self.graph is not None:
            W = W * self.graph
        return W

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        offset = self.window - 1
        length = ys.size(1) - self.window + 1

        # self-correlation
        Z = F.conv1d(
            ys.unsqueeze(0), F.softplus(self.repro).unsqueeze(1), groups=self.M
        ).squeeze()
        Z.div_(float(self.window))

        # cross-correlation
        W = self.metapopulation_weights()
        Ys = th.stack([ys.narrow(1, i, length) for i in range(self.window)])
        # Ys = th.bmm(W, Ys).mean(dim=0)
        Ys = th.bmm(W.unsqueeze(0).expand(self.window, self.M, self.M), Ys).mean(dim=0)

        # beta evolution
        ys = ys.narrow(1, offset, ys.size(1) - offset)
        beta = self.beta(t).narrow(1, -ys.size(1), ys.size(1))
        if self.features is not None:
            beta = beta + th.sigmoid(self.w_feat(self.features))
        return beta * Z.add_(Ys)
        # return beta * (Z.addmm_(W, ys))
        # return beta * Z

    def simulate(self, tobs, ys, days, deterministic=True):
        preds = ys.clone()
        offset = self.window + 1
        t = th.arange(offset).to(ys.device) + (tobs + 1 - offset)
        for d in range(days):
            p = preds.narrow(-1, -offset, offset)
            s = self.score(t + d, p).narrow(1, -1, 1)
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
        return f"AR | {self.beta}"


def train(model, cases, regions, optimizer, checkpoint, args):
    M = len(regions)
    device = cases.device
    new_cases = cases[:, 1:] - cases[:, :-1]
    assert (new_cases >= 0).all()
    tmax = new_cases.size(1)
    t = th.arange(tmax).to(device) + 1

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores = model.score(t, new_cases).clamp_(min=1e-8)

        # compute loss
        dist = model.dist(scores.narrow(1, 0, tmax - args.window))
        _loss = -dist.log_prob(
            new_cases.narrow(1, args.window, tmax - args.window)  # .clamp(min=1e-8)
        )
        loss = _loss.sum()
        reg = 0
        # if args.weight_decay > 0:
        #    reg = args.weight_decay * (
        #        th.sigmoid(model.alphas).sum() + F.softplus(model.repro).sum()
        #    )

        assert loss == loss, (loss, scores, _loss)

        # back prop
        (loss + reg).backward()
        optimizer.step()

        # control
        if itr % 50 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                pred = dist.mean[:, -3:]
                gt = new_cases[:, -3:]
                maes = th.abs(gt - pred)
            print(
                f"Iter {itr:04d} | Loss {loss.item() / M:.2f} | Reg {reg:.2f} | MAE {maes.mean():.2f} | {model} | {args.loss}"
            )
            th.save(model.state_dict(), checkpoint)
    print(f"Train MAE,{maes.mean():.2f}")
    return model


def simulate(model, cases, regions, args, dstart=None):
    new_cases = cases[:, 1:] - cases[:, :-1]
    assert (new_cases >= 0).all()
    tmax = new_cases.size(1)

    test_preds = model.simulate(tmax, new_cases, args.test_on)
    test_preds = test_preds.cpu().numpy()

    df = pd.DataFrame(test_preds.T, columns=regions)
    if dstart is not None:
        base = pd.to_datetime(dstart)
        ds = [base + timedelta(i) for i in range(1, args.test_on + 1)]
        df["date"] = ds

        df.set_index("date", inplace=True)

    # print(model.beta(th.arange(tmax + args.test_on).to(cases.device) + 1))
    return df


def prediction_interval(model, cases, regions, nsamples, args, dstart=None):
    new_cases = cases[:, 1:] - cases[:, :-1]
    assert (new_cases >= 0).all()
    tmax = new_cases.size(1)
    samples = []

    for i in range(nsamples):
        test_preds = model.simulate(tmax, new_cases, args.test_on, False)
        test_preds = test_preds.cpu().numpy()
        test_preds = np.cumsum(test_preds, axis=1)
        test_preds = test_preds + cases.narrow(1, -1, 1).cpu().numpy()
        samples.append(test_preds)
    samples = np.stack(samples, axis=0)
    print(samples.shape)
    test_preds_mean = np.mean(samples, axis=0)
    test_preds_std = np.std(samples, axis=0)
    df_mean = pd.DataFrame(test_preds_mean.T, columns=regions)
    df_std = pd.DataFrame(test_preds_std.T, columns=regions)
    if dstart is not None:
        base = pd.to_datetime(dstart)
        ds = [base + timedelta(i) for i in range(1, args.test_on + 1)]
        df_mean["date"] = ds
        df_std["date"] = ds

        df_mean.set_index("date", inplace=True)
        df_std.set_index("date", inplace=True)
    return df_mean, df_std


def initialize(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    cases, regions, basedate = load.load_confirmed_csv(args.fdat)
    cases = cases.float().to(device)[:, args.t0 :]
    print("Timeseries length", cases.size(1))

    # cheat to test compatibility
    # if zero (def value) it'll continue
    if not hasattr(args, "keep_counties"):
        pass
    elif args.keep_counties == -1:
        cases = cases.sum(0).reshape(1, -1)
        regions = ["all"]
    elif args.keep_counties > 0:
        k = args.keep_counties
        # can also sort on case numbers and pick top-k
        cases = cases[:k]
        regions = regions[:k]

    return cases, regions, basedate, device


def _get_arg(args, v, device):
    if hasattr(args, v):
        return th.load(getattr(args, v)).to(device).float()
    else:
        return None


class ARCV(cv.CV):
    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        cases, regions, _, device = initialize(args)
        tmax = cases.size(1) + 1

        # setup optional features
        graph = _get_arg(args, "graph", device)
        features = _get_arg(args, "features", device)
        time_features = _get_arg(args, "time_features", device)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            time_features = time_features.narrow(0, args.t0, cases.size(1))
            print(time_features.size(), cases.size())

        weight_decay = 0
        # setup beta function
        if args.decay == "const":
            beta_net = BetaConst(regions)
        elif args.decay == "exp":
            beta_net = BetaExpDecay(regions)
        elif args.decay == "logistic":
            beta_net = BetaLogistic(regions)
        elif args.decay == "powerlaw":
            beta_net = BetaPowerLawDecay(regions)
        elif args.decay.startswith("poly"):
            degree = int(args.decay[4:])
            beta_net = BetaPolynomial(regions, degree, tmax)
        elif args.decay.startswith("rbf"):
            dim = int(args.decay[3:])
            beta_net = BetaRBF(regions, dim, "gaussian", tmax)
        elif args.decay.startswith("latent"):
            dim, layers = args.decay[6:].split("_")
            beta_net = BetaLatent(regions, int(dim), int(layers), tmax, time_features)
            weight_decay = args.weight_decay
        else:
            raise ValueError("Unknown beta function")

        func = AR(regions, beta_net, args.loss, args.window, graph, features).to(device)
        params = []
        # exclude = {"nu", "beta.w_feat.weight", "beta.w_feat.bias"}
        # exclude = {"nu", "alphas", "repro"}
        exclude = {"nu"}
        for name, p in dict(func.named_parameters()).items():
            wd = 0 if name in exclude else weight_decay
            print(name, wd)
            params.append({"params": p, "weight_decay": wd})
        optimizer = optim.AdamW(
            params, lr=args.lr, betas=[args.momentum, 0.999], weight_decay=weight_decay
        )

        model = train(func, cases, regions, optimizer, checkpoint, args)

        return model

    def run_simulate(self, dset, args, model=None, sim_params=None):
        args.fdat = dset
        if model is None:
            raise NotImplementedError

        cases, regions, basedate, device = initialize(args)
        forecast = simulate(model, cases, regions, args, basedate)

        return forecast

    def run_prediction_interval(self, args, nsamples, model=None):
        if model is None:
            raise NotImplementedError

        cases, regions, basedate, device = initialize(args)
        df_mean, df_std = prediction_interval(
            model, cases, regions, nsamples, args, basedate
        )
        return df_mean, df_std


CV_CLS = ARCV


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

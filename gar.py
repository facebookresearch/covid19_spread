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


class GAR(nn.Module):
    def __init__(self, regions, beta_net, dist, window_size, graph, features):
        super(GAR, self).__init__()
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

        # cross-correlation
        W = self.metapopulation_weights()

        Ys = th.nn.utils.rnn.pad_sequence(
            [ys.narrow(1, i, ys.size(1) - i).t() for i in range(self.window)]
        )
        Ys = Ys.permute(1, 2, 0)
        # print(ys.size(), Ys.size())
        # Ys = th.bmm(W, Ys).mean(dim=0)
        Ys = th.bmm(W.unsqueeze(0).expand(self.window, self.M, self.M), Ys).mean(dim=0)
        # Ys = th.mm(W, ys)

        Z = F.conv1d(
            Ys.unsqueeze(0), F.softplus(self.repro).unsqueeze(1), groups=self.M
        )
        Z = Z.squeeze(0)
        Z.div_(float(self.window))
        # beta evolution
        ys = ys.narrow(1, offset, ys.size(1) - offset)
        beta = self.beta(t).narrow(1, -ys.size(1), ys.size(1))
        if self.features is not None:
            beta = beta + th.sigmoid(self.w_feat(self.features))
        Ys = beta * Z
        # Ys = beta * Ys

        assert Ys.size(-1) == t.size(-1) - offset, (Ys.size(-1), t.size(-1), offset)
        return Ys
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
    print(args)
    M = len(regions)
    device = cases.device
    new_cases = cases[:, 1:] - cases[:, :-1]
    print(f"max count = {cases.max()}, max inc = {new_cases.max()}")
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
                    f"[{itr:04d}] Loss {loss.item() / M:.2f} | "
                    f"MAE {maes.mean():.2f} | "
                    f"{model} | "
                    f"{args.loss} ({scores[:, -1].min().item():.2f}, {scores[:, -1].max().item():.2f}) | "
                    f"alpha ({_a.min().item():.2f}, {_a.mean().item():.2f}, {_a.max().item():.2f})"
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

    args.window = min(args.window, cases.size(1) - 4)

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


class GARCV(cv.CV):
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
        elif args.decay.startswith("latent"):
            dim, layers = args.decay[6:].split("_")
            beta_net = BetaLatent(regions, int(dim), int(layers), tmax, time_features)
            weight_decay = args.weight_decay
        else:
            raise ValueError("Unknown beta function")

        func = GAR(regions, beta_net, args.loss, args.window, graph, features).to(
            device
        )
        params = []
        # exclude = {"nu", "beta.w_feat.weight", "beta.w_feat.bias"}
        # exclude = {"nu", "alphas"}
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

    def run_prediction_interval(self, dset, args, nsamples, model=None):
        args.fdat = dset
        if model is None:
            raise NotImplementedError

        cases, regions, basedate, device = initialize(args)
        df_mean, df_std = prediction_interval(
            model, cases, regions, nsamples, args, basedate
        )
        return df_mean, df_std


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

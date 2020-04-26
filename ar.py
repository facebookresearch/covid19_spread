import argparse
import numpy as np
import pandas as pd
from datetime import timedelta

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import NegativeBinomial, Poisson

from load import load_data


def load_confirmed(path):
    nodes, ns, ts, basedate, = load_data(path)
    nodes = np.array(nodes)
    tmax = int(np.ceil(ts.max()))
    cases = np.zeros((len(nodes), tmax))
    for n in range(len(nodes)):
        ix2 = np.where(ns == n)[0]
        for i in range(1, tmax + 1):
            ix1 = np.where(ts < i)[0]
            cases[n, i - 1] = len(np.intersect1d(ix1, ix2))
    unk = np.where(nodes == "Unknown")[0]
    cases = np.delete(cases, unk, axis=0)
    nodes = np.delete(nodes, unk)
    return th.from_numpy(cases), nodes, basedate


def load_population(path, nodes, col=1):
    df = pd.read_csv(path, header=None)
    _pop = df.iloc[:, col].to_numpy()
    regions = df.iloc[:, 0].to_numpy()
    pop = []
    for node in nodes:
        if node == "Unknown":
            continue
        ix = np.where(regions == node)[0][0]
        pop.append(_pop[ix])
    print(nodes, pop)
    return th.from_numpy(np.array(pop))


class BetaConst(nn.Module):
    def __init__(self, population):
        super(BetaConst, self).__init__()
        M = len(population)
        self.b = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        return self.fpos(self.b)


class BetaExpDecay(nn.Module):
    def __init__(self, population):
        super(BetaExpDecay, self).__init__()
        M = len(population)
        # self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        # self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.a = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        # beta = self.fpos(self.a) * th.exp(-self.fpos(self.b) * t) * self.fpos(self.c)
        t = t.unsqueeze(0)
        beta = self.fpos(self.a) * th.exp(-self.fpos(self.b) * t)
        return beta

    def __repr__(self):
        with th.no_grad():
            return f"Exp = ({self.fpos(self.a).mean().item():.3f}, {self.fpos(self.b).mean().item():.3f})"


class BetaLogistic(nn.Module):
    def __init__(self, population):
        super(BetaLogistic, self).__init__()
        M = len(population)
        self.C = th.nn.Parameter(th.ones(M, 1, dtype=th.float))
        self.k = th.nn.Parameter(th.ones(M, 1, dtype=th.float))
        self.m = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        return self.fpos(self.C) / (
            1 + th.exp(self.fpos(self.k) * (t - self.fpos(self.m)))
        )


class BetaPowerLawDecay(nn.Module):
    def __init__(self, population):
        super(BetaPowerLawDecay, self).__init__()
        M = len(population)
        # self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        # self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.a = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.c = th.nn.Parameter(th.ones(M, 1, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t):
        t = t.unsqueeze(0).float()
        # beta = self.fpos(self.a) * th.exp(
        #    -self.fpos(self.b) * th.log(t) - self.fpos(self.c)
        # )
        a = self.fpos(self.a)
        m = self.fpos(self.b)
        #  beta = a * m ** a / t ** (a + 1) + self.fpos(self.c)  # pareto
        beta = a * t ** -m + self.fpos(self.c)
        return beta

    def __repr__(self):
        with th.no_grad():
            return f"Power law = ({self.fpos(self.a).mean().item():.3f}, {self.fpos(self.b).mean().item():.3f})"


class BetaLatent(nn.Module):
    def __init__(self, population, dim, tmax):
        super(BetaLatent, self).__init__()
        self.M = len(population)
        self.dim = dim
        self.tmax = tmax
        self.Wbeta = nn.Linear(dim, dim, bias=True)
        self.Wbeta2 = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)
        self.b0 = nn.Parameter(th.tanh(th.randn(dim, dtype=th.float)))
        self.c = nn.Parameter(th.randn(1, 1, dtype=th.float))
        self.fpos = F.softplus
        nn.init.xavier_uniform_(self.Wbeta.weight)
        nn.init.xavier_uniform_(self.Wbeta2.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, t, y):
        # beta_last = y.narrow(0, self.M * 3, self.M * self.dim).reshape(self.M, self.dim)
        beta_last = y.narrow(0, self.M * 3, self.dim)
        # beta_last = y.narrow(0, self.M * 3, self.dim)
        # beta_now = self.Wbeta2(th.tanh(self.Wbeta(beta_last)))
        # beta_now = self.Wbeta(beta_last)

        beta_now = th.tanh(self.Wbeta(beta_last) + self.c * t / self.tmax)
        # a = tmp.narrow(-1, 0, 1)
        # b = tmp.narrow(-1, 1, 1)
        # c = tmp.narrow(-1, 2, 1)
        # beta = th.sigmoid(a) * th.exp(-self.fpos(b) * t)
        # beta = th.sigmoid(a * t + b * (t - 1) + c * (t - 2))
        # beta = th.sigmoid(tmp.mean()) * F.softplus(self.c)
        beta = th.sigmoid(self.v(beta_now))
        # beta = self.fpos(a) * th.exp(-self.fpos(b) * t) + self.fpos(c)
        # assert beta == beta, (beta_last, beta_now, self.
        return beta.squeeze(), beta_now.view(-1)


class BetaRBF(nn.Module):
    def __init__(self, population, dim):
        super(BetaRBF, self).__init__()
        self.M = len(population)
        self.dim = dim
        # self.bs = nn.Parameter(th.ones(self.M, dim, dtype=th.float).fill_(-4))
        # self.bs = nn.Parameter(th.randn(self.M, dim, dtype=th.float))
        self.v = nn.Parameter(th.randn(self.M, dim, dtype=th.float))
        self.c = nn.Parameter(th.randn(self.M, dim, dtype=th.float))
        self.temp = nn.Parameter(th.ones(self.M, 1, dtype=th.float))
        self.fpos = F.softplus

    def forward(self, t, y):
        d = (t - self.c) ** 2  # / self.fpos(self.temp)
        beta = th.sigmoid(th.sum(self.v * th.exp(-d), dim=1))
        # beta = self.fpos(self.bs.narrow(-1, int(t), 1))
        return beta.squeeze()


class AR(nn.Module):
    def __init__(self, population, beta_net, dist, window_size):
        super(AR, self).__init__()
        self.M = len(population)
        self.alphas = th.nn.Parameter(th.zeros((self.M, self.M)))
        self.repro = th.nn.Parameter(th.ones((self.M, window_size)))
        self.beta = beta_net
        self.nu = th.nn.Parameter(th.ones((self.M, 1)))
        self.fpos = F.softplus
        self._dist = dist
        self.window = window_size

    def dist(self, scores):
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu)
        else:
            raise RuntimeError(f"Unknown loss")

    def metapopulation_weights(self):
        with th.no_grad():
            self.alphas.fill_diagonal_(-1e10)
        # W = F.softmax(self.alphas, dim=1)
        W = th.sigmoid(self.alphas)
        # W = W / W.sum()
        return W

    def score(self, t, ys):
        W = self.metapopulation_weights()
        # Z = th.cat(
        #     [
        #         F.conv1d(
        #             ys.narrow(0, i, 1).unsqueeze(0),
        #             self.fpos(self.repro.narrow(0, i, 1)).unsqueeze(0),
        #         )
        #         for i in range(self.M)
        #     ],
        #     dim=0,
        # ).squeeze()
        Z = F.conv1d(ys.unsqueeze(0), self.repro.unsqueeze(1), groups=self.M).squeeze()
        offset = self.window - 1
        ys = ys.narrow(1, offset, ys.size(1) - offset)
        beta = self.beta(t).narrow(1, offset, ys.size(1))
        return beta * (self.fpos(Z) + th.mm(W, ys))  # + self.fpos(self.nu)

    def simulate(self, ys, days):
        t = th.arange(ys.size(1)).to(ys.device) + 1
        preds = [ys]
        for d in range(days):
            cases = preds[-1]
            s = self.score(t + d, cases)
            y = self.dist(s).mean
            preds.append(y)
        preds = [p.narrow(1, -1, 1) for p in preds]
        return th.cat(preds, dim=1)


def train(model, cases, population, optimizer, checkpoint, args):
    M = len(population)
    device = cases.device
    new_cases = cases[:, 1:] - cases[:, :-1]
    tmax = new_cases.size(1)
    t = th.arange(tmax).to(device) + 1

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores = model.score(t, new_cases)
        # print(new_cases.size(), scores.size())
        # print(scores)

        # compute loss
        dist = model.dist(scores.narrow(1, 0, tmax - args.window))
        loss = -dist.log_prob(
            new_cases.narrow(1, args.window, tmax - args.window)
        ).mean()

        # back prop
        loss.backward()
        optimizer.step()

        # control
        if itr % 50 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                pred = dist.mean[:, -3:]
                gt = new_cases[:, -3:]
                maes = th.abs(gt - pred)
                # print(th.cat([gt, pred, maes], dim=1).cpu().numpy())
            print(
                f"Iter {itr:04d} | Loss {loss.item() / M:.2f} | MAE {maes.mean()} | {model}"
            )
            th.save(model.state_dict(), checkpoint)
    return model


def simulate(model, cases, regions, population, args, dstart=None):
    M = len(population)
    device = cases.device
    tmax = cases.size(1)
    t = th.arange(tmax).float().to(device) + 1
    new_cases = cases[:, 1:] - cases[:, :-1]

    test_preds = model.simulate(new_cases, args.test_on)[:, 1:]
    test_preds = test_preds.cpu().numpy()
    test_preds = np.cumsum(test_preds, axis=1)
    test_preds = test_preds + cases.narrow(1, -1, 1).cpu().numpy()

    df = pd.DataFrame(test_preds.T)
    df.columns = regions
    if dstart is not None:
        base = pd.to_datetime(dstart)
        ds = [base + timedelta(i) for i in range(1, args.test_on + 1)]
        df["date"] = ds
        df.set_index("date", inplace=True)
    return df


def initialize(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    cases, regions, basedate = load_confirmed(args.fdat)
    population = load_population(args.fpop, regions)
    cases = cases.float().to(device)[:, args.t0 :]
    population = population.float().to(device)

    return cases, regions, population, basedate, device


def run_train(args, checkpoint):
    cases, regions, population, _, device = initialize(args)
    tmax = np.max([len(ts) for ts in cases])

    weight_decay = 0
    if args.decay == "const":
        beta_net = BetaConst(population)
    elif args.decay == "exp":
        beta_net = BetaExpDecay(population)
    elif args.decay == "logistic":
        beta_net = BetaLogistic(population)
    elif args.decay == "powerlaw":
        beta_net = BetaPowerLawDecay(population)
    elif args.decay == "rbf":
        beta_net = BetaRBF(population, 10)
    elif args.decay == "latent":
        beta_net = BetaLatent(population, 16, float(len(cases)))
        weight_decay = args.weight_decay

    func = AR(population, beta_net, args.loss, args.window).to(device)
    optimizer = optim.AdamW(
        func.parameters(), lr=args.lr, betas=[0.99, 0.999], weight_decay=weight_decay
    )

    model = train(func, cases, population, optimizer, checkpoint, args)

    return model


def run_simulate(args, model=None):
    if model is None:
        raise NotImplementedError

    cases, regions, population, basedate, device = initialize(args)

    forecast = simulate(model, cases, regions, population, args, basedate)

    adj = model.metapopulation_weights().cpu().numpy()
    df = pd.DataFrame(adj).round(2)
    df.columns = regions
    df["regions"] = regions
    df.set_index("regions", inplace=True)
    print(df)

    return forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ODE demo")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument("-lr", type=float, default=5e-2)
    parser.add_argument("-weight-decay", type=float, default=0)
    parser.add_argument("-niters", type=int, default=2000)
    parser.add_argument("-adjoint", default=False, action="store_true")
    parser.add_argument("-amsgrad", default=False, action="store_true")
    parser.add_argument("-method", default="euler", choices=["euler", "rk4", "dopri5"])
    parser.add_argument("-loss", default="poisson", choices=["nb", "poisson"])
    parser.add_argument("-decay", default="exp", choices=["exp", "powerlaw", "latent"])
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=5, type=int)
    parser.add_argument("-test-on", default=5, type=int)
    parser.add_argument("-window", default=5, type=int)
    parser.add_argument("-checkpoint", type=str, default="/tmp/metasir_model.bin")
    args = parser.parse_args()

    model = run_train(args, args.checkpoint)

    with th.no_grad():
        forecast = run_simulate(args, model)

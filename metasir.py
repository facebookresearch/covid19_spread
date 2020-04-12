import argparse
import numpy as np
import pandas as pd

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import load_data


def load_confirmed(path, regions):
    # df = pd.read_csv(path, usecols=regions)
    # cases = df.to_numpy().sum(axis=1)
    nodes, ns, ts, _ = load_data(path)
    regions = np.array(regions)
    unk = np.where(nodes == "Unknown")[0]
    if len(unk) > 0:
        ix = np.where(ns != unk[0])
        ts = ts[ix]
    tmax = int(np.ceil(ts.max()))
    cases = np.zeros((len(nodes) - len(unk), tmax))
    for n in range(len(nodes)):
        if nodes[n] == "Unknown":
            continue
        ix2 = np.where(ns == n)[0]
        j = np.where(regions == nodes[n])[0][0]
        for i in range(1, tmax + 1):
            ix1 = np.where(ts < i)[0]
            cases[j, i - 1] = len(np.intersect1d(ix1, ix2))
    # print(cases)
    return th.from_numpy(cases)


def load_population(path, col=1):
    df = pd.read_csv(path, header=None)
    pop = df.iloc[:, col].to_numpy()
    regions = df.iloc[:, 0].to_numpy().tolist()
    print(regions)
    return th.from_numpy(pop), regions


class BetaExpDecay(nn.Module):
    def __init__(self, population):
        super(BetaExpDecay, self).__init__()
        M = len(population)
        # self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        # self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.a = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.c = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t, y):
        beta = self.fpos(self.a) * th.exp(-self.fpos(self.b) * t * self.fpos(self.c))
        return beta, None

    def __repr__(self):
        with th.no_grad():
            return f"Exp = ({self.fpos(self.a).mean().item():.3f}, {self.fpos(self.b).mean().item():.3f})"

    def y0(self):
        return None


class BetaPowerLawDecay(nn.Module):
    def __init__(self, population):
        super(BetaPowerLawDecay, self).__init__()
        M = len(population)
        # self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        # self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.a = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.c = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t, y):
        # beta = self.fpos(self.a) * th.exp(
        #    -self.fpos(self.b) * th.log(t) - self.fpos(self.c)
        # )
        a = self.fpos(self.a)
        m = self.fpos(self.b)
        beta = a * m ** a / t ** (a + 1) + self.fpos(self.c)
        return beta, None

    def __repr__(self):
        with th.no_grad():
            return f"Power law = ({self.fpos(self.a).mean().item():.3f}, {self.fpos(self.b).mean().item():.3f})"

    def y0(self):
        return None


class BetaLatent(nn.Module):
    def __init__(self, population, dim):
        super(BetaLatent, self).__init__()
        self.M = len(population)
        self.dim = dim
        self.Wbeta = nn.Linear(dim, dim, bias=True)
        self.Wbeta2 = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 3, bias=True)
        self.b0 = nn.Parameter(th.ones(dim, dtype=th.float))
        self.fpos = F.softplus
        nn.init.xavier_uniform_(self.Wbeta.weight)
        nn.init.xavier_uniform_(self.Wbeta2.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, t, y):
        beta_last = y.narrow(0, self.M * 2, self.dim)
        beta_now = self.Wbeta2(th.tanh(self.Wbeta(beta_last)))

        tmp = self.v(th.tanh(beta_now))
        a = tmp.narrow(0, 0, 1)
        b = tmp.narrow(0, 1, 1)
        c = tmp.narrow(0, 2, 1)
        beta = self.fpos(a) * th.exp(-self.fpos(b) * t - self.fpos(c))
        # beta = th.sigmoid(tmp)
        assert beta == beta, (beta_last, beta_now, self.Wbeta.weight)
        return beta, beta_now

    def y0(self):
        return self.b0


class MetaSIR(nn.Module):
    def __init__(self, population, beta_net):
        super(MetaSIR, self).__init__()
        self.M = len(population)
        self.alphas = th.nn.Parameter(th.ones((self.M, self.M), dtype=th.float))
        self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-5))
        # self.gamma = th.tensor([1.0 / 14]).to(device)
        # self.c = th.nn.Parameter(th.zeros(1, dtype=th.float))
        self.fpos = F.softplus
        self.Ns = population
        self.beta_net = beta_net

    def y0(self, S, I):
        elems = list(filter(None.__ne__, [S.view(-1), I.view(-1), self.beta_net.y0()]))
        return th.cat(elems, dim=0).to(device).float()

    def forward(self, t, y):
        # prepare input
        Ss = y.narrow(0, 0, self.M)
        Is = y.narrow(0, self.M, self.M)
        beta, dBeta = self.beta_net(t, y)
        assert beta.ndim <= 1, beta.size()

        # compute dynamics
        W = F.softmax(self.alphas, dim=0)
        WIs = th.mv(W, Is) / self.Ns
        dSs = -beta * Ss * WIs  # / th.sigmoid(self.c)
        dIs = beta * Ss * WIs - self.fpos(self.gamma) * Is

        # prepare dy output
        elems = list(filter(None.__ne__, [dSs, dIs, dBeta]))
        dy = th.cat(elems, dim=0)
        return dy

    def __repr__(self):
        with th.no_grad():
            return f" gamma {self.fpos(self.gamma).item():.3f} | {self.beta_net}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ODE demo")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument(
        "--method", type=str, choices=["dopri5", "adams"], default="dopri5"
    )
    parser.add_argument("-lr", type=float, default=5e-2)
    parser.add_argument("-weight-decay", type=float, default=0)
    parser.add_argument("-niters", type=int, default=2000)
    parser.add_argument("-adjoint", default=False, action="store_true")
    parser.add_argument("-amsgrad", default=False, action="store_true")
    parser.add_argument("-method", default="euler", choices=["euler", "rk4", "dopri5"])
    parser.add_argument("-loss", default="lsq", choices=["lsq", "poisson"])
    parser.add_argument("-decay", default="exp", choices=["exp", "powerlaw", "latent"])
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=5, type=int)
    parser.add_argument("-test-on", default=5, type=int)
    parser.add_argument("-checkpoint", type=str, default="/tmp/metasir_model.bin")
    args = parser.parse_args()

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # load data
    population, regions = load_population(args.fpop)
    cases = load_confirmed(args.fdat, regions)
    test_cases = cases[:, -args.test_on :]
    cases = cases.float().to(device)[:, args.t0 : -args.test_on]
    population = population.float().to(device)
    tmax = cases.size(1)
    t = th.arange(tmax).float().to(device) + 1

    I0 = cases.narrow(1, 0, 1)
    S0 = population.unsqueeze(1) - I0
    M = len(population)

    if args.decay == "exp":
        beta_net = BetaExpDecay(population)
    elif args.decay == "powerlaw":
        beta_net = BetaPowerLawDecay(population)
    elif args.decay == "latent":
        beta_net = BetaLatent(population, 10)
    func = MetaSIR(population, beta_net).to(device)
    y0 = func.y0(S0, I0)

    optimizer = optim.AdamW(
        func.parameters(),
        lr=args.lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
    )
    nsteps = args.niters

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, y0, t, method=args.method, options={"step_size": 1})
        pred_Is = pred_y.narrow(1, M, M).squeeze().t()

        predicted = pred_Is[:, -args.fit_on :]
        observed = cases[:, -args.fit_on :]

        if args.loss == "lsq":
            loss = th.sum((predicted - observed) ** 2)
        elif args.loss == "poisson":
            loss = th.sum(predicted - observed * th.log(predicted))
        else:
            raise RuntimeError(f"Unknown loss")
        loss.backward()
        optimizer.step()
        if itr % 50 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                maes = th.abs(cases[:, -3:] - pred_Is[:, -3:])
                print(
                    th.cat([cases[:, -3:], pred_Is[:, -3:], maes], dim=1)
                    .cpu()
                    .numpy()
                    .round(2)
                )
            print(
                f"Iter {itr:04d} | Loss {loss.item() / M:.2f} | MAE {maes[:, -1].mean():.2f} | {func}"
            )
            th.save(func.state_dict(), args.checkpoint)

    with th.no_grad():
        print(F.softmax(func.alphas, dim=1).cpu().numpy().round(2))

    It = cases.narrow(1, -1, 1)
    St = population.unsqueeze(1) - I0
    yt = func.y0(St, It)
    test_preds = odeint(
        func,
        yt,
        th.arange(tmax + 1, tmax + 7).float(),
        method="euler",
        options={"step_size": 1},
    )
    # print(test_preds)
    test_preds = test_preds.narrow(1, M, M).squeeze().t().narrow(1, -5, 5)
    print(test_cases)
    print(test_preds)
    print(th.abs(test_cases.to(device) - test_preds).mean(dim=0))

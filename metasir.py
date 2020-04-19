import argparse
import numpy as np
import pandas as pd
from datetime import timedelta

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import load_data


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


class BetaExpDecay(nn.Module):
    def __init__(self, population):
        super(BetaExpDecay, self).__init__()
        M = len(population)
        # self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        # self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.c = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t, y):
        # beta = self.fpos(self.a) * th.exp(-self.fpos(self.b) * t) * self.fpos(self.c)
        beta = self.fpos(self.a) * th.exp(-self.fpos(self.b) * t)
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
        self.v = nn.Linear(dim, 3, bias=False)
        self.c = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.b0 = nn.Parameter(th.ones(dim, dtype=th.float))
        self.fpos = F.softplus
        # nn.init.xavier_uniform_(self.Wbeta.weight)
        # nn.init.xavier_uniform_(self.Wbeta2.weight)
        # nn.init.xavier_uniform_(self.v.weight)

    def forward(self, t, y):
        # beta_last = y.narrow(0, self.M * 3, self.M * self.dim).reshape(self.M, self.dim)
        beta_last = y.narrow(0, self.M * 3, self.dim)
        # beta_now = self.Wbeta2(th.tanh(self.Wbeta(beta_last)))
        beta_now = self.Wbeta(beta_last)

        tmp = self.v(th.sigmoid(beta_now))
        a = tmp.narrow(-1, 0, 1)
        b = tmp.narrow(-1, 1, 1)
        c = tmp.narrow(-1, 2, 1)
        beta = self.fpos(a) * th.exp(-self.fpos(b) * t - self.fpos(c))
        # beta = th.sigmoid(tmp.mean()) * F.softplus(self.c)
        # beta = th.exp(tmp.mean())
        # beta = self.fpos(a) * th.exp(-self.fpos(b) * t) + self.fpos(c)
        # assert beta == beta, (beta_last, beta_now, self.Wbeta.weight)
        return beta.squeeze(), beta_now.view(-1)

    def y0(self):
        return self.b0.view(-1)


class MetaSIR(nn.Module):
    def __init__(self, population, beta_net):
        super(MetaSIR, self).__init__()
        self.M = len(population)
        self.alphas = th.nn.Parameter(th.ones((self.M, self.M)))
        self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(0.1))
        # self.gamma = th.tensor([1.0 / 14]).to(device)
        self.c = th.nn.Parameter(th.zeros(1, dtype=th.float))
        self.fpos = F.softplus
        self.Ns = population
        self.beta_net = beta_net

    def y0(self, S, I, R):
        elems = list(
            filter(
                None.__ne__, [S.view(-1), I.view(-1), R.view(-1), self.beta_net.y0()]
            )
        )
        return th.cat(elems, dim=0).float()

    def forward(self, t, y):
        # prepare input
        Ss = y.narrow(0, 0, self.M)
        Is = y.narrow(0, self.M, self.M)
        Rs = y.narrow(0, 2 * self.M, self.M)
        beta, dBeta = self.beta_net(t, y)
        assert beta.ndim <= 1, beta.size()

        # compute dynamics
        W = F.softmax(self.alphas, dim=0)
        # W = th.sigmoid(self.alphas)
        # W = W / W.sum()
        WIs = beta * th.mv(W, Is) / self.Ns
        bSIs = beta * Ss * Is / self.Ns
        gIs = self.fpos(self.gamma) * Is
        dSs = -bSIs - WIs
        dIs = bSIs + WIs - gIs
        dRs = gIs
        # dSs = -beta * Ss * WIs  # / th.sigmoid(self.c)
        # dIs = beta * Ss * WIs - self.fpos(self.gamma) * Is

        # prepare dy output
        elems = list(filter(None.__ne__, [dSs, dIs, dRs, dBeta]))
        dy = th.cat(elems, dim=0)
        return dy

    def __repr__(self):
        with th.no_grad():
            return f"SIR | gamma {self.fpos(self.gamma).item():.3f} | {self.beta_net}"


class MetaSEIR(nn.Module):
    def __init__(self, population, beta_net):
        super(MetaSEIR, self).__init__()
        self.M = len(population)
        self.alphas = th.nn.Parameter(th.ones((self.M, self.M), dtype=th.float))
        self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-5))
        self.sigma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-5))
        # self.gamma = th.tensor([1.0 / 14]).to(device)
        # self.c = th.nn.Parameter(th.zeros(1, dtype=th.float))
        self.fpos = F.softplus
        self.Ns = population
        self.beta_net = beta_net

    def y0(self, S, I, E=None):
        if E is None:
            E = th.ones_like(I)
        elems = list(
            filter(
                None.__ne__, [S.view(-1), I.view(-1), E.view(-1), self.beta_net.y0()]
            )
        )
        return th.cat(elems, dim=0).to(device).float()

    def forward(self, t, y):
        # prepare input
        Ss = y.narrow(0, 0, self.M)
        Is = y.narrow(0, self.M, self.M)
        Es = y.narrow(0, 2 * self.M, self.M)
        beta, dBeta = self.beta_net(t, y)
        sigma = self.fpos(self.sigma)
        assert beta.ndim <= 1, beta.size()

        # compute dynamics
        W = th.sigmoid(self.alphas)
        W = W / W.sum()
        WIs = beta * th.mv(W, Is) / self.Ns
        bSIs = beta * Ss * Is / self.Ns
        dSs = -bSIs - WIs
        dEs = bSIs + WIs - sigma * Es
        dIs = sigma * Es - self.fpos(self.gamma) * Is

        # prepare dy output
        elems = list(filter(None.__ne__, [dSs, dIs, dEs, dBeta]))
        dy = th.cat(elems, dim=0)
        return dy

    def __repr__(self):
        with th.no_grad():
            return f"SEIR | gamma {self.fpos(self.gamma).item():.3f} | {self.beta_net}"


def train(model, cases, population, odeint, optimizer, checkpoint, args):
    M = len(population)
    device = cases.device
    tmax = cases.size(1)
    t = th.arange(tmax).float().to(device) + 1

    I0 = cases.narrow(1, 0, 1)
    S0 = population.unsqueeze(1) - I0
    R0 = th.zeros_like(I0)
    y0 = model.y0(S0, I0, R0).to(device)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(model, y0, t, method=args.method, options={"step_size": 1})
        # pred_y = odeint(func, y0, t, method=args.method)
        pred_Is = pred_y.narrow(1, M, M).squeeze().t()
        pred_Rs = pred_y.narrow(1, 2 * M, M).squeeze().t()
        pred_Cs = pred_Is + pred_Rs

        predicted = pred_Cs[:, -args.fit_on :]
        observed = cases[:, -args.fit_on :]

        # compute loss
        if args.loss == "lsq":
            loss = th.sum((predicted - observed) ** 2)
        elif args.loss == "poisson":
            loss = th.sum(predicted - observed * th.log(predicted))
        else:
            raise RuntimeError(f"Unknown loss")

        # back prop
        loss.backward()
        optimizer.step()

        # control
        if itr % 50 == 0:
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                maes = th.abs(cases[:, -3:] - pred_Cs[:, -3:])
                print(
                    th.cat([cases[:, -3:], pred_Cs[:, -3:], maes], dim=1)
                    .cpu()
                    .numpy()
                    .round(2)
                )
            print(
                f"Iter {itr:04d} | Loss {loss.item() / M:.2f} | MAE {maes[:, -1].mean():.2f} | {model}"
            )
            th.save(model.state_dict(), checkpoint)
    return model


def simulate(model, cases, regions, population, odeint, args, dstart=None):
    M = len(population)
    device = cases.device
    tmax = cases.size(1)
    t = th.arange(tmax).float().to(device) + 1

    I0 = cases.narrow(1, 0, 1)
    S0 = population.unsqueeze(1) - I0
    R0 = th.zeros_like(I0)
    y0 = model.y0(S0, I0, R0).to(device)

    pred_y = odeint(model, y0, t, method=args.method, options={"step_size": 1})
    pred_Rs = pred_y.narrow(1, 2 * M, M).squeeze().t()

    Rt = pred_Rs.narrow(1, -1, 1)
    It = cases.narrow(1, -1, 1) - Rt
    St = population.unsqueeze(1) - It - Rt
    print(It.size(), Rt.size(), St.size())
    # Et = pred_y.narrow(1, 2 * M, M).squeeze().t()[:, -1]
    # yt = func.y0(St, It, Et)
    yt = model.y0(St, It, Rt)
    test_preds = odeint(
        model,
        yt,
        th.arange(tmax + 1, tmax + args.test_on + 2).float(),
        method="euler",
        options={"step_size": 1},
    )

    test_preds = test_preds.narrow(1, M, M).squeeze().t().narrow(
        1, -args.test_on, args.test_on
    ) + test_preds.narrow(1, 2 * M, M).squeeze().t().narrow(
        1, -args.test_on, args.test_on
    )
    df = pd.DataFrame(test_preds.cpu().numpy().T)
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

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    return cases, regions, population, basedate, odeint, device


def run_train(args, checkpoint):
    cases, regions, population, _, odeint, device = initialize(args)

    if args.decay == "exp":
        beta_net = BetaExpDecay(population)
    elif args.decay == "powerlaw":
        beta_net = BetaPowerLawDecay(population)
    elif args.decay == "latent":
        beta_net = BetaLatent(population, 64)

    func = MetaSIR(population, beta_net).to(device)
    optimizer = optim.AdamW(
        func.parameters(),
        lr=args.lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
    )

    model = train(func, cases, population, odeint, optimizer, args)

    return model


def run_simulate(args, model=None):
    if model is None:
        raise NotImplementedError

    cases, regions, population, basedate, odeint, device = initialize(args)

    forecast = simulate(model, cases, regions, population, odeint, args, basedate)

    adj = th.sigmoid(model.alphas).cpu().numpy()
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
    parser.add_argument("-loss", default="lsq", choices=["lsq", "poisson"])
    parser.add_argument("-decay", default="exp", choices=["exp", "powerlaw", "latent"])
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=5, type=int)
    parser.add_argument("-test-on", default=5, type=int)
    parser.add_argument("-checkpoint", type=str, default="/tmp/metasir_model.bin")
    args = parser.parse_args()

    # load data
    # test_cases = cases[:, -args.test_on :]

    model = run_train(args, args.checkpoint)

    with th.no_grad():
        forecast = run_simulate(args, model)

    # print(test_cases)
    # print(forecast)
    # print(np.abs(test_cases.numpy().T - forecast).mean(axis=1))

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
    unk = np.where(nodes == "Unknown")[0]
    if len(unk) > 0:
        ix = np.where(ns != unk[0])
        ts = ts[ix]
    cases = []
    for i in range(1, int(np.ceil(ts.max())) + 1):
        ix = np.where(ts < i)[0]
        cases.append((i, len(ix)))
    print(cases)
    days, cases = zip(*cases)
    return np.array(cases)


def load_population(path, col=1):
    df = pd.read_csv(path, header=None)
    pop = df.iloc[:, col].sum()
    regions = df.iloc[:, 0].to_numpy().tolist()
    print(regions)
    return pop, regions


class SIR(nn.Module):
    def __init__(self, population):
        super(SIR, self).__init__()
        self.beta = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        # self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-5))
        self.gamma = th.tensor([1.0 / 14]).to(device)
        self.fpos = F.softplus
        self.N = population

    def y0(self, S, I):
        return th.tensor([S0, I0]).to(device).float()

    def forward(self, t, y):
        # S, I, R = y
        beta = self.fpos(self.beta)
        ds = -beta * y[0] * y[1] / self.N
        di = beta * y[0] * y[1] / self.N - self.gamma * y[1]
        dy = th.cat([ds, di])
        return dy

    def __repr__(self):
        with th.no_grad():
            return f"SIR ({self.fpos(self.beta).data.item():.3f})"


class DecaySIR(nn.Module):
    def __init__(self, population):
        super(DecaySIR, self).__init__()
        self.a = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        # self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-5))
        self.gamma = th.tensor([1.0 / 14]).to(device)
        self.fpos = F.softplus
        self.N = population

    def y0(self, S, I):
        return th.tensor([S, I]).to(device)

    def forward(self, t, y):
        S, I = y[0], y[1]
        beta = self.fpos(self.a) * th.exp(self.fpos(self.b) * t)
        ds = -beta * S * I / self.N
        di = beta * S * I / self.N - self.gamma * I
        dy = th.cat([ds, di])
        return dy

    def __repr__(self):
        with th.no_grad():
            return f"DecaySIR ({self.fpos(self.a).data.item():.3f}, {self.fpos(self.b).data.item():.3f})"


class LatentSIR(nn.Module):
    def __init__(self, dim, population):
        super(LatentSIR, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 1, bias=False)
        self.relu = th.sigmoid
        self.b0 = nn.Parameter(th.randn(dim, dtype=th.float))
        # self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-5))
        self.gamma = th.tensor([1 / 14]).to(device)
        self.fpos = th.sigmoid
        self.N = population
        # nn.init.xavier_uniform_(self.Wbeta.weight)
        # nn.init.xavier_uniform_(self.Wbeta2.weight)

    def y0(self, S, I):
        return th.cat([th.tensor([S, I]).to(device), self.b0]).float()

    def forward(self, t, y):
        S, I = y[0], y[1]
        b_last = y.narrow(0, 2, self.dim)
        b_now = self.fc1(b_last)
        beta = self.fc2(self.relu(b_last))
        beta = th.sigmoid(beta).squeeze()
        ds = -beta * S * I / self.N
        di = beta * S * I / self.N - self.gamma * I
        dy = th.cat([ds.unsqueeze(0), di, b_now])
        return dy


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ODE demo")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument(
        "--method", type=str, choices=["dopri5", "adams"], default="dopri5"
    )
    parser.add_argument("--batch_time", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--niters", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--adjoint", action="store_true")
    args = parser.parse_args()

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    device = th.device("cuda:" + str(args.gpu) if th.cuda.is_available() else "cpu")

    # load data
    population, regions = load_population(args.fpop)
    cases = load_confirmed(args.fdat, regions)[10:]
    cases = th.from_numpy(cases).float().to(device)
    tmax = len(cases)
    t = th.arange(tmax).float().to(device) + 1

    I0 = cases[0]
    S0 = population - I0
    R0 = 0.0

    # def get_batch():
    #     s = np.random.randint(0, len(cases) - args.batch_size, args.batch_size)
    #     batch_y0 = cases[s]  # (M, D)
    #     batch_y = th.stack(
    #         [cases[s + i] for i in range(args.batch_time)], dim=0
    #     )  # (T, M, D)
    #     return batch_y0, batch_t, batch_y

    ii = 0

    # func = SIR(population).to(device)
    # func = DecaySIR(population).to(device)
    func = LatentSIR(10, population).to(device)
    y0 = func.y0(S0, I0)

    optimizer = optim.AdamW(func.parameters(), lr=1e-3, weight_decay=0.1)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, y0, t)
        pred_y = pred_y.narrow(1, 1, 1)
        print(th.cat([cases[-10:].unsqueeze(1), pred_y[-10:]], dim=1))

        loss = th.mean((pred_y - cases) ** 2)
        loss.backward()
        optimizer.step()

        print(
            f"Iter {itr:04d} | Loss {loss.item():.2f} | last {pred_y[-1].item()} | model {func}"
        )

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


class MetaSIR(nn.Module):
    def __init__(self, population):
        super(MetaSIR, self).__init__()
        self.M = len(population)
        # self.beta = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.a = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-4))
        self.alphas = th.nn.Parameter(th.randn((self.M, self.M), dtype=th.float))
        # self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(-5))
        self.gamma = th.tensor([1.0 / 14]).to(device)
        self.fpos = F.softplus
        self.Ns = population

    def y0(self, S, I):
        return th.cat([S, I], axis=1).to(device).float()

    def forward(self, t, ys):
        Ss, Is = ys.narrow(1, 0, 1), ys.narrow(1, 1, 1)
        # beta = self.fpos(self.beta)
        beta = self.fpos(self.a) * th.exp(self.fpos(self.b) * t)
        W = self.fpos(self.alphas) / self.Ns
        WIs = th.mm(W, Is)
        dSs = -beta * Ss * WIs
        dIs = beta * Ss * WIs - self.gamma * Is
        dy = th.cat([dSs, dIs], dim=1)
        return dy

    def __repr__(self):
        with th.no_grad():
            return f"SIR ({self.fpos(self.a).data.item():.3f})"


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
    cases = load_confirmed(args.fdat, regions)
    cases = cases.float().to(device)[:, 10:]
    population = population.float().to(device).unsqueeze(1)
    tmax = cases.size(1)
    t = th.arange(tmax).float().to(device) + 1

    I0 = cases.narrow(1, 0, 1)
    print(I0, population, regions, tmax)
    S0 = population - I0

    func = MetaSIR(population).to(device)
    y0 = func.y0(S0, I0)

    optimizer = optim.Adam(func.parameters(), lr=5e-2, betas=[0.9, 0.999])

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, y0, t, method="euler", options={"step_size": 1})
        pred_y = pred_y.narrow(2, 1, 1).squeeze().t()

        loss = th.sum((pred_y[:, -5:] - cases[:, -5:]) ** 2)
        loss.backward()
        optimizer.step()
        if itr % 50 == 0:
            print(th.cat([cases[:, -3:], pred_y[:, -3:]], dim=1))
            print(f"Iter {itr:04d} | Loss {loss.item():.2f} | model {func}")

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


class LatentMLP(nn.Module):
    def __init__(self, layers, dim, input_dim, nlin=th.tanh, bias=True):
        super(LatentMLP, self).__init__()
        self.fc = nn.ModuleList(
            [nn.Linear(dim * 2 + 1, dim, bias=bias)]
            + [nn.Linear(dim + 1, dim, bias=bias) for _ in range(layers - 1)]
        )
        self.nlin = nlin
        for fc in self.fc:
            nn.init.xavier_uniform_(fc.weight)
        with th.no_grad():
            self.fc[-1].weight.fill_(0)

    def forward(self, t, h0, h, x):
        t_ = t.reshape(1, 1).expand(h.size(0), 1)
        # tmp = th.cat([x, h, h0, t_], dim=-1)
        tmp = th.cat([h, h0, t_], dim=-1)
        tmp = self.fc[0](tmp)
        for i in range(1, len(self.fc)):
            tmp = th.cat([self.nlin(tmp), t_], dim=-1)
            tmp = self.fc[i](tmp)
        return tmp

    def __repr__(self):
        return f"MLP {len(self.fc)}"


class FeatMLP(nn.Module):
    def __init__(self, layers, dim, input_dim, nlin=th.tanh):
        super(FeatMLP, self).__init__()
        self.fc = nn.ModuleList(
            [nn.Linear(input_dim, dim, bias=False)]
            + [nn.Linear(dim, dim, bias=False) for _ in range(layers - 1)]
        )
        self.nlin = nlin
        for fc in self.fc:
            nn.init.xavier_uniform_(fc.weight)

    def forward(self, x):
        tmp = self.fc[0](x)
        for i in range(1, len(self.fc)):
            tmp = self.fc[i](self.nlin(tmp))
        return tmp

    def __repr__(self):
        return f"MLP {len(self.fc)}"


class BetaLatent(nn.Module):
    def __init__(self, population, layers, dim, features):
        super(BetaLatent, self).__init__()
        self.M = len(population)
        self.dim = dim
        self.feats = features
        self.latent = LatentMLP(layers, dim, features[0].size(1), bias=True)
        self.observ = FeatMLP(layers, dim, features[0].size(1))
        self.v = nn.Linear(self.dim, 1, bias=True)
        self.h0 = nn.Parameter(th.randn(self.M, dim, dtype=th.float))
        with th.no_grad():
            self.v.weight.fill_(-2)
            nn.init.xavier_uniform_(self.h0)

    def forward(self, t, h, batch):
        x = self.feats[t.long(), batch, :]
        # dh = self.latent(t, self.h0[batch], h) + self.observ(x)
        dh = self.latent(t, self.h0[batch], h, x)
        beta = F.softplus(self.v(th.tanh(h)))
        return beta.squeeze(-1), dh.view(-1)

    def y0(self, batch):
        x = self.feats[0, batch, :]
        return self.h0[batch].view(-1) + self.observ(x).view(-1)


class DiffSIR(nn.Module):
    def __init__(self, population, beta_net, dist):
        super(DiffSIR, self).__init__()
        self.M = len(population)
        # self.gamma = th.nn.Parameter(th.ones(1, dtype=th.float).fill_(0.1))
        self.gamma = th.tensor([1.0 / 14]).to(population.device)
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(8))
        self._observed = nn.Parameter(th.zeros(1))
        self.fpos = F.softplus
        self.Ns = population
        self.beta_net = beta_net
        self._dist = dist
        self.dim = beta_net.dim

    def set_region(self, batch):
        self.region = batch

    def dist(self, scores):
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu[self.region])
        elif self._dist == "normal":
            # return Normal(scores, th.exp(self.nu))
            return Normal(scores, 1)
        else:
            raise RuntimeError(f"Unknown loss")

    def observed(self):
        return th.sigmoid(self._observed)

    def y0(self, S, I, R, region):
        elems = list(
            filter(
                None.__ne__,
                [S.view(-1), I.view(-1), R.view(-1), self.beta_net.y0(region)],
            )
        )
        return th.cat(elems, dim=0).float()

    def forward(self, t, y):
        # prepare input
        n = len(self.region)
        Ss = y.narrow(0, 0, n)
        Is = y.narrow(0, n, n)
        h = y.narrow(0, 3 * n, self.dim * n)  # dbeta??
        h = h.reshape(n, self.dim)

        beta, dBeta = self.beta_net(t, h, self.region)
        assert beta.ndim <= 1, beta.size()

        # compute dynamics
        gIs = self.gamma * Is
        bSIs = beta * gIs * Ss / self.Ns[self.region]
        dSs = -bSIs
        dIs = bSIs - gIs
        dRs = gIs

        self.train_log = (beta, bSIs)
        # print(self.train_log)
        # prepare dy output
        elems = list(filter(None.__ne__, [dSs, dIs, dRs, dBeta]))
        dy = th.cat(elems, dim=0)
        assert y.size() == dy.size(), (y.size(), dy.size())
        return dy

    def __repr__(self):
        with th.no_grad():
            return f"SIR | gamma {self.gamma.item():.3f} | {self.train_log}"


class DiffSEIR(nn.Module):
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

    def y0(self, S, I, R, E=None):
        if E is None:
            E = th.ones_like(I)
        elems = list(
            filter(
                None.__ne__,
                [S.view(-1), I.view(-1), R.view(-1), E.view(-1), self.beta_net.y0()],
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
        elems = list(filter(None.__ne__, [dSs, dIs, dRs, dEs, dBeta]))
        dy = th.cat(elems, dim=0)
        return dy

    def __repr__(self):
        with th.no_grad():
            return f"SEIR | gamma {self.fpos(self.gamma).item():.3f} "


def train(model, dset, mask, population, odeint, optimizer, checkpoint, args):
    M = len(population)
    device = dset.device
    nbatches = 1
    tmax = dset.size(1)
    t = th.arange(tmax).float().to(device)
    nobserved = np.prod(dset.size()) - th.isnan(dset).float().sum()
    print(f"Observed {nobserved} of {np.prod(dset.size())}")

    ix = np.arange(dset.size(0))
    for itr in range(1, args.niters + 1):
        loss = 0
        ae = 0
        np.random.shuffle(ix)
        for batch in [ix[i::nbatches] for i in range(nbatches)]:
            optimizer.zero_grad()

            n = len(batch)
            model.set_region(batch)
            _dset = dset[batch]
            # _mask = mask[batch]
            I0 = _dset.narrow(1, 0, 1).squeeze()
            S0 = population[batch] - I0
            R0 = th.zeros(n).to(device)
            assert I0.size() == R0.size() and I0.size() == S0.size()
            y0 = model.y0(S0, I0, R0, batch).to(device)

            pYs = odeint(model, y0, t, method=args.method, options={"step_size": 1})
            pIs = pYs.narrow(1, n, n)
            pRs = pYs.narrow(1, n * 2, n)
            predicted = (pIs + pRs).t()

            predicted = predicted[:, 1:] - predicted[:, :-1]
            new_cases = _dset[:, 1:] - _dset[:, :-1]
            predicted = predicted[:, :-1].clamp(min=1e-8)
            new_cases = new_cases[:, 1:]
            # print(new_cases)
            # print(predicted)
            _mask = th.isnan(new_cases)  # .logical_not()

            with th.no_grad():
                predicted[_mask] = 1
                new_cases[_mask] = 1
                _mask = _mask.logical_not()
                ae += th.abs(new_cases - predicted)[_mask].sum()

            # compute loss
            dist = model.dist(predicted)
            _loss = -dist.log_prob(new_cases)[_mask]
            # _loss = th.pow(predicted[_mask] - new_cases[_mask], 2)
            assert (_loss == _loss).all(), (_loss, th.where(th.isnan(_loss)))
            _loss = _loss.sum()
            loss += _loss

            # back prop
            _loss.backward()
            optimizer.step()

        # sometimes infected goes below 0 - prevent that
        # check if initial betas are large enough ...
        # perhaps start from large beta and minimize it??

        # control
        if itr % 50 == 0 or loss == 0:
            print(predicted[_mask].max())
            print(new_cases[_mask], predicted[_mask])
            # target betas and estimated ones
            print(
                f"Iter {itr:04d} | Loss {loss / M:.2f} | MAE {ae / nobserved:.2f} | {model} | {model.beta_net} "
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
    pred_Rs = pred_y.narrow(1, 2 * M, M).t()

    Rt = pred_Rs.narrow(1, -1, 1)
    It = cases.narrow(1, -1, 1) - Rt
    St = population.unsqueeze(1) - It - Rt
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

    # report the total I and R
    test_Is = test_preds.narrow(1, 0, M).t().narrow(1, -args.test_on, args.test_on)
    test_Rs = test_preds.narrow(1, M, M).t().narrow(1, -args.test_on, args.test_on)
    test_preds = model.observed() * (test_Is + test_Rs)

    df = pd.DataFrame(test_preds.cpu().int().numpy().T)
    df.columns = regions
    if dstart is not None:
        base = pd.to_datetime(dstart)
        ds = [base + timedelta(i) for i in range(1, args.test_on + 1)]
        df["date"] = ds
        df.set_index("date", inplace=True)
    return df


def _get_dict(args, v, device, regions):
    if hasattr(args, v):
        _feats = []
        if getattr(args, v) is None:
            return None
        for _file in getattr(args, v):
            print(f"Loading {_file}")
            d = th.load(_file)
            feats = None
            for i, r in enumerate(regions):
                if r not in d:
                    print(r)
                    continue
                _f = d[r]
                if feats is None:
                    feats = th.zeros(len(regions), d[r].size(0), _f.size(1))
                feats[i, :, : _f.size(1)] = _f
            # feats.div_(feats.abs().max())
            _feats.append(feats.to(device).float())
        return th.cat(_feats, dim=2)
    else:
        return None


class DiffSIRCV(cv.CV):
    def initialize(self, args):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        cases, regions, basedate = load.load_confirmed_csv(args.fdat)
        cases = cases.to(device)

        time_features = _get_dict(args, "time_features", device, regions)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            time_features = time_features.narrow(0, 0, cases.size(1))
        print("Feature size = {} x {} x {}".format(*time_features.size()))

        dset = th.zeros_like(cases).float().fill_(np.nan)
        feats = th.zeros_like(time_features).float()
        mask = th.zeros_like(cases).bool()
        for i in range(cases.size(0)):
            ix = th.where(cases[i] > args.min_count)[0][0].item()
            ln = cases.size(1) - ix
            dset.narrow(0, i, 1).narrow(1, 0, ln).copy_(cases[i, ix:].float())
            mask.narrow(0, i, 1).narrow(1, -ln, ln).fill_(1)
            feats.narrow(1, i, 1).narrow(0, 0, ln).copy_(
                time_features[ix:, i, :].unsqueeze(1)
            )

        # load population
        populations = load.load_populations_by_region(args.fpop, regions=regions)
        populations = th.from_numpy(populations["population"].values).to(device)
        assert populations.size(0) == len(regions), (regions, populations)

        # initialize at time t0
        self.populations = populations.float().to(device)

        if args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
        self.odeint = odeint

        self.weight_decay = 0
        if args.decay.startswith("latent"):
            dim, layers = args.decay[6:].split("_")
            beta_net = BetaLatent(self.populations, int(layers), int(dim), feats)
            # self.weight_decay = args.weight_decay
        else:
            raise ValueError("Unknown beta function")

        self.func = DiffSIR(self.populations, beta_net, args.loss).to(device)
        return dset, mask, regions, basedate, device

    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        dset, mask, regions, _, device = self.initialize(args)

        optimizer = optim.AdamW(
            self.func.parameters(),
            lr=args.lr,
            betas=[args.momentum, 0.999],
            weight_decay=self.weight_decay,
        )

        # optimization is unstable, quickly it tends to explode
        # check norm_grad weight norm etc...
        # optimizer = optim.RMSprop(func.parameters(), lr=args.lr, weight_decay=weight_decay)

        model = train(
            self.func,
            dset,
            mask,
            self.populations,
            self.odeint,
            optimizer,
            checkpoint,
            args,
        )

        return model

    def run_simulate(self, dset, args, model=None, sim_params=None):
        if model is None:
            raise NotImplementedError
        args.fdat = dset
        cases, regions, basedate, device = self.initialize(args)

        forecast = simulate(
            model, cases, regions, self.populations, self.odeint, args, basedate
        )

        gt = pd.DataFrame(cases.cpu().numpy().transpose(), columns=regions)
        gt.index = pd.date_range(end=basedate, periods=len(gt))
        return pd.concat([gt, forecast]).sort_index().diff().loc[forecast.index]


CV_CLS = DiffSIRCV

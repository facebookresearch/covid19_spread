import argparse
import numpy as np
import pandas as pd
from . import load
from datetime import timedelta

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .cross_val import CV


class BetaExpDecay(nn.Module):
    def __init__(self, population):
        super(BetaExpDecay, self).__init__()
        M = len(population)
        self.a = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.b = th.nn.Parameter(th.ones(M, dtype=th.float).fill_(-4))
        self.fpos = F.softplus

    def forward(self, t, y):
        beta = self.fpos(self.a) * th.exp(-self.fpos(self.b) * t)
        return beta, None

    def __repr__(self):
        with th.no_grad():
            return f"Exp = ({self.fpos(self.a).mean().item():.3f}, {self.fpos(self.b).mean().item():.3f})"

    def y0(self):
        return None


def convolve(x, k=1):
    # np adds padding by default, we remove the edges
    if k == 1:
        return x
    f = [1 / k] * k
    # return np.convolve(x, f)[k-1:1-k]
    return np.convolve(x, f, "valid")


def prep_matrix(cases, eps, k=7):

    # convolution in native pytorch - batched
    device = cases.device
    M = len(cases)

    tmax_pre = cases.size(1)

    # k = 1 case also works as intended!!
    f = th.Tensor([1 / k] * k).view(1, 1, -1).to(device)
    tmp = cases.view(M, 1, -1)

    cases = th.nn.functional.conv1d(tmp, f)
    cases = cases.view(M, -1)

    tmax_post = cases.size(1)

    # they are the same if k == 1
    # and differ by k - 1 otherwise
    print("pre and post: ", tmax_pre, tmax_post)

    new_cases = cases[:, 1:] - cases[:, :-1]
    assert (new_cases >= 0).all()
    tmax = new_cases.size(1)
    # this one less than tmax_post
    # so k less than tmax_pre
    print("after diff: ", tmax)
    # t = th.arange(tmax).to(device) + 1 # 1 to tmax inclusive

    # prepare training matrix
    train = new_cases / cases[:, :-1]  # elementwise division
    train += eps  # add an epsilon to lift zero elements

    train[th.isinf(train)] = np.nan  # set inf's to nan's | each row contains one
    # mask or ~mask # all I need is to swap nans and keep the size of the matrix
    # mask = th.isinf(train) + th.isnan(train) # + is logical or

    mask = th.isnan(train)

    hop = train[~mask]
    print("betas larger than 5: ", len(hop[hop > 5]))

    start_times = mask.sum(1)  # picking this index gives the first value...

    for j, idx in enumerate(start_times):
        row = train[j].clone()
        train[j] = th.cat((row[idx:], row[:idx]))

    mask = th.isnan(train)

    return train, start_times, mask


def train(model, cases, regions, optimizer, checkpoint, args):
    # since cv.py doesn't run this file add this seed here
    th.manual_seed(args.seed)

    train_set, start_times, _ = prep_matrix(cases, args.eps, args.k_avg)

    # TODO
    # instead of t0 add a number cutoff!!
    # or perhaps a beta cutoff??
    # prep tr data with propoer masking shifting and reverting of it ...
    # add convolutions before train
    # add deconvolutions for prediction and evaluation
    # add MC implementation by hand (will help with GPU utilization on pytorch)
    # long term prediction should be evaluated either (1) in an averaged fashion or (just that last day)
    # no reason for beta to fluctuate - but we do need to capture the effects of policies/trends!!
    # check the rank of final matrix in both methods

    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    # TODO different losses
    loss_func = th.nn.MSELoss()

    # set nan's to zero
    train_set[th.isnan(train_set)] = 0
    # mini batch over non-zero entries
    idx = train_set.nonzero()
    bs = args.bs

    for epoch in range(args.epochs):
        p = th.randperm(len(idx))
        idx = idx[p]
        for e in batch(idx, bs):
            rows, cols = e.T[0], e.T[1]
            optimizer.zero_grad()
            pred = model.score(rows, cols)
            rating = train_set[rows, cols]
            loss = loss_func(pred, rating)
            # print(loss.item())

            # back prop
            loss.backward()
            optimizer.step()
        # TODO this is sample loss, implement full tr loss
        print(loss.item())

    print(args)
    return model


def simulate(model, cases, regions, args, dstart=None):
    # print(args.t0, args.test_on)
    device = cases.device

    train_set, start_times, mask = prep_matrix(cases, args.eps, args.k_avg)

    #     for j, idx in enumerate(start_times):
    #         row = train[j].clone()
    #         train[j] = th.cat((row[idx:], row[:idx]))

    # indices of what I want to complete
    idx = mask.nonzero()
    rows, cols = idx.T[0], idx.T[1]
    preds = model.score(rows, cols)

    preds[th.isnan(preds)] = args.eps  # nan cheat (why do I get nans sometimes!!)
    # quite a bit can be neg but not too bad they are at -eps...
    preds[preds < args.eps] = args.eps  # eps cheat (why do I get lots of infs??)

    test_preds = train_set.clone()
    test_preds[mask] = preds

    # test_preds = model.transform(train.cpu())
    # test_preds = model.simulate(tmax, new_cases, args.test_on)

    def fit(x, y):
        # x: times; y: betas
        one = th.ones(len(x))  # .to(device)
        A = th.stack([x, one]).T
        res = th.lstsq(y.view(-1, 1), A)  # see doc for conventions...
        slope, intercept = res[0][:2]  # m > n see documentation
        return slope, intercept

    # keep betas
    test_preds_full = []
    test_preds_part = []
    # keep case numbers
    test_cases_full = []
    test_cases_part = []

    test_preds = test_preds.cpu()
    cases = cases.cpu()
    # do this part on CPU
    # on GPU this loop is toooo slow
    # actually numpy is much faster at this for some reason
    tmax = len(
        test_preds[0]
    )  # after conv, after diff so original tmax_pre - k of the convolution
    for j, idx in enumerate(start_times):

        row = test_preds[j]
        x = th.arange(tmax - args.fit_on + 1, tmax + 1).float()  # .to(device)
        x_pred = th.arange(tmax + 1, tmax + 1 + args.test_on)  # .float()#.to(device)
        y = th.log(row[-args.fit_on :])
        m, b = fit(x, y)
        beta_pred = th.exp(m * x_pred + b)

        # concat nan's... not a square matrix anymore
        nans = th.Tensor([np.nan] * idx)
        tmp = th.cat(
            (nans, row, beta_pred)
        )  # here I shift everything back to real time
        test_preds_full.append(tmp)
        test_preds_part.append(tmp[tmax : tmax + args.test_on])

        # get cases
        case_pred = [cases[j][-1].item()]
        for b in tmp[tmax:]:
            increment = (b + 1) * case_pred[-1]
            case_pred.append(increment.item())

        case_pred = th.Tensor(case_pred)  # .int()#.to(device)

        tmp = th.cat((cases[j], case_pred[1:]))

        #### here deconvolve everything ###

        test_cases_full.append(tmp)
        # here it is tmax + 1 bc tmax has been calculated on the difference!!!
        test_cases_part.append(tmp[tmax + 1 : tmax + 1 + args.test_on])

    # extend beta's with exp pred for args.test_on more days
    # use args.fit_on to fit betas

    test_cases_part = th.stack(test_cases_part)

    df = pd.DataFrame(test_cases_part.T.int().numpy(), columns=regions)

    if dstart is not None:
        base = pd.to_datetime(dstart)
        ds = [base + timedelta(i) for i in range(1, args.test_on + 1)]
        df["date"] = ds

        df.set_index("date", inplace=True)

    return df


def initialize(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    cases, regions, basedate = load.load_confirmed_csv(args.fdat)
    cases = cases.float().to(device)[:, args.t0 :]
    return cases, regions, basedate, device


class MC(nn.Module):
    def __init__(self, regions, tmax, n_emb):
        super(MC, self).__init__()
        self.n_regions = len(regions)
        self.n_days = tmax
        self.n_emb = n_emb
        self.regions_emb = th.nn.Embedding(self.n_regions, self.n_emb, sparse=False)
        self.days_emb = th.nn.Embedding(self.n_days, self.n_emb, sparse=False)

    def score(self, rs, ds, shift=0.1, scale=5):
        # TODO how to pick scale properly??
        # rs and ds are the samples fed
        _score = (self.regions_emb(rs) * self.days_emb(ds)).sum(1)
        #         _score = (th.sigmoid(_score) - shift) * scale
        #         _score = th.relu(_score)
        return _score


class MatrixCCV(CV):
    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        cases, regions, _, device = initialize(args)
        tmax = cases.size(1) + 1
        # is this tmax correct????
        # even if this is somewhat larger those ones just won't be trained...

        # setup beta function for future days of long sequences
        if args.decay == "const":
            beta_net = BetaConst(regions)
        elif args.decay == "exp":
            beta_net = BetaExpDecay(regions)
        else:
            raise ValueError("Unknown beta function")

        model = MC(regions, tmax, args.n_embedding).to(device)

        # TODO play with the optimizer...
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=[args.momentum, 0.999],
            weight_decay=args.weight_decay,
        )
        model = train(model, cases, regions, optimizer, checkpoint, args)
        return model

    def run_simulate(self, dset, args, model=None, sim_params=None):
        args.fdat = dset
        if model is None:
            raise NotImplementedError
        cases, regions, basedate, device = initialize(args)
        forecast = simulate(model, cases, regions, args, basedate)
        gt = pd.DataFrame(cases.cpu().numpy().transpose(), columns=regions)
        gt.index = pd.date_range(end=basedate, periods=len(gt))
        return pd.concat([gt, forecast]).sort_index().diff().loc[forecast.index]


# needed in cv.py
CV_CLS = MatrixCCV


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ODE demo")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-lr", type=float, default=5e-4)
    parser.add_argument("-momentum", type=float, default=0.99)
    parser.add_argument("-eps", type=float, default=1e-4)
    parser.add_argument("-weight-decay", type=float, default=0.95)  # strong wd is req
    parser.add_argument("-niters", type=int, default=2000)
    parser.add_argument("-epochs", type=int, default=20000)
    parser.add_argument("-bs", type=int, default=10000)
    parser.add_argument("-n-embedding", type=int, default=30)
    parser.add_argument("-k_avg", type=int, default=7, help="size of conv window")
    parser.add_argument("-amsgrad", default=False, action="store_true")
    parser.add_argument("-sample-posterior", type=int, default=0, help="0 or 1 for T/F")
    parser.add_argument("-nfeatures", type=int, default=3)
    parser.add_argument("-loss", default="lsq", choices=["lsq", "poisson"])
    parser.add_argument("-decay", default="exp")
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=14, type=int)
    parser.add_argument("-test-on", default=7, type=int)
    parser.add_argument("-window", default=7, type=int)  # ??
    parser.add_argument("-checkpoint", type=str, default="/tmp/metasir_model.bin")
    parser.add_argument("-keep-counties", type=int, default=0)
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    mod = MatrixCCV()

    th.manual_seed(args.seed)
    model = mod.run_train(args.fdat, args, args.checkpoint)

    with th.no_grad():
        forecast = mod.run_simulate(args.fdat, args, model)
        print(forecast)

import argparse
import numpy as np
import pandas as pd
import load
from datetime import timedelta

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

###
# run_train and run_simulate are key!!!
###


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


def train(model, cases, regions, checkpoint, args):

    #     import pdb
    #     pdb.set_trace()
    # fitter is the model fit class
    M = len(regions)
    device = cases.device

    ##### convolve ####
    cases = cases.cpu().numpy()
    cases_ = []
    k = 7
    for j in range(len(cases)):
        cases_.append(np.convolve(cases[j], [1 / k] * k, "valid"))
    cases = th.from_numpy(np.asarray(cases_))
    ##### end convolve ####

    new_cases = cases[:, 1:] - cases[:, :-1]
    #     assert (new_cases >= 0).all()
    tmax = new_cases.size(1)
    # t = th.arange(tmax).to(device) + 1 # 1 to tmax inclusive

    # prepare training matrix
    train = new_cases / cases[:, :-1]  # elementwise division
    train += args.beta_min
    train[th.isinf(train)] = np.nan  # set inf's to nan's
    # mask or ~mask # all I need is to swap nans and keep the size of the matrix
    # mask = th.isinf(train) + th.isnan(train) # + is logical or
    mask = th.isnan(train)
    start_times = mask.sum(1)  # picking this index gives the first value...

    for j, idx in enumerate(start_times):
        row = train[j].clone()
        train[j] = th.cat((row[idx:], row[:idx]))

    # TODO
    # prep tr data with propoer masking shifting and reverting of it ...
    # add convolutions before train
    # add deconvolutions for prediction and evaluation
    # add MC implementation by hand (will help with GPU utilization on pytorch)
    # long term prediction should be evaluated either (1) in an averaged fashion or (just that last day)
    # no reason for beta to fluctuate - but we do need to capture the effects of policies/trends!!
    #     import pdb
    #     pdb.set_trace()

    ### compute the fit matrix model...
    model.fit(train.cpu())
    # res = model.transform(train.cpu())
    #     print(model)
    print(args)

    return model


def simulate(model, cases, regions, args, dstart=None):
    # print(args.t0, args.test_on)

    offset = 0
    ##### convolve ####
    cases = cases.cpu().numpy()
    cases_ = []
    k = 7
    offset = 0
    for j in range(len(cases)):
        cases_.append(np.convolve(cases[j], [1 / k] * k, "valid"))
    cases = th.from_numpy(np.asarray(cases_))
    ##### end convolve ####

    new_cases = cases[:, 1:] - cases[:, :-1]
    #     assert (new_cases >= 0).all()
    tmax = new_cases.size(1)  # this is one less than total len of cases

    ## prep the data in the same way as train but model is fit already
    train = new_cases / cases[:, :-1]  # elementwise division
    train += args.beta_min
    train[th.isinf(train)] = np.nan  # set inf's to nan's
    mask = th.isnan(train)
    start_times = mask.sum(1)  # picking this index gives the first value...
    for j, idx in enumerate(start_times):
        row = train[j].clone()
        train[j] = th.cat((row[idx:], row[:idx]))

    test_preds = model.transform(train.cpu())

    # test_preds = model.simulate(tmax, new_cases, args.test_on)

    def fit(x, y):
        # x: times; y: betas
        A = np.vstack([x, np.ones(len(x))]).T
        res = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = res[0]
        return slope, intercept

    # keep betas
    test_preds_full = []
    test_preds_part = []
    # keep case numbers
    test_cases_full = []
    test_cases_part = []

    for j, idx in enumerate(start_times):
        row = test_preds[j]
        x = range(tmax - args.fit_on + 1, tmax + 1)
        x_pred = range(tmax + 1, tmax + 1 + args.test_on + offset)
        y = np.log(row[-args.fit_on :])
        m, b = fit(x, y)
        beta_pred = np.exp(m * x_pred + b)

        # concat nan's... not a square matrix anymore
        tmp = np.concatenate(([np.nan] * idx, row, beta_pred))
        test_preds_full.append(tmp)
        test_preds_part.append(tmp[tmax : tmax + args.test_on])

        # get cases
        case_pred = [cases[j][-1].item()]
        for b in tmp[tmax:]:
            increment = (b + 1) * case_pred[-1]
            case_pred.append(increment.item())

        case_pred = np.asarray([int(e) for e in case_pred])

        tmp = np.concatenate((cases[j].cpu(), case_pred[1:]))

        #### here deconvolve everything ###

        #         import pdb
        #         pdb.set_trace()

        test_cases_full.append(tmp)
        # here it is tmax + 1 bc tmax has been calculated on the difference!!!
        test_cases_part.append(
            tmp[tmax + 1 + offset : tmax + 1 + args.test_on + offset]
        )

    # extend beta's with exp pred for args.test_on more days
    # use args.fit_on to fit betas

    test_cases_part = np.asarray(test_cases_part)
    test_cases_part = test_cases_part.astype(float)

    df = pd.DataFrame(test_cases_part.T, columns=regions)

    if dstart is not None:
        base = pd.to_datetime(dstart)
        ds = [base + timedelta(i) for i in range(1, args.test_on + 1)]
        df["date"] = ds

        df.set_index("date", inplace=True)
    # print(model.beta(th.arange(tmax + args.test_on).to(cases.device) + 1))
    return df


def initialize(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    cases, regions, basedate = load.load_confirmed_csv(args.fdat)
    cases = cases.float().to(device)[:, args.t0 :]
    return cases, regions, basedate, device


class MatrixCCV(cv.CV):
    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        cases, regions, _, device = initialize(args)
        tmax = cases.size(1) + 1
        weight_decay = 0

        # setup beta function for future days of long sequences
        if args.decay == "const":
            beta_net = BetaConst(regions)
        elif args.decay == "exp":
            beta_net = BetaExpDecay(regions)
        else:
            raise ValueError("Unknown beta function")

        # func = MC(regions, beta_net, args.window).to(device)

        model = IterativeImputer(
            random_state=args.seed,
            tol=1e-3,
            max_iter=args.max_iter,
            sample_posterior=args.sample_posterior,  # changes a lot | True
            n_nearest_features=args.nfeatures,  # may replace averaging | None
            min_value=1e-6,  # min value
            initial_strategy="median",  # try mean/median
        )

        model = train(model, cases, regions, checkpoint, args)
        return model

    def run_simulate(self, dset, args, model=None, sim_params=None):
        args.fdat = dset
        if model is None:
            raise NotImplementedError
        cases, regions, basedate, device = initialize(args)
        forecast = simulate(model, cases, regions, args, basedate)
        return forecast


# needed in cv.py
CV_CLS = MatrixCCV


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ODE demo")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-lr", type=float, default=5e-2)
    parser.add_argument("-beta-min", type=float, default=1e-6)
    parser.add_argument("-weight-decay", type=float, default=0)
    parser.add_argument("-niters", type=int, default=2000)
    parser.add_argument("-max-iter", type=int, default=30, help="for the imputer")
    parser.add_argument("-amsgrad", default=False, action="store_true")
    parser.add_argument("-sample-posterior", type=int, default=0, help="0 or 1 for T/F")
    parser.add_argument("-nfeatures", type=int, default=3)
    parser.add_argument("-loss", default="lsq", choices=["lsq", "poisson"])
    parser.add_argument("-decay", default="exp")
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=7, type=int)
    parser.add_argument("-test-on", default=7, type=int)
    parser.add_argument("-window", default=7, type=int)  # ??
    parser.add_argument("-checkpoint", type=str, default="/tmp/metasir_model.bin")
    parser.add_argument("-keep-counties", type=int, default=0)
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    cv = MatrixCCV()

    th.manual_seed(args.seed)
    model = cv.run_train(args.fdat, args, args.checkpoint)

    with th.no_grad():
        forecast = cv.run_simulate(args.fdat, args, model)
        print(forecast)

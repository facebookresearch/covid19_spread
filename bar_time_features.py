import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import load
import cv
import yaml
import click
import sys
from bar import BAR, BARCV


class BARTimeFeatures(BAR):
    def __init__(
        self, regions, beta, window, dist, graph, features, self_correlation=True
    ):
        super().__init__(regions, beta, window, dist, graph, None, self_correlation)
        if features is not None:
            self.features = features.transpose(0, 1)
        features_size = features.size(-1) if features is not None else 0
        self.z = nn.Parameter(th.zeros((1 + features_size, window)).fill_(1))

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        offset = self.window - 1
        length = ys.size(1) - self.window + 1

        Z = th.zeros(0).sum()
        if self.self_correlation:
            ws = F.softplus(self.z)
            ws = ws.unsqueeze(1)
            if self.features is not None:
                if ys.size(-1) - 1 == self.features.size(1):
                    # In simulation mode, forward fill the features.
                    self.features = th.cat(
                        [self.features, self.features.narrow(1, -1, 1)], dim=1
                    )
                assert ys.size(-1) == self.features.size(1)
                # Concatenate cases onto time features
                ys_ = th.cat([ys.unsqueeze(-1), self.features], axis=-1)
                # Convolve ws (time_feats x 1 x window) over ys_ (ncounties x time_feats x time)
                Z = F.conv1d(
                    F.pad(ys_.transpose(1, 2), (self.z.size(1) - 1, 0)),
                    ws,
                    groups=self.z.size(0),
                )
                # Z (ncounties x time_feats x time)
                Z = Z.sum(1) / float(self.window)
                # Z (ncounties x time)
            else:
                Z = F.conv1d(F.pad(ys, (self.z.size(1) - 1, 0)).unsqueeze(1), ws)
                Z = Z.squeeze(1).div_(float(self.window))

        # cross-correlation
        W = self.metapopulation_weights()

        if self.features is not None:
            # Use the concatenated cases/time features tensor
            Ys_ = th.stack(
                [
                    F.pad(ys_.narrow(1, i, length), (0, 0, self.window - 1, 0))
                    for i in range(self.window)
                ]
            )
            # Ys_ (window x ncounties x time x time_features)
            # When doing bmm, view Ys_ such that the last 2 dimensions are merged (window x ncounties x time * time_features)
            temp = th.bmm(
                W.unsqueeze(0).expand(self.window, -1, -1),
                Ys_.view(Ys_.size(0), Ys_.size(1), -1),
            )
            # temp (window x ncounties x time * time_features)
            # Average over window and time_features dimensions
            Ys = temp.view_as(Ys_).mean(0).mean(-1)
            # Ys (ncounties x time)
        else:
            Ys = th.stack(
                [
                    F.pad(ys.narrow(1, i, length), (self.window - 1, 0))
                    for i in range(self.window)
                ]
            )
            Ys = th.bmm(W.unsqueeze(0).expand(self.window, self.M, self.M), Ys).mean(
                dim=0
            )

        with th.no_grad():
            self.train_stats = (Z.sum().item(), Ys.sum().item())

        # beta evolution
        beta = self.beta(t, ys)
        Ys = beta * (Z + Ys)
        return Ys, beta, W


class BARTimeFeaturesCV(BARCV):
    model_cls = BARTimeFeatures

    def initialize(self, args):
        result = super().initialize(args)
        # Recreate the model, but pass in the time features as AR features
        self.func = self.model_cls(
            self.func.regions,
            self.func.beta,
            args.window,
            args.loss,
            self.func.graph,
            self.func.beta.time_features.clone(),
            self_correlation=getattr(args, "self_correlation", True),
        ).to(self.func._alphas.device)
        return result


CV_CLS = BARTimeFeaturesCV
MODEL_CLS = BARTimeFeatures


def main(args):
    parser = argparse.ArgumentParser("beta-AR")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument("-lr", type=float, default=5e-2)
    parser.add_argument("-weight-decay", type=float, default=0)
    parser.add_argument("-niters", type=int, default=2000)
    parser.add_argument("-amsgrad", default=False, action="store_true")
    parser.add_argument("-loss", default="lsq", choices=["nb", "poisson"])
    parser.add_argument("-decay", default="exp")
    parser.add_argument("-t0", default=10, type=int)
    parser.add_argument("-fit-on", default=5, type=int)
    parser.add_argument("-test-on", default=5, type=int)
    parser.add_argument("-checkpoint", type=str, default="/tmp/bar_model.bin")
    parser.add_argument("-window", type=int, default=25)
    parser.add_argument("-momentum", type=float, default=0.99)
    args = parser.parse_args()

    cv = BARTimeFeaturesCV()

    model = cv.run_train(args.fdat, args, args.checkpoint)

    with th.no_grad():
        forecast = cv.run_simulate(args, model)


if __name__ == "__main__":
    main(sys.argv[1:])

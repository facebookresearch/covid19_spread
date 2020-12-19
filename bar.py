import argparse
import copy
import numpy as np
import pandas as pd
import warnings
from datetime import timedelta

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import NegativeBinomial, Normal, Poisson
from torchcontrib.optim import SWA

import load
import cv
from cv import rebase_forecast_deltas
import yaml
import metrics
import click
import sys
from wavenet import Wavenet, CausalConv1d
from functools import partial
import math
from scipy.stats import nbinom, norm
from bisect import bisect_left, bisect_right
from tqdm import tqdm
import timeit
from typing import List
import os


warnings.filterwarnings("ignore", category=UserWarning)


class MLP(nn.Module):
    def __init__(self, layers, dim, input_dim, nlin=th.tanh):
        super(MLP, self).__init__()
        self.fc = nn.ModuleList(
            [nn.Linear(input_dim, dim)]
            + [nn.Linear(dim, dim) for _ in range(layers - 1)]
        )
        self.nlin = nlin
        for fc in self.fc:
            nn.init.xavier_uniform_(fc.weight)

    def forward(self, x):
        tmp = x
        for i in range(len(self.fc)):
            tmp = self.nlin(self.fc[i](tmp))
        return tmp

    def __repr__(self):
        return f"MLP {len(self.fc)}"


class BetaRNN(nn.Module):
    def __init__(self, M, layers, dim, input_dim, dropout=0.0):
        # initialize parameters
        super(BetaRNN, self).__init__()
        self.h0 = nn.Parameter(th.zeros(layers, M, dim))
        self.rnn = nn.RNN(input_dim, dim, layers, dropout=dropout)
        self.v = nn.Linear(dim, 1, bias=False)
        self.fpos = th.sigmoid

        # initialize weights
        nn.init.xavier_normal_(self.v.weight)
        for p in self.rnn.parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        ht, hn = self.rnn(x, self.h0)
        beta = self.fpos(self.v(ht))
        return beta

    def __repr__(self):
        return str(self.rnn)


class BetaGRU(BetaRNN):
    def __init__(self, M, layers, dim, input_dim, dropout=0.0):
        super().__init__(M, layers, dim, input_dim, dropout)
        self.rnn = nn.GRU(input_dim, dim, layers, dropout=dropout)
        self.rnn.reset_parameters()
        self.h0 = nn.Parameter(th.randn(layers, M, dim))


class BetaLSTM(BetaRNN):
    def __init__(self, M, layers, dim, input_dim, dropout=0.0):
        super().__init__(M, layers, dim, input_dim, dropout)
        self.rnn = nn.LSTM(input_dim, dim, layers, dropout=dropout)
        self.rnn.reset_parameters()
        self.h0 = nn.Parameter(th.zeros(layers, M, dim))
        self.c0 = nn.Parameter(th.randn(layers, M, dim))

    def forward(self, x):
        ht, (hn, cn) = self.rnn(x, (self.h0, self.c0))
        beta = self.fpos(self.v(ht))
        return beta


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, init_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(
            SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim),
            requires_grad=False,
        )

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """
        Stolen from fairseq
        https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, dtype=th.float) * -emb)
        emb = th.arange(num_embeddings, dtype=th.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = th.cat([th.sin(emb), th.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = th.cat([emb, th.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        with th.no_grad():
            max_pos = input.size(0) + 1
            if self.weights is None or max_pos > self.weights.size(0):
                # recompute/expand embeddings if needed
                self.weights = nn.Parameter(
                    SinusoidalPositionalEmbedding.get_embedding(
                        max_pos, self.embedding_dim
                    ),
                    requires_grad=True,
                )
            return self.weights.index_select(0, input).detach()


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout, tmax):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.xavier_normal_(self.in_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)

        nn.init.constant_(self.in_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.positional_embeddings = SinusoidalPositionalEmbedding(embed_dim, tmax + 21)

    def forward(self, x, attn_mask):
        """
        Args:
            x (Tensor[Seq, Bsz, Emb])
            attn_mask (Tensor[Seq, Seq]) - mask that gets added to the attention weights
                Use -inf to mask out
        """
        pos_idxs = th.arange(x.size(0), device=x.device).float()
        seq, bsz, emb = x.shape
        proj = self.in_proj(x).view(seq, bsz, emb, 3)
        query, key, val = map(lambda x: x.squeeze(), proj.chunk(3, -1))

        scaling = float(self.embed_dim) ** -0.5
        query *= scaling

        query = query.transpose(0, 1)  # Bsz x Seq x Emb
        key = key.transpose(0, 1)
        val = val.transpose(0, 1)

        attn_output_weights = query.bmm(key.transpose(1, 2))  # Bsx x Seq x Seq
        attn_output_weights += attn_mask.unsqueeze(0)

        time_decay = (pos_idxs.unsqueeze(0) - pos_idxs.unsqueeze(1)) / x.size(0)

        attn_output_weights = F.softmax(attn_output_weights + time_decay, dim=-1)

        attn_output = attn_output_weights.bmm(val)  # Bsz x Seq x Emb
        attn_output = attn_output.transpose(0, 1)  # Seq x Bsz x Emb

        attn_output = self.out_proj(attn_output)  # Seq x Bsz x Emb
        return attn_output, attn_output_weights


class BetaTransformer(nn.Module):
    def __init__(self, M, layers, dim, input_dim, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(input_dim, 1, dim, dropout, "relu")
        encoder_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, encoder_norm)
        self.rep = f"Transformer(layers={layers}, input_dim={input_dim}, dim={dim}, dropout={dropout})"
        self.v = nn.Linear(input_dim, 1)

    def forward(self, x):
        return th.sigmoid(self.v(self.encoder(x)))

    def __repr__(self):
        return self.rep


class BetaAttn(nn.Module):
    def __init__(self, M, input_dim, tmax, dropout=0.0):
        super().__init__()
        self.attn = SelfAttention(input_dim, dropout, tmax)
        self.v = nn.Linear(input_dim, 1, bias=False)
        nn.init.xavier_normal_(self.v.weight.data)

    def forward(self, x):
        seq_len = x.size(0)
        # This gets added to the attention weights, fill with -inf to zero out the softmax weights
        mask = th.triu(
            th.full((seq_len, seq_len), -float("inf"), device=x.device), diagonal=1
        )
        attn_output, attn_weights = self.attn(x, attn_mask=mask)
        return th.sigmoid(self.v(attn_output))

    def __repr__(self):
        return "attn"


class BetaWavenet(nn.Module):
    def __init__(self, M, blocks, layers, channels, kernel, nfilters):
        super(BetaWavenet, self).__init__()
        self.embeddings = nn.Parameter(th.randn(M, channels * nfilters))
        self.blocks = blocks
        self.layers = layers
        self.kernel = kernel
        self.nlin = th.tanh
        self.wv = Wavenet(
            blocks,
            layers,
            channels,
            kernel,
            embeddings=self.embeddings,
            groups=channels,
            nlin=self.nlin,
            nfilters=nfilters,
        )
        self.v = nn.Linear(channels, 1, bias=True)

    def forward(self, x):
        hs = self.wv(x.permute(0, 2, 1))
        hs = hs.permute(0, 2, 1)
        beta = self.v(hs)
        beta = th.sigmoid(beta)
        return beta

    def __repr__(self):
        return f"Wave ({self.blocks},{self.layers},{self.kernel})"


class BetaConv(nn.Module):
    def __init__(self, M, channels, kernel, nfilters=2):
        super(BetaConv, self).__init__()
        self.kernel = kernel
        self.nfilters = nfilters
        self.M = M
        self.conv = CausalConv1d(
            M * channels,
            nfilters * M * channels,
            kernel,
            dilation=1,
            groups=M * channels,
            bias=True,
        )
        self.v = nn.Linear(nfilters * channels, 1, bias=True)
        self.ones = th.ones(M, 1, 7).div_(7.0).cuda()
        with th.no_grad():
            self.conv.weight.fill_(-5)

    def forward(self, x):
        _x = x.view(x.size(0), -1, 1)
        hs = self.conv(_x)
        hs = hs.view(x.size(0), x.size(1), self.nfilters * x.size(2))
        beta = self.v(hs)
        beta = th.sigmoid(beta)
        beta = F.pad(beta.permute(2, 1, 0), (6, 0))
        beta = F.conv1d(beta, self.ones, groups=self.M)
        beta = beta.permute(2, 1, 0)
        return beta

    def __repr__(self):
        return f"Conv1d ({self.kernel}, {self.nfilters})"


class BetaLatent(nn.Module):
    def __init__(self, fbeta, regions, tmax, time_features):
        """
        Params
        ======
        - regions: names of regions (list)
        - dim: dimensionality of hidden vector (int)
        - layer: number of RNN layers (int)
        - tmax: maximum observation time (float)
        - time_features: tensor of temporal features (time x region x features)
        """
        super(BetaLatent, self).__init__()
        self.M = len(regions)
        self.tmax = tmax
        self.time_features = time_features
        input_dim = 0

        if time_features is not None:
            input_dim += time_features.size(2)

        self.fbeta = fbeta(self.M, input_dim)

    def forward(self, t, ys):
        x = []
        if self.time_features is not None:
            if self.time_features.size(0) > t.size(0):
                f = self.time_features.narrow(0, 0, t.size(0))
            else:
                f = th.zeros(
                    t.size(0), self.M, self.time_features.size(2), device=t.device
                )
                f.copy_(self.time_features.narrow(0, -1, 1))
                f.narrow(0, 0, self.time_features.size(0)).copy_(self.time_features)
            x.append(f)
        x = th.cat(x, dim=2)
        beta = self.fbeta(x)
        return beta.squeeze().t()

    def apply(self, x):
        ht, hn = self.rnn(x, self.h0)
        return self.fpos(self.v(ht))

    def __repr__(self):
        return str(self.fbeta)


class BAR(nn.Module):
    def __init__(
        self,
        regions,
        population,
        beta,
        window,
        dist,
        graph,
        features,
        self_correlation=True,
        cross_correlation=True,
        offset=None,
    ):
        super(BAR, self).__init__()
        self.regions = regions
        self.M = len(regions)
        self.beta = beta
        self.features = features
        self.population = population
        self.self_correlation = self_correlation
        self.cross_correlation = cross_correlation
        self.window = window
        self.z = nn.Parameter(th.ones((self.M, 7)).fill_(1))
        self._alphas = nn.Parameter(th.zeros((self.M, self.M)).fill_(-3))
        self.nu = nn.Parameter(th.ones((self.M, 1)).fill_(8))
        self.scale = nn.Parameter(th.ones((self.M, 1)))
        self._dist = dist
        self.graph = graph
        self.offset = offset
        self.neighbors = self.M
        self.adjdrop = nn.Dropout2d(0.1)
        if graph is not None:
            assert graph.size(0) == self.M, graph.size()
            assert graph.size(1) == self.M, graph.size()
            self.neighbors = graph.sum(axis=1)
        if features is not None:
            self.w_feat = nn.Linear(features.size(1), 1)
            nn.init.xavier_normal_(self.w_feat.weight)

    def dist(self, scores):
        nu = th.zeros_like(scores)
        if self._dist == "poisson":
            return Poisson(scores)
        elif self._dist == "nb":
            return NegativeBinomial(scores, logits=self.nu)
        elif self._dist == "normal":
            return Normal(scores, th.exp(self.nu))
        else:
            raise RuntimeError(f"Unknown loss")

    def alphas(self):
        alphas = self._alphas
        if self.self_correlation:
            with th.no_grad():
                alphas.fill_diagonal_(-1e10)
        return alphas

    def metapopulation_weights(self):
        alphas = self.alphas()
        W = th.sigmoid(alphas)
        W = self.adjdrop(W.unsqueeze(0).unsqueeze(-1))
        W = W.squeeze(0).squeeze(-1).t()
        if self.graph is not None:
            W = W * self.graph
        return W

    def score(self, t, ys):
        assert t.size(-1) == ys.size(-1), (t.size(), ys.size())
        length = ys.size(-1) - self.window + 1

        # beta evolution
        beta = self.beta(t, ys)

        Z = th.zeros(0).sum()
        if self.self_correlation:
            ws = F.softplus(self.z)
            ws = ws.expand(self.M, self.z.size(1))
            # self-correlation
            Z = F.conv1d(
                F.pad(ys.unsqueeze(0) if ys.ndim == 2 else ys, (self.z.size(1) - 1, 0)),
                ws.unsqueeze(1),
                groups=self.M,
            )
            Z = Z.squeeze(0)
            Z = Z.div(float(self.z.size(1)))

        # cross-correlation
        Ys = th.zeros(0).sum(0)
        W = th.zeros(1, 1)
        if self.cross_correlation:
            W = self.metapopulation_weights()
            Ys = th.stack(
                [
                    F.pad(ys.narrow(-1, i, length), (self.window - 1, 0))
                    for i in range(self.window)
                ]
            )
            orig_shape = Ys.shape
            Ys = Ys.view(-1, Ys.size(-2), Ys.size(-1)) if Ys.ndim == 4 else Ys
            Ys = (
                th.bmm(W.unsqueeze(0).expand(Ys.size(0), self.M, self.M), Ys)
                .view(orig_shape)
                .mean(dim=0)
            )
        with th.no_grad():
            self.train_stats = (Z.mean().item(), Ys.mean().item())

        if self.features is not None:
            Ys = Ys + F.softplus(self.w_feat(self.features))

        Ys = beta * (Z + Ys) / self.neighbors
        return Ys, beta, W

    def simulate(self, tobs, ys, days, deterministic=True, return_stds=False):
        preds = ys.clone()
        self.eval()
        assert tobs == preds.size(-1), (tobs, preds.size())
        stds = []
        for d in range(days):
            t = th.arange(tobs + d, device=ys.device) + 1
            s, _, _ = self.score(t, preds)
            assert (s >= 0).all(), s.squeeze()
            if deterministic:
                y = self.dist(s).mean
            else:
                y = self.dist(s).sample()
            assert (y >= 0).all(), y.squeeze()
            y = y.narrow(-1, -1, 1).clamp(min=1e-8)
            preds = th.cat([preds, y], dim=-1)
            stds.append(self.dist(s).stddev)
        preds = preds.narrow(-1, -days, days)
        self.train()
        if return_stds:
            return preds, stds
        return preds

    def __repr__(self):
        return f"bAR({self.window}) | {self.beta} | EX ({self.train_stats[0]:.1e}, {self.train_stats[1]:.1e})"


def train(model, new_cases, regions, optimizer, checkpoint, args):
    print(args)
    days_ahead = getattr(args, "days_ahead", 1)
    M = len(regions)
    device = new_cases.device
    tmax = new_cases.size(1)
    t = th.arange(tmax, device=device) + 1
    size_pred = tmax - days_ahead
    reg = th.tensor([0], device=device)
    target = new_cases.narrow(1, days_ahead, size_pred)

    start_time = timeit.default_timer()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        scores, beta, W = model.score(t, new_cases)
        scores = scores.clamp(min=1e-8)
        assert scores.dim() == 2, scores.size()
        assert scores.size(1) == size_pred + 1
        assert beta.size(0) == M

        # compute loss
        dist = model.dist(scores.narrow(1, days_ahead - 1, size_pred))
        _loss = dist.log_prob(target)
        loss = -_loss.sum(axis=1).mean()

        stddev = model.dist(scores).stddev.mean()
        # loss += stddev * args.weight_decay

        # temporal smoothness
        if args.temporal > 0:
            reg = th.pow(beta[:, 1:] - beta[:, :-1], 2).sum(axis=1).mean()

        # back prop
        (loss + args.temporal * reg).backward()

        # do AdamW-like update for Granger regularization
        if args.granger > 0:
            with th.no_grad():
                mu = np.log(args.granger / (1 - args.granger))
                r = args.lr * args.eta
                err = model.alphas() - mu
                err.fill_diagonal_(0)
                model._alphas.copy_(model._alphas - r * err)

        # make sure we have no NaNs
        assert loss == loss, (loss, scores, _loss)

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        # take gradient step
        optimizer.step()

        # control
        if itr % 100 == 0:
            time = timeit.default_timer() - start_time
            with th.no_grad(), np.printoptions(precision=3, suppress=True):
                length = scores.size(1) - 1
                maes = th.abs(dist.mean - new_cases.narrow(1, 1, length))
                z = model.z
                nu = th.sigmoid(model.nu)
                means = model.dist(scores).mean
                # For some reason, the following `np.histogram` can be non-determinstically very slow.
                # No idea why...
                # hist = np.histogram(
                #     W.cpu().contiguous().numpy().flatten(), bins=np.arange(0, 1.1, 0.1), density=True
                # )
                # print("W hist =", hist[0])
                print(
                    f"[{itr:04d}] Loss {loss.item():.2f} | "
                    f"Temporal {reg.item():.5f} | "
                    f"MAE {maes.mean():.2f} | "
                    f"{model} | "
                    f"{args.loss} ({means[:, -1].min().item():.2f}, {means[:, -1].max().item():.2f}) | "
                    f"z ({z.min().item():.2f}, {z.mean().item():.2f}, {z.max().item():.2f}) | "
                    f"alpha ({W.min().item():.2f}, {W.mean().item():.2f}, {W.max().item():.2f}) | "
                    f"nu ({nu.min().item():.2f}, {nu.mean().item():.2f}, {nu.max().item():.2f}) | "
                    f"nb_stddev ({stddev.data.mean().item():.2f}) | "
                    f"scale ({th.exp(model.scale).mean():.2f}) | "
                    f"time = {time:.2f}s"
                )
                th.save(model.state_dict(), checkpoint)
                start_time = timeit.default_timer()
    print(f"Train MAE,{maes.mean():.2f}")
    return model


def _get_arg(args, v, device, regions):
    if hasattr(args, v):
        print(getattr(args, v))
        fs = []
        for _file in getattr(args, v):
            d = th.load(_file)
            _fs = th.cat([d[r].unsqueeze(0) for r in regions], dim=0)
            fs.append(_fs)
        return th.cat(fs, dim=1).float().to(device)
    else:
        return None


def _get_dict(args, v, device, regions):
    if hasattr(args, v):
        _feats = []
        for _file in getattr(args, v):
            print(f"Loading {_file}")
            d = th.load(_file)
            feats = None
            for i, r in enumerate(regions):
                if r not in d:
                    continue
                _f = d[r]
                if feats is None:
                    feats = th.zeros(len(regions), d[r].size(0), _f.size(1))
                feats[i, :, : _f.size(1)] = _f
            _feats.append(feats.to(device).float())
        return th.cat(_feats, dim=2)
    else:
        return None


def to_one_hot(feats, nbins):
    disc = KBinsDiscretizer(nbins, encode="onehot", strategy="uniform")
    out = th.zeros(feats.size(0), feats.size(1), feats.size(2))
    feats = feats.permute(2, 0, 1)
    for i in range(1, feats.size(0)):
        _f = feats[i].cpu().numpy().flatten().reshape(-1, 1)
        print(_f.min(), _f.max(), _f.mean())
        if _f.min() == _f.max():
            continue
        est = disc.fit(_f)
        _f = disc.transform(_f).nonzero()[1]
        _f = _f.reshape(feats.size(1), feats.size(2))
        out[:, :, i] = th.from_numpy(_f)
    return out.to(feats.device).long()


class BARCV(cv.CV):
    def initialize(self, args):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        cases, regions, basedate = load.load_confirmed_csv(args.fdat)
        assert (cases == cases).all(), th.where(cases != cases)

        # Cumulative max across time
        cases = np.maximum.accumulate(cases, axis=1)

        new_cases = th.zeros_like(cases)
        new_cases.narrow(1, 1, cases.size(1) - 1).copy_(cases[:, 1:] - cases[:, :-1])

        assert (new_cases >= 0).all(), new_cases[th.where(new_cases < 0)]
        new_cases = new_cases.float().to(device)[:, args.t0 :]

        populations = None

        print("Number of Regions =", new_cases.size(0))
        print("Timeseries length =", new_cases.size(1))
        print(
            "Increase: max all = {}, max last = {}, min last = {}".format(
                new_cases.max().item(),
                new_cases[:, -1].max().item(),
                new_cases[:, -1].min().item(),
            )
        )
        tmax = new_cases.size(1) + 1

        # adjust max window size to available data
        args.window = min(args.window, new_cases.size(1) - 4)

        # setup optional features
        graph = (
            th.load(args.graph).to(device).float() if hasattr(args, "graph") else None
        )
        features = _get_arg(args, "features", device, regions)
        time_features = _get_dict(args, "time_features", device, regions)
        if time_features is not None:
            time_features = time_features.transpose(0, 1)
            # TODO/FIXME: uncomment for rigorous test setting
            time_features = time_features.narrow(0, args.t0, new_cases.size(1))
            print("Feature size = {} x {} x {}".format(*time_features.size()))
            print(time_features.min(), time_features.max())

        self.weight_decay = 0
        # setup beta function
        if args.decay == "const":
            beta_net = decay.BetaConst(regions)
        elif args.decay == "exp":
            beta_net = decay.BetaExpDecay(regions)
        elif args.decay == "logistic":
            beta_net = decay.BetaLogistic(regions)
        elif args.decay == "powerlaw":
            beta_net = decay.BetaPowerLawDecay(regions)
        elif args.decay.startswith("poly"):
            degree = int(args.decay[4:])
            beta_net = decay.BetaPolynomial(regions, degree, tmax)
        elif args.decay.startswith("rbf"):
            dim = int(args.decay[3:])
            beta_net = decay.BetaRBF(regions, dim, "gaussian", tmax)
        elif args.decay.startswith("latent"):
            dim, layers = args.decay[6:].split("_")
            fbeta = lambda M, input_dim: BetaRNN(
                M,
                int(layers),
                int(dim),
                input_dim,
                dropout=getattr(args, "dropout", 0.0),
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        elif args.decay.startswith("conv"):
            kernel, nfilters = args.decay[4:].split("_")
            fbeta = lambda M, input_dim: BetaConv(
                M, input_dim, int(kernel), int(nfilters)
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        elif args.decay.startswith("attn"):
            fbeta = partial(BetaAttn, tmax=tmax, dropout=getattr(args, "dropout", 0))
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        elif args.decay.startswith("lstm"):
            dim, layers = args.decay[len("lstm") :].split("_")
            fbeta = lambda M, input_dim: BetaLSTM(
                M,
                int(layers),
                int(dim),
                input_dim,
                dropout=getattr(args, "dropout", 0.0),
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        elif args.decay.startswith("gru"):
            dim, layers = args.decay[len("gru") :].split("_")
            fbeta = lambda M, input_dim: BetaGRU(
                M,
                int(layers),
                int(dim),
                input_dim,
                dropout=getattr(args, "dropout", 0.0),
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        elif args.decay.startswith("transformer"):
            dim, layers = args.decay[len("transformer") :].split("_")
            fbeta = lambda M, input_dim: BetaTransformer(
                M,
                int(layers),
                int(dim),
                input_dim,
                dropout=getattr(args, "dropout", 0.0),
            )
            beta_net = BetaLatent(fbeta, regions, tmax, time_features)
            self.weight_decay = args.weight_decay
        else:
            raise ValueError("Unknown beta function")

        self.func = BAR(
            regions,
            populations,
            beta_net,
            args.window,
            args.loss,
            graph,
            features,
            self_correlation=getattr(args, "self_correlation", True),
            cross_correlation=not getattr(args, "no_cross_correlation", False),
            offset=cases[:, 0].unsqueeze(1).to(device).float(),
        ).to(device)

        return new_cases, regions, basedate, device

    def run_train(self, dset, args, checkpoint):
        args.fdat = dset
        new_cases, regions, _, device = self.initialize(args)

        params = []
        exclude = {
            "z",
            "nu",
            "_alphas",
            "_alpha_weights",
            "beta.fbeta.h0",
            "beta.fbeta.c0",
            "beta.fbeta.conv.weight",
            "beta.fbeta.conv.bias",
            "scale",
        }
        for name, p in dict(self.func.named_parameters()).items():
            wd = 0 if name in exclude else args.weight_decay
            if wd != 0:
                print(f"Regularizing {name} = {wd}")
            params.append({"params": p, "weight_decay": wd})

        optimizer = optim.AdamW(params, lr=args.lr, betas=[args.momentum, 0.999])

        model = train(self.func, new_cases, regions, optimizer, checkpoint, args)
        return model

    def run_prediction_interval(
        self,
        means_pth: str,
        stds_pth: str,
        intervals: List[float],
        quiet=False,
        gaussian=True,
    ):
        means = pd.read_csv(means_pth, index_col="date", parse_dates=["date"])
        stds = pd.read_csv(stds_pth, index_col="date", parse_dates=["date"])

        means_t = means.values
        stds_t = stds.values

        multipliers = np.array([norm.ppf(1 - (1 - x) / 2) for x in intervals])
        result = np.empty((means_t.shape[0], means_t.shape[1], len(intervals), 3))
        if gaussian:
            lower = (
                means_t[:, :, None] - multipliers.reshape(1, 1, -1) * stds_t[:, :, None]
            )
            upper = (
                means_t[:, :, None] + multipliers.reshape(1, 1, -1) * stds_t[:, :, None]
            )
            result = np.stack(
                [np.clip(lower, a_min=0, a_max=None), upper, np.ones(lower.shape)],
                axis=-1,
            )
        else:
            variances = stds_t ** 2
            nfailures = means_t * means_t / (variances - means_t)
            probs = (variances - means_t) / variances
            with tqdm(total=means_t.size * len(intervals), disable=quiet) as pbar:
                for i in range(means_t.shape[0]):
                    for j in range(means_t.shape[1]):
                        if probs[i, j] < 0 or probs[i, j] > 1:
                            result[i, j, :, 0] = np.clip(
                                np.round(
                                    means_t[i, j] - multipliers * stds_t[i, j] - 0.5
                                )
                                + 1,
                                a_min=0,
                                a_max=None,
                            )
                            result[i, j, :, 1] = (
                                np.round(
                                    means_t[i, j] + multipliers * stds_t[i, j] - 0.5
                                )
                                + 1
                            )
                            result[i, j, :, 2] = 1  # fallback to old method
                            pbar.update(len(intervals))
                            continue
                        x = np.arange(0, max(2 * means_t[i, j] + 10 * stds_t[i, j], 20))
                        y = nbinom.cdf(x, nfailures[i, j], probs[i, j])
                        for k, interval in enumerate(intervals):
                            lower = bisect_right(y, (1 - interval) / 2)
                            upper = bisect_left(y, 1 - (1 - interval) / 2)
                            fallback = 0
                            if (
                                lower == len(y)
                                or upper == 0
                                or lower > means_t[i, j]
                                or upper < means_t[i, j]
                            ):
                                fallback = 1
                                lower = np.clip(
                                    np.round(
                                        means_t[i, j]
                                        - multipliers[k] * stds_t[i, j]
                                        - 0.5
                                    )
                                    + 1,
                                    a_min=0,
                                    a_max=None,
                                )
                                upper = (
                                    np.round(
                                        means_t[i, j]
                                        + multipliers[k] * stds_t[i, j]
                                        - 0.5
                                    )
                                    + 1
                                )
                            result[i, j, k, 0] = lower
                            result[i, j, k, 1] = upper
                            result[i, j, k, 2] = fallback
                            pbar.update(1)
        cols = pd.MultiIndex.from_product(
            [means.columns, intervals, ["lower", "upper", "fallback"]]
        )
        result_df = pd.DataFrame(result.reshape(result.shape[0], -1), columns=cols)
        result_df["date"] = means.index
        melted = result_df.melt(
            id_vars=["date"], var_name=["location", "interval", "lower/upper"]
        )
        pivot = melted.pivot(
            index=["date", "location", "interval"],
            columns="lower/upper",
            values="value",
        ).reset_index()
        return pivot.merge(
            means.reset_index().melt(
                id_vars=["date"], var_name="location", value_name="mean"
            ),
            on=["date", "location"],
        ).merge(
            stds.reset_index().melt(
                id_vars=["date"], var_name="location", value_name="std"
            ),
            on=["date", "location"],
        )


CV_CLS = BARCV


@click.group()
def cli():
    pass


@cli.command()
@click.argument("pth")
def simulate(pth):
    chkpnt = th.load(pth)
    cv = BARCV()
    prefix = ""
    if "final_model" in pth:
        prefix = "final_model_"
    cfg = yaml.safe_load(open(f"{os.path.dirname(pth)}/{prefix}bar.yml"))
    args = argparse.Namespace(**cfg["train"])
    new_cases, regions, basedate, device = cv.initialize(args)
    cv.func.load_state_dict(chkpnt)
    res = cv.func.simulate(new_cases.size(1), new_cases, args.test_on)
    df = pd.DataFrame(res.cpu().data.numpy().transpose(), columns=regions)
    df.index = pd.date_range(
        start=pd.to_datetime(basedate) + timedelta(days=1), periods=len(df)
    )
    df = rebase_forecast_deltas(cfg["data"], df)
    gt = pd.read_csv(cfg["data"], index_col="region").transpose()
    gt.index = pd.to_datetime(gt.index)
    print(metrics._compute_metrics(gt, df, nanfill=True))


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

    cv = BARCV()

    model = cv.run_train(args.fdat, args, args.checkpoint)

    with th.no_grad():
        forecast = cv.run_simulate(args, model)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in cli.commands:
        cli()
    else:
        main(sys.argv[1:])

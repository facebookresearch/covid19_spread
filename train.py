#!/usr/bin/env python3

# HACK: set path to timelord submodule
import sys

sys.path.insert(0, "./timelord")

import torch as th
import json
import numpy as np
import os
import pickle
import sys
from collections import defaultdict as ddict
from tqdm import tqdm, trange
import shutil
import time
from torch.utils.data import DataLoader
from datasets import SizeOrderedSampler, TimelordDSet
from ll import SparseEmbeddingSoftplus
from utils import expand_checkpoint_path, load_checkpoint, prepare_dset
import evaluation
from tensorboardX import SummaryWriter
from typing import Optional, Callable, Dict, Any
import argparse
import random
from optim import MultiOptim
from lr_schedulers import CosineScheduler, ConstantScheduler
from scipy.optimize import minimize


class LBFGS:
    def __init__(self, model, dataloader, counts, T_missing, device, n_jumps):
        self.model = model
        self.dataloader = dataloader
        self.counts = counts
        self.T_missing = T_missing
        self.device = device
        self.n_jumps = n_jumps
        self.epoch = -1
        self.last_nll = None

    def _gather_flat_grad(self):
        views = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return th.cat(views, 0)

    def _set_params(self, theta):
        offset = 0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.copy_(th.from_numpy(theta[offset : offset + numel]).view_as(p))
            offset += numel

    def __call__(self, theta):
        with th.no_grad():
            self._set_params(theta)
        self.model.refresh_cache()
        self.model.zero_grad()
        total_nll = 0
        self.model.grad_cache_dest.data.zero_()
        for batch in tqdm(self.dataloader):
            gpu_batch = {k: v.to(self.device) for k, v in batch.items()}
            ll, _, _ = self.model(T_missing=self.T_missing, **gpu_batch)
            nll = ll.sum().neg()
            nll.backward()
            total_nll += nll.item()
        self.model.update_gradient_correction(None)
        self.model.correct_u_grads(th.arange(self.model.nnodes).to(self.device))
        gradients = self._gather_flat_grad()
        self.last_nll = -total_nll
        self.epoch += 1
        return total_nll, (gradients / self.n_jumps).cpu().numpy()


def step(
    optimizer: Optional[th.optim.Optimizer],
    model: th.nn.Module,
    batch: Dict[str, th.Tensor],
    device: th.device,
    T_missing: th.Tensor,
    init: bool,
    opt: argparse.Namespace,
    lr_scheduler=None,
    zero_grad: bool = True,
) -> th.Tensor:
    if zero_grad:
        model.zero_grad()
    gpu_batch = {k: v.to(device) for k, v in batch.items()}
    ll, _, _ = model(T_missing=T_missing, **gpu_batch)
    nll = ll.sum().neg()
    nll.backward()
    if not init:
        model.correct_u_grads(gpu_batch["xs"])
        if optimizer:
            model.pre_update_hook()
            if opt.condition_grads:
                model.condition_gradients()
            optimizer.step()
            if opt.normalize_params:
                model.normalize()
            model.post_update_hook()
        if lr_scheduler:
            lr_scheduler.step()
        if opt.weight_decay > 0:
            model.regularize(opt.weight_decay * lr_scheduler.current_lr)
    return nll, ll


def run(opt, user_control):
    # get correct path for checkpoint (e.g., replace SLURM vars)
    checkpoint = expand_checkpoint_path(opt.checkpoint)

    # print config (i.e., all local variables right after function call)
    print(f"json_conf: {json.dumps(vars(opt))}")

    # setup tensorboard logging
    writer = None
    if opt.tensorboard_logdir is not None:
        writer = SummaryWriter(logdir=expand_checkpoint_path(opt.tensorboard_logdir))

    # load train data
    print("Loading train events")
    # FIXME: the -1 is wrong here. prepare_dset need to be changed such that
    # it uses T_max for endtimes
    episodes, dnames, cnames, T_missing, counts, n_jumps = prepare_dset(
        opt.dset, -1, opt.sparse, opt.quiet, True, opt.timescale
    )

    H = len(cnames)
    M = len(dnames)

    # print data stats
    print(f"Number of episodes   = {H}")
    print(f"Number of dimensions = {M}")
    print(f"Number of events     = {n_jumps}")

    last_epoch = -1

    print("Setting up model...")
    model = SparseEmbeddingSoftplus(M, opt.dim, opt.scale)
    if opt.resume is None:
        model.initialize_weights()

    if opt.sparse:
        args = [(i, dim.item()) for i in range(H) for dim in episodes[i].active]
    else:
        args = [(i, dim) for i in range(H) for dim in range(M)]

    all_log_msgs = []

    def control(epoch, total_ll, t_start, model):
        elapsed_ns = time.perf_counter_ns() - t_start
        with th.no_grad():
            nodes = th.arange(M, dtype=th.long, device=model.self_A.weight.device)
            mus = model.mus(nodes)
            self_a = model.fpos(model.self_A(nodes))
            betas = model.beta()

        correction = np.array([episodes[i].timestamps[-1].item() * M for i in range(H)])

        rnd_ = lambda x: np.round_(x, 4)
        stats = {
            "epoch": epoch,
            "ll": rnd_(total_ll / n_jumps),
            "non_global_ll": rnd_((total_ll - correction.sum()) / n_jumps),
            "mu_avg": rnd_(mus.mean().item()),
            "beta_avg": betas.mean().item(),
            "self_a": rnd_(self_a.mean().item()),
            "elapsed": rnd_(elapsed_ns / 1e9),
        }
        if user_control is not None:
            stats = {**stats, **user_control(model)}

        stats["best_ll"] = max(stats["ll"], control.best_stats["ll"])
        stats["best_epoch"] = (
            epoch
            if stats["ll"] > control.best_stats["ll"]
            else control.best_stats["epoch"]
        )

        all_log_msgs.append(stats)
        th.save(
            {
                "sampler": sampler,
                "lr_scheduler": lr_scheduler.state_dict(),
                "model": model.state_dict(),
                "opt": opt,
                "epoch": epoch,
                "log": all_log_msgs,
                "optimizer": optimizer.state_dict(),
                "random_state": random.getstate(),
            },
            checkpoint,
        )

        if control.best_stats is None or stats["ll"] > control.best_stats["ll"]:
            control.best_stats = stats
            shutil.copy(checkpoint, checkpoint + ".best")

        print(f"json_stats: {json.dumps(stats)}")

    control.best_stats = {"ll": -float("inf"), "epoch": -1}

    # This is an annoying hack to deal with SparseAdam not being able to accomodate dense gradients (beta)
    if opt.optim == "adam":
        eps = 1e-6
        optimizer = MultiOptim(
            th.optim.SparseAdam(
                model.sparse_params(), lr=opt.lr, eps=eps, betas=(opt.momentum, 0.999)
            ),
            th.optim.Adam(
                model.dense_params(), lr=opt.lr, eps=eps, betas=(opt.momentum, 0.999)
            ),
        )
    elif opt.optim == "sgd":
        optimizer = th.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    elif opt.optim == "lbfgs":
        optimizer = th.optim.LBFGS(model.parameters(), lr=opt.lr)

    device = th.device("cuda")
    model = model.to(device)

    print("Setting up dataset")
    dset = TimelordDSet(episodes, args)
    sampler = SizeOrderedSampler(
        th.Tensor([episodes[x].timestamps.size(0) for x, _ in args]), opt.max_events
    )
    dataloader = DataLoader(
        dset, batch_sampler=sampler, num_workers=0, collate_fn=dset.collate
    )
    print("Training...")

    T_missing = th.tensor(T_missing, device=device, dtype=th.double)

    lr_scheduler = ConstantScheduler(optimizer=optimizer, lr=opt.lr)

    if opt.lr_scheduler == "cosine":
        nsteps = len(dataloader) * opt.epochs
        lr_scheduler = CosineScheduler(optimizer, nsteps, int(nsteps * 0.05), opt.lr)

    if opt.resume is not None:
        print("Loading checkpoint...")
        chkpnt = th.load(opt.resume)
        model.load_state_dict(chkpnt["model"])
        last_epoch = chkpnt.get("epoch", -2) + 1
        if "lr_scheduler" in chkpnt:
            lr_scheduler.load_state_dict(chkpnt["lr_scheduler"])
        if "sampler" in chkpnt:
            sampler = chkpnt["sampler"]
            dataloader = DataLoader(
                dset, batch_sampler=sampler, num_workers=0, collate_fn=dset.collate
            )
        if "optimizer" in chkpnt:
            optimizer.load_state_dict(chkpnt["state_dict"])
        if "random_state" in chkpnt:
            random.setstate(chkpnt["random_state"])
        print(f"Resuming from epoch {last_epoch}")

    counts = counts.to(device)

    def train(epoch: int, init: bool):
        t_start = time.perf_counter_ns()
        sampler.shuffle()
        if not init and opt.sparse:
            model.refresh_cache()
            model.update_gradient_correction(counts)
        total_ll = 0
        with tqdm(total=len(dataloader)) as pbar:
            for batch_id, batch in enumerate(dataloader):
                nll, ll = step(
                    optimizer, model, batch, device, T_missing, init, opt, lr_scheduler
                )
                total_ll += -nll.item()
                pbar.set_postfix(lr=f"{lr_scheduler.current_lr:.8f}")
                pbar.update()
        control(epoch, total_ll, t_start, model)
        return total_ll

    if opt.optim == "lbfgs":
        lbfgs = LBFGS(model, dataloader, counts, T_missing, device, n_jumps)
        theta0 = (
            th.cat([x.view(-1) for x in model.parameters() if x.requires_grad])
            .data.cpu()
            .numpy()
        )
        t_start = time.perf_counter_ns()
        minimize(
            lbfgs,
            theta0,
            method="L-BFGS-B",
            jac=True,
            bounds=None,
            callback=lambda _: control(lbfgs.epoch, lbfgs.last_nll, t_start, model),
            options={
                "iprint": 1,
                "ftol": 1e-5,
                "gtol": 1e-4,
                "maxiter": opt.epochs,
                "maxcor": opt.maxcor,
            },
        )
    else:
        for epoch in range(last_epoch, opt.epochs):
            res = train(epoch=epoch, init=epoch == -1 and opt.sparse)


def parse_opt(args):
    parser = argparse.ArgumentParser(description="Train Hawkes Embeddings")
    parser.add_argument("-dset", type=str, help="Train dataset")
    parser.add_argument("-dim", type=int, default=20, help="Embedding dimension")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-momentum", type=float, default=0.9, help="optimizer momentum")
    parser.add_argument("-scale", type=float, default=0.5, help="parameter scale")
    parser.add_argument("-weight-decay", type=float, default=0, help="Weight decay")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "-timescale", type=float, default=1, help="Scale time by 1 / timescale"
    )
    parser.add_argument(
        "-optim",
        default="adam",
        nargs="?",
        const="adam",
        choices=["sgd", "adam", "lbfgs"],
    )
    parser.add_argument(
        "-sparse", default=False, action="store_true", help="Train on sparse episodes"
    )
    parser.add_argument(
        "-quiet", default=False, action="store_true", help="Less output"
    )
    parser.add_argument(
        "-checkpoint",
        default="/tmp/timelord_model.bin",
        help="Name of checkpoint to save",
    )
    parser.add_argument("-lr-scheduler", choices=["cosine", "constant"])
    parser.add_argument(
        "-max-events", type=int, default=4000, help="Maximum number of events per batch"
    )
    parser.add_argument("-seed", default=42, type=int, help="RNG seed")
    parser.add_argument(
        "-fresh",
        default=False,
        action="store_true",
        help="Do not load previous checkpoints",
    )
    parser.add_argument("-tensorboard_logdir", help="Directory to log to")
    parser.add_argument("-resume")
    parser.add_argument(
        "-repro", help="Reproduce a given run using the exact same hyper parameters"
    )
    parser.add_argument("-maxcor", type=int, default=10)
    parser.add_argument(
        "-condition-gradients", action="store_true", dest="condition_grads"
    )
    parser.add_argument(
        "-no-condition-gradients", action="store_false", dest="condition_grads"
    )
    parser.add_argument(
        "-normalize-params", action="store_true", dest="normalize_params"
    )
    parser.add_argument(
        "-no-normalize-params", action="store_false", dest="normalize_params"
    )
    parser.set_defaults(condition_grads=True, normalize_params=True)
    opt = parser.parse_args(args)
    return opt


def main(args, user_control=None):
    opt = parse_opt(args)
    if opt.repro is not None:
        opt = th.load(opt.repro)["opt"]

    np.random.seed(opt.seed)
    th.manual_seed(opt.seed)
    random.seed(opt.seed)
    run(opt, user_control)


if __name__ == "__main__":
    main(sys.argv[1:])

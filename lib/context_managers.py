#!/usr/bin/env python3

import contextlib
import os


@contextlib.contextmanager
def env_var(key_vals):
    old_dict = {k: os.environ.get(k, None) for k in key_vals.keys()}
    os.environ.update(key_vals)
    yield
    for k, v in old_dict.items():
        if v:
            os.environ[k] = v
        else:
            del os.environ[k]


@contextlib.contextmanager
def chdir(d):
    old_dir = os.getcwd()
    os.chdir(d)
    yield
    os.chdir(old_dir)

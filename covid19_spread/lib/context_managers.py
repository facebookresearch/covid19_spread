import contextlib
import os
import copy
import sys
import typing as tp


@contextlib.contextmanager
def env_var(key_vals: tp.Dict[str, tp.Union[str, None]]):
    """
    Context manager for manipulating environment variables.  Environment is restored
    upon exiting the context manager
    Params:
        key_vals - mapping of environment variables to their values.  Of a value is 
        `None`, then it is deleted from the environment.  
    """
    old_dict = {k: os.environ.get(k, None) for k in key_vals.keys()}
    for k, v in key_vals.items():
        if v is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = v
    yield
    for k, v in old_dict.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


@contextlib.contextmanager
def chdir(d):
    old_dir = os.getcwd()
    os.chdir(d)
    yield
    os.chdir(old_dir)


@contextlib.contextmanager
def sys_path(x):
    old_path = copy.deepcopy(sys.path)
    sys.path.insert(0, x)
    yield
    sys.path = old_path

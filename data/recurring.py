#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
import cv
import argparse
import contextlib
import tempfile
from subprocess import check_call, check_output
import sqlite3
import click
import datetime

script_dir = os.path.dirname(os.path.realpath(__file__))
DB = os.path.join(script_dir, ".sweep.db")


def mk_db():
    if not os.path.exists(DB):
        conn = sqlite3.connect(DB)
        conn.execute(
            """
        CREATE TABLE sweeps(
            path text primary key,
            basedate text NOT NULL,
            launch_time real NOT NULL,
            module text NOT NULL,
            slurm_job text,
            id text
        )
        """
        )


@contextlib.contextmanager
def env_var(key, value):
    old_val = os.environ.get(key, None)
    os.environ[key] = value
    yield
    if old_val:
        os.environ[key] = old_val
    else:
        del os.environ[key]


@contextlib.contextmanager
def chdir(d):
    old_dir = os.getcwd()
    os.chdir(d)
    yield
    os.chdir(old_dir)


class Recurring:
    script_dir = script_dir

    def __init__(self):
        mk_db()

    def get_id(self) -> str:
        raise NotImplementedError

    def update_data(self) -> None:
        raise NotImplementedError

    def marker(self) -> str:
        raise NotImplementedError

    def command(self) -> str:
        raise NotImplementedError

    def latest_date(self) -> datetime.date:
        raise NotImplementedError

    def schedule(self) -> str:
        return "*/5 * * * *"  # run every 5 minutes

    def install(self) -> None:
        crontab = check_output(["crontab", "-l"]).decode("utf-8")
        if self.marker() in crontab:
            raise ValueError(
                "Cron job already installed, cleanup crontab"
                " with `crontab -e` before installing again"
            )
        with tempfile.NamedTemporaryFile() as tfile:
            with open(tfile.name, "w") as fout:
                print(crontab, file=fout)
                print(f"# {self.marker()}", file=fout)
                user = os.environ["USER"]
                script = os.path.realpath(__file__)
                schedule = self.schedule()
                logfile = os.path.join(self.script_dir, ".launch_sweep.log")
                home = os.path.expanduser("~")
                cmd = [
                    "source /etc/profile.d/modules.sh",
                    f"source {home}/.profile",
                    f"source {home}/.bash_profile",
                    f"source {home}/.bashrc",
                    "source activate covid19_spread",
                    self.command(),
                ]
                envs = ['PATH="/usr/local/bin:$PATH"', f"USER={user}"]
                print(
                    f'{schedule} {" ".join(envs)} bash -c "{" && ".join(cmd)}" >> {logfile} 2>&1',
                    file=fout,
                )
            check_call(["crontab", tfile.name])

    def refresh(self) -> None:
        latest_date = self.latest_date()
        conn = sqlite3.connect(DB)
        res = conn.execute(
            "SELECT path, launch_time FROM sweeps WHERE basedate=? AND id=?",
            (str(latest_date), self.get_id()),
        )
        for pth, launch_time in res:
            if os.path.exists(pth):
                print(f"Already launched {pth} at {launch_time}, exiting...")
                return
            # This directory got deleted, remove it from the database...
            conn.execute(
                "DELETE FROM sweeps WHERE path=? AND id=?", (pth, self.get_id())
            )
            conn.commit()

        self.update_data()
        sweep_job, sweep_path = self.launch_job()

        vals = (
            sweep_path,
            str(latest_date),
            datetime.datetime.now().timestamp(),
            "mhp",
            sweep_job,
            self.get_id(),
        )
        conn.execute(
            "INSERT INTO sweeps(path, basedate, launch_time, module, slurm_job, id) VALUES (?,?,?,?,?,?)",
            vals,
        )
        conn.commit()

    def launch_job(self):
        # Launch the sweep
        config = os.path.join(script_dir, "../cv/nj.yml")
        with chdir(f"{script_dir}/../"):
            sweep_path, jobs = click.Context(cv.cv).invoke(
                cv.cv, config_pth=config, module="mhp", remote=True
            )
        return jobs[0].job_id.split("_")[0], sweep_path

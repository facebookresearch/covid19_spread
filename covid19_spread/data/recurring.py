#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
import cv
import tempfile
from subprocess import check_call, check_output
import sqlite3
import click
import datetime
from covid19_spread.lib.context_managers import chdir

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
        );
        """
        )
        conn.execute(
            """
        CREATE TABLE submitted(
            sweep_path text UNIQUE,
            submitted_at real NOT NULL,
            FOREIGN KEY(sweep_path) REFERENCES sweeps(path)
        );
        """
        )


class Recurring:
    script_dir = script_dir

    def __init__(self, force=False):
        self.force = force
        mk_db()

    def get_id(self) -> str:
        """Return a unique ID to be used in the database"""
        raise NotImplementedError

    def update_data(self) -> None:
        """Fetch new data (should be idempotent)"""
        raise NotImplementedError

    def command(self) -> str:
        """The command to run in cron"""
        raise NotImplementedError

    def latest_date(self) -> datetime.date:
        """"Return the latest date that we have data for"""
        raise NotImplementedError

    def module(self):
        """CV module to run"""
        return "mhp"

    def schedule(self) -> str:
        """Cron schedule"""
        return "*/5 * * * *"  # run every 5 minutes

    def install(self) -> None:
        """Method to install cron job"""
        crontab = check_output(["crontab", "-l"]).decode("utf-8")
        marker = f"__JOB_{self.get_id()}__"
        if marker in crontab:
            raise ValueError(
                "Cron job already installed, cleanup crontab"
                " with `crontab -e` before installing again"
            )
        envs = (
            check_output(["conda", "env", "list"]).decode("utf-8").strip().split("\n")
        )
        active = [e for e in envs if "*" in e]
        conda_env = None
        if len(active) == 1:
            conda_env = f"source activate {active[0].split()[0]}"

        with tempfile.NamedTemporaryFile() as tfile:
            with open(tfile.name, "w") as fout:
                print(crontab, file=fout)
                print(f"# {marker}", file=fout)
                user = os.environ["USER"]
                script = os.path.realpath(__file__)
                schedule = self.schedule()
                stdoutfile = os.path.join(self.script_dir, f".{self.get_id()}.log")
                stderrfile = os.path.join(self.script_dir, f".{self.get_id()}.err")
                home = os.path.expanduser("~")
                cmd = [
                    "source /etc/profile.d/modules.sh",
                    f"source {home}/.profile",
                    f"source {home}/.bash_profile",
                    f"source {home}/.bashrc",
                    conda_env,
                    "slack-on-fail " + self.command(),
                ]
                cmd = [c for c in cmd if c is not None]
                subject = f"ERROR in recurring sweep: {self.get_id()}"
                envs = [
                    f'PATH="/usr/local/bin:/private/home/{user}/bin:/usr/sbin:$PATH"',
                    "__PROD__=1",
                    f"USER={user}",
                ]
                print(
                    f'{schedule} {" ".join(envs)} bash -c "{" && ".join(cmd)} >> {stdoutfile} 2>> {stderrfile}"',
                    file=fout,
                )
            check_call(["crontab", tfile.name])

    def refresh(self) -> None:
        """Check for new data, schedule a job if new data is found"""
        self.update_data()
        latest_date = self.latest_date()
        conn = sqlite3.connect(DB)
        res = conn.execute(
            "SELECT path, launch_time FROM sweeps WHERE basedate=? AND id=?",
            (str(latest_date), self.get_id()),
        )
        if not self.force:
            for pth, launch_time in res:
                launch_time = datetime.datetime.fromtimestamp(launch_time)
                if os.path.exists(pth):

                    print(f"Already launched {pth} at {launch_time}, exiting...")
                    return
                # This directory got deleted, remove it from the database...
                conn.execute(
                    "DELETE FROM sweeps WHERE path=? AND id=?", (pth, self.get_id())
                )
                conn.commit()

        sweep_path = self.launch_job()

        vals = (
            sweep_path,
            str(latest_date),
            datetime.datetime.now().timestamp(),
            self.module(),
            self.get_id(),
        )
        conn.execute(
            "INSERT INTO sweeps(path, basedate, launch_time, module, id) VALUES (?,?,?,?,?)",
            vals,
        )
        conn.commit()

    def launch_job(self, **kwargs):
        """Launch the sweep job"""
        # Launch the sweep
        config = os.path.join(script_dir, f"../../cv/{kwargs.get('cv_config')}.yml")
        with chdir(f"{script_dir}/../../"):
            sweep_path, jobs = click.Context(cv.cv).invoke(
                cv.cv,
                config_pth=config,
                module=kwargs.get("module", "bar"),
                remote=True,
                array_parallelism=kwargs.get("array_parallelism", 20),
            )
        return sweep_path

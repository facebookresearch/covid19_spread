#!/usr/bin/env python3


from submitit.slurm.slurm import SlurmExecutor, SlurmJob
from submitit.core import core, utils, job_environment
from submitit.core.submission import process_job
import uuid
import typing as tp
import time
import sys
import os
import sqlite3
from functools import partial
import enum
import random
import re
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
    AbstractContextManager,
)
import traceback
import pandas
import itertools
import timeit
import psycopg2


DB_HOST = os.environ.get("SLURM_DB_HOST", "devfair0222")


@contextmanager
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


class TransactionManager(AbstractContextManager):
    """
    Class for managing exclusive database transactions.  This locks the entire 
    database to ensure atomicity.  This allows nesting transactions, where
    the inner transaction is idempotent.
    """

    def __init__(self, db_url: str, nretries: int = 20):
        self.retries = nretries
        self.db_url = db_url
        self.exn = None
        self.conn = None
        self.nesting = 0
        self.start_time = None

    def __getstate__(self):
        state = self.__dict__.copy()
        assert (
            self.conn is None
        ), "You should not pickle a transaction manager with an open connection!!!!"
        del state["conn"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = None

    def __enter__(self):
        print(f"Entering transaction, nesting = {self.nesting}")
        if self.conn is None:
            self.conn = psycopg2.connect(dbname="slurm_pool", host=DB_HOST)
            self.cursor = self.conn.cursor()
            self.start_time = timeit.default_timer()
        self.nesting += 1
        return self.cursor

    def __exit__(self, *args, **kwargs):
        self.nesting -= 1
        print(f"Exiting transaction, nesting = {self.nesting}")
        if self.nesting == 0:
            self.conn.commit()
            self.cursor.close()
            self.cursor = None
            self.conn = None
            print(f"Finished transaction in {timeit.default_timer() - self.start_time}")
            self.start_time = None


class JobStatus(enum.IntEnum):
    pending = 0
    success = 1
    failure = 2
    final = 3  # pending if all other jobs are finished

    def __conform__(self, protocol):
        if protocol is sqlite3.PrepareProtocol:
            return self.value


class Worker:
    def __init__(self, db_pth: str, worker_id: int):
        self.db_pth = db_pth
        self.worker_id = worker_id
        self.sleep = 0

    def fetch_ready_job(self, conn):
        # Select a pending job that doesn't have any unfinished dependencies
        query = f"""
        SELECT 
            jobs.pickle, 
            jobs.job_id, 
            MIN(COALESCE(j2.status, {JobStatus.success})) as min_status, 
            MAX(COALESCE(j2.status, {JobStatus.failure})) AS max_status
        FROM jobs 
        LEFT JOIN dependencies USING(pickle)
        LEFT JOIN jobs j2 ON dependencies.depends_on=j2.pickle
        WHERE 
            jobs.status={JobStatus.pending} AND 
            jobs.id='{self.db_pth}' AND 
            (dependencies.id='{self.db_pth}' OR dependencies.id IS NULL) AND 
            (j2.id='{self.db_pth}' OR j2.id IS NULL)
        GROUP BY jobs.pickle, jobs.job_id 
            HAVING MIN(COALESCE(j2.status, {JobStatus.success})) >= {JobStatus.success} 
            AND MAX(COALESCE(j2.status, {JobStatus.failure})) <= {JobStatus.failure}
        LIMIT 1
        """
        conn.execute(query)
        return conn.fetchall()

    def finished(self, conn):
        conn.execute(
            f"""
        SELECT COUNT(1) FROM jobs 
        WHERE status NOT IN ({JobStatus.success}, {JobStatus.failure}) AND id='{self.db_pth}'
        """
        )
        return conn.fetchone()[0] == 0

    def count_running(self, conn):
        conn.execute(f"SELECT COUNT(1) FROM jobs WHERE status > {len(JobStatus)}")
        return conn.fetchone()[0]

    def get_final_jobs(self, conn):
        conn.execute(
            f"SELECT pickle, job_id FROM jobs WHERE status={JobStatus.final} LIMIT 1"
        )
        return conn.fetchall()

    def run(self):
        worker_job_id = f"worker_{self.worker_id}"
        running_status = (
            len(JobStatus) + self.worker_id + 1
        )  # mark in progress with this code
        transaction_manager = TransactionManager(self.db_pth)
        while True:
            if self.sleep > 0:
                print(f"Sleeping...")
                time.sleep(self.sleep)
            print(f"Worker {self.worker_id} getting job to run")
            with transaction_manager as conn:
                ready = self.fetch_ready_job(conn)
                status = JobStatus.pending
                if len(ready) == 0:  # no jobs ready
                    if self.finished(conn):
                        return  # all jobs are finished, exiting...

                    if self.count_running(conn) > 0:
                        self.sleep = min(max(self.sleep * 2, 1), 30)
                        continue

                    ready = self.get_final_jobs(conn)
                    status = JobStatus.final
                    if len(ready) == 0:
                        self.sleep = min(max(self.sleep * 2, 1), 30)
                        continue
                    print(
                        f"Worker {self.worker_id} is executing final_job: {ready[0][0]}"
                    )

                pickle, job_id = ready[0][0], ready[0][1]
                # Mark that we're working on this job.
                res = conn.execute(
                    f"""
                    UPDATE jobs SET status={running_status}, worker_id='{worker_job_id}'
                    WHERE pickle='{pickle}' AND status={status} AND id='{self.db_pth}'
                    """
                )
            print(f"Worker {self.worker_id} got job to run")

            # Run the job
            # Some environment variable trickery to get submitit to find the correct pickle file
            env_vars = {
                "SLURM_JOB_ID": job_id,
                "SLURM_ARRAY_JOB_ID": None,
                "SLURM_ARRAY_TASK_ID": None,
                "SLURM_PICKLE_PTH": pickle,
            }
            if re.match(r"job_(\d+)", job_id):
                env_vars["SLURM_ARRAY_JOB_ID"] = "job"
                env_vars["SLURM_ARRAY_TASK_ID"] = re.search(r"job_(\d+)", job_id).group(
                    1
                )
            with env_var(
                env_vars
            ):  # will reset os.environ when leaving the context manager
                job_dir = os.path.dirname(pickle)
                env = job_environment.JobEnvironment()
                paths = utils.JobPaths(
                    job_dir, job_id=env.job_id, task_id=env.global_rank
                )
                with paths.stderr.open("w", buffering=1) as stderr, paths.stdout.open(
                    "w", buffering=1
                ) as stdout:
                    with redirect_stderr(stderr), redirect_stdout(stdout):
                        try:
                            process_job(job_dir)
                            status = JobStatus.success
                        except Exception:
                            status = JobStatus.failure
                            traceback.print_exc(file=sys.stderr)

                print(f"Worker {self.worker_id} finished job with status {status}")
                with transaction_manager as conn:
                    conn.execute(
                        f"UPDATE jobs SET status={status.value} WHERE pickle='{pickle}'"
                    )
                print(f"Worker {self.worker_id} updated job status")


class SlurmPoolExecutor(SlurmExecutor):
    def __init__(self, *args, **kwargs):
        db_pth = kwargs.pop("db_pth", None)
        super().__init__(*args, **kwargs)
        self.nested = False
        os.makedirs(self.folder, exist_ok=True)
        if db_pth is None:
            # Place the actual database in ~/.slurm_pool/<unique_id>.db
            unique_filename = str(uuid.uuid4())
            self.db_pth = os.path.expanduser(f"~/.slurm_pool/{unique_filename}.db")
            os.makedirs(os.path.dirname(self.db_pth), exist_ok=True)
            if not os.path.exists(os.path.join(str(self.folder), ".job.db")):
                os.symlink(self.db_pth, os.path.join(str(self.folder), ".job.db"))
        else:
            self.db_pth = db_pth
        print(self.db_pth)
        self.transaction_manager = TransactionManager(self.db_pth)
        with self.transaction_manager as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs(status int, pickle text, job_id text, worker_id text, id TEXT)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_p_idx ON jobs(pickle)")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_id_idx ON jobs(id)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dependencies(pickle text, depends_on text, id TEXT)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS dep_p_idx ON dependencies(pickle)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS dep_d_idx ON dependencies(depends_on)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS dep_id_idx ON dependencies(id)")

    def _submit_command(self, command):
        tmp_uuid = uuid.uuid4().hex
        tasks_ids = list(range(self._num_tasks()))
        job = self.job_class(folder=self.folder, job_id=tmp_uuid, tasks=tasks_ids)
        return job

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        if len(delayed_submissions) == 1:
            jobs = super()._internal_process_submissions(delayed_submissions)
            vals = (
                JobStatus.pending,
                str(jobs[0].paths.submitted_pickle),
                jobs[0].job_id,
                self.db_pth,
            )
            with self.transaction_manager as conn:
                conn.execute(
                    "INSERT INTO jobs(status, pickle, job_id, id) VALUES(%s, %s, %s, %s)",
                    vals,
                )
                return jobs
        # array
        folder = utils.JobPaths.get_first_id_independent_folder(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        pickle_paths = []
        for d in delayed_submissions:
            pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.timeout_countdown = self.max_num_timeout
            d.dump(pickle_path)
            pickle_paths.append(pickle_path)
        n = len(delayed_submissions)
        self._throttle()

        tasks_ids = list(range(len(pickle_paths)))
        jobs: tp.List[core.Job[tp.Any]] = [
            SlurmJob(folder=self.folder, job_id=f"job_{a}", tasks=tasks_ids)
            for a in range(n)
        ]
        with self.transaction_manager as conn:
            for job, pickle_path in zip(jobs, pickle_paths):
                job.paths.move_temporary_file(pickle_path, "submitted_pickle")
                vals = (
                    JobStatus.pending,
                    str(job.paths.submitted_pickle),
                    job.job_id,
                    self.db_pth,
                )
                conn.execute(
                    "INSERT INTO jobs(status, pickle, job_id, id) VALUES(%s, %s, %s, %s)",
                    vals,
                )
        return jobs

    def submit_final_job(
        self, fn: tp.Callable[..., core.R], *args: tp.Any, **kwargs: tp.Any,
    ) -> core.Job[core.R]:
        with self.transaction_manager as conn:
            job = self.submit(fn, *args, **kwargs)
            conn.execute(
                f"UPDATE jobs SET status={JobStatus.final} WHERE pickle=%s AND id=%s",
                (str(job.paths.submitted_pickle), self.db_pth),
            )

    def submit_dependent(
        self,
        depends_on: tp.List[core.Job],
        fn: tp.Callable[..., core.R],
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> core.Job[core.R]:
        ds = utils.DelayedSubmission(fn, *args, **kwargs)
        with self.transaction_manager as conn:
            job = self._internal_process_submissions([ds])[0]
            for dep in depends_on:
                vals = (
                    str(job.paths.submitted_pickle),
                    str(dep.paths.submitted_pickle),
                    self.db_pth,
                )
                conn.execute(
                    "INSERT INTO dependencies(pickle, depends_on, id) VALUES (%s,%s,%s)",
                    vals,
                )
        return job

    def launch(self, folder=None, workers: int = 2):
        if not self.nested:
            with self.transaction_manager as conn:
                vals = (self.db_pth,)
                conn.execute("SELECT COUNT(1) FROM jobs WHERE id=%s", vals)
                (njobs,) = conn.fetchone()
            # workers = min(workers, njobs)
            workers = njobs if workers == -1 else workers
            ex = SlurmExecutor(folder or self.folder)
            ex.update_parameters(**self.parameters)
            ex.map_array(
                lambda x: x.run(), [Worker(self.db_pth, i) for i in range(workers)]
            )
            # ex.map_array(partial(worker, self.db_pth), list(range(workers)))

    def extend_dependencies(self, jobs: tp.List[core.Job]):
        with self.transaction_manager as conn:
            conn.execute(
                """
            SELECT pickle
            FROM dependencies
            WHERE depends_on=%s AND id=%s
            """,
                (os.environ["SLURM_PICKLE_PTH"], self.db_pth),
            )
            my_deps = conn.fetchall()
            for (pickle,), depends_on in itertools.product(my_deps, jobs):
                vals = (
                    str(pickle),
                    str(depends_on.paths.submitted_pickle),
                    self.db_pth,
                )
                conn.execute(
                    "INSERT INTO dependencies (pickle, depends_on, id) VALUES(%s,%s,%s)",
                    vals,
                )

    @contextmanager
    def nest(self):
        self.nested = True
        yield
        self.nested = False

    @contextmanager
    def set_folder(self, folder):
        old_folder = self.folder
        self.folder = folder
        yield
        self.folder = old_folder

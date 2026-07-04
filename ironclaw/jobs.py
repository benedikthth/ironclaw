"""Durable background jobs — the spine of the long-running case.

A PhD that enqueues a slurm job, starts a training run, or sleeps waiting for a
crawl cannot hold that in a live chat loop for six hours. So a job is *durable*:
it is submitted, the PhD *suspends*, and something later polls the job and
*resumes* the PhD when it finishes. This must survive the PhD process dying and
restarting.

``LocalProcessBackend`` achieves restart-safety with sentinel files: the spawned
process writes its output and a final exit-code file itself, so completion is
discoverable by any later process with no live handle. Real slurm/pufferlib/etc.
backends implement the same tiny protocol.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobRecord:
    id: str
    kind: str
    command: str
    status: JobStatus = JobStatus.RUNNING


class JobBackend(Protocol):
    # ``request`` (a scheduler.ResourceRequest) is accepted for a uniform call
    # site; raw backends ignore it, the Scheduler uses it for admission.
    def submit(self, kind: str, command: str, request: object | None = None) -> str: ...
    def poll(self, job_id: str) -> tuple[JobStatus, str]: ...


class JobStore:
    """Durable metadata index of submitted jobs (JSON on disk)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._records: dict[str, JobRecord] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for r in json.load(fh):
                    self._records[r["id"]] = JobRecord(
                        r["id"], r["kind"], r["command"], JobStatus(r["status"])
                    )

    def put(self, record: JobRecord) -> None:
        self._records[record.id] = record
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump([_rec_dict(r) for r in self._records.values()], fh, indent=2)

    def get(self, job_id: str) -> JobRecord | None:
        return self._records.get(job_id)


def _rec_dict(r: JobRecord) -> dict:
    d = asdict(r)
    d["status"] = r.status.value
    return d


class LocalProcessBackend(JobBackend):
    """Runs a shell command as a detached process; completion is restart-safe.

    Output goes to ``<jobs_dir>/<id>.out`` and the exit code to ``<id>.code`` once
    the process finishes, so ``poll`` works from a fresh process with no live
    handle to the child.
    """

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.jobs_dir = os.path.join(workspace, ".jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._counter = 0
        # Hold handles so the launching process can reap its own children instead
        # of leaking zombies. Restart-safety does not depend on these — a *fresh*
        # process resumes purely from the on-disk sentinels.
        self._procs: list[subprocess.Popen] = []

    def _new_id(self) -> str:
        # Deterministic within a process; unique across the store via file check.
        while True:
            self._counter += 1
            jid = f"job_{self._counter}"
            if not os.path.exists(os.path.join(self.jobs_dir, f"{jid}.code")):
                return jid

    def submit(self, kind: str, command: str, request: object | None = None) -> str:
        jid = self._new_id()
        out = os.path.join(self.jobs_dir, f"{jid}.out")
        code = os.path.join(self.jobs_dir, f"{jid}.code")
        # Wrapper runs the command in a subshell (so a command that calls `exit`
        # cannot kill the wrapper before it records the code), captures output,
        # then writes the exit code atomically as the last step so poll never
        # sees a partial result.
        wrapper = f"( {command} ) > {out!r} 2>&1 ; echo $? > {code!r}.tmp ; mv {code!r}.tmp {code!r}"
        proc = subprocess.Popen(
            ["sh", "-c", wrapper],
            cwd=self.workspace,
            start_new_session=True,
        )
        self._procs.append(proc)
        return jid

    def _reap(self) -> None:
        for p in self._procs:
            try:
                p.poll()  # non-blocking; sets returncode for exited children
            except Exception:
                pass

    def __del__(self) -> None:  # best-effort child cleanup
        self._reap()

    def poll(self, job_id: str) -> tuple[JobStatus, str]:
        # Reap any of our finished children (non-blocking) so they don't linger
        # as zombies while the launcher is alive.
        self._reap()
        code_path = os.path.join(self.jobs_dir, f"{job_id}.code")
        out_path = os.path.join(self.jobs_dir, f"{job_id}.out")
        if not os.path.exists(code_path):
            return JobStatus.RUNNING, ""
        with open(code_path, "r", encoding="utf-8") as fh:
            exit_code = int((fh.read() or "1").strip() or "1")
        output = ""
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as fh:
                output = fh.read()
        status = JobStatus.DONE if exit_code == 0 else JobStatus.FAILED
        return status, f"[exit {exit_code}]\n{output}".rstrip()


class FakeJobBackend(JobBackend):
    """Test backend with externally-controlled completion, so the suspend/resume
    path is deterministic (no races on process timing)."""

    def __init__(self) -> None:
        self._jobs: dict[str, tuple[JobStatus, str]] = {}
        self._n = 0

    def submit(self, kind: str, command: str, request: object | None = None) -> str:
        self._n += 1
        jid = f"fake_{self._n}"
        self._jobs[jid] = (JobStatus.RUNNING, "")
        return jid

    def complete(self, job_id: str, output: str, ok: bool = True) -> None:
        self._jobs[job_id] = (JobStatus.DONE if ok else JobStatus.FAILED, output)

    def poll(self, job_id: str) -> tuple[JobStatus, str]:
        return self._jobs[job_id]

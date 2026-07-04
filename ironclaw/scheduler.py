"""Resource-aware job scheduler — the arbiter between agents and finite compute.

Because a PhD *suspends to disk* while a job runs (see runtime/state.py), the
scheduler does not gate reasoning loops — those are cheap and I/O-bound. It gates
the heavy *jobs* the agents spawn, against a declared resource pool.

Two design choices matter:

- **Backfill admission, not FIFO.** The scheduler scans the whole pending queue
  in priority order and admits every job that *fits* current capacity, skipping
  (not stopping at) ones that don't. This is the fix for head-of-line blocking:
  a CPU-only job is admitted immediately even while a GPU job sits queued waiting
  for a GPU to free. Pure FIFO would make the CPU job wait behind the GPU job for
  no reason.

- **Resource domains.** A job targeting an external scheduler (Slurm) is not
  gated by *our* local pool — the cluster owns that admission. ``domain="local"``
  jobs are leased against the pool; other domains are handed straight to their
  backend.

The scheduler implements the same ``submit``/``poll`` shape as a JobBackend, so
it is a drop-in for what the PhD already awaits: "queued" and "running" both read
as not-yet-done, so the PhD simply stays suspended.

Not yet implemented (v2): preemption (checkpoint a low-priority GPU job to admit
a higher one) and fair-share weighting. Backfill can starve a large job if small
jobs keep fragmenting the pool; the intended fix is Slurm-style reservations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .jobs import JobBackend, JobStatus


@dataclass(frozen=True)
class ResourceRequest:
    resources: dict[str, float] = field(default_factory=dict)  # {"gpu": 1, "cpu": 4}
    priority: int = 0  # higher is admitted first
    domain: str = "local"  # "local" gates on our pool; else delegated


class ResourcePool:
    """Counting capacity across named resources. The Infra Manager declares it."""

    def __init__(self, capacity: dict[str, float]) -> None:
        self.capacity = dict(capacity)
        self.used: dict[str, float] = {r: 0.0 for r in capacity}

    def fits(self, req: ResourceRequest) -> bool:
        for r, amt in req.resources.items():
            if self.used.get(r, 0.0) + amt > self.capacity.get(r, 0.0):
                return False
        return True

    def acquire(self, req: ResourceRequest) -> None:
        for r, amt in req.resources.items():
            self.used[r] = self.used.get(r, 0.0) + amt

    def release(self, req: ResourceRequest) -> None:
        for r, amt in req.resources.items():
            self.used[r] = max(0.0, self.used.get(r, 0.0) - amt)

    def available(self) -> dict[str, float]:
        return {r: self.capacity[r] - self.used.get(r, 0.0) for r in self.capacity}


class SchedState(str, Enum):
    QUEUED = "queued"  # waiting for a resource lease
    RUNNING = "running"  # admitted and executing on the backend
    DONE = "done"
    FAILED = "failed"


@dataclass
class ScheduledJob:
    id: str
    kind: str
    command: str
    request: ResourceRequest
    seq: int
    state: SchedState = SchedState.QUEUED
    backend_id: str | None = None  # backend job id once admitted
    output: str = ""


_TERMINAL = {SchedState.DONE, SchedState.FAILED}


class Scheduler(JobBackend):
    """Admission control over a JobBackend, gated by a ResourcePool.

    Call ``tick()`` to drive admission and reap completions; a daemon calls it on
    a loop / on events. ``tick`` returns the ids of jobs that just finished, so
    the caller can resume the PhDs suspended on them.
    """

    def __init__(self, backend: JobBackend, pool: ResourcePool) -> None:
        self.backend = backend
        self.pool = pool
        self._jobs: dict[str, ScheduledJob] = {}
        self._seq = 0

    # -- JobBackend surface (drop-in for the PhD's await) ------------------- #

    def submit(self, kind: str, command: str, request: ResourceRequest | None = None) -> str:
        self._seq += 1
        jid = f"sched_{self._seq}"
        self._jobs[jid] = ScheduledJob(
            id=jid,
            kind=kind,
            command=command,
            request=request or ResourceRequest(),
            seq=self._seq,
        )
        return jid

    def poll(self, job_id: str) -> tuple[JobStatus, str]:
        job = self._jobs[job_id]
        # Refresh state for a running job so poll reflects reality even between
        # ticks (a PhD resume polls directly).
        if job.state is SchedState.RUNNING:
            self._refresh(job)
        if job.state is SchedState.DONE:
            return JobStatus.DONE, job.output
        if job.state is SchedState.FAILED:
            return JobStatus.FAILED, job.output
        # QUEUED or RUNNING both read as "not done" to the waiting PhD.
        return JobStatus.RUNNING, ""

    # -- scheduler driver --------------------------------------------------- #

    def tick(self) -> list[str]:
        """Reap finished jobs (release leases), then backfill-admit what fits."""
        finished: list[str] = []
        for job in self._jobs.values():
            if job.state is SchedState.RUNNING and self._refresh(job) in _TERMINAL:
                finished.append(job.id)
        self._admit()
        return finished

    def state_of(self, job_id: str) -> SchedState:
        return self._jobs[job_id].state

    # -- internals ---------------------------------------------------------- #

    def _refresh(self, job: ScheduledJob) -> SchedState:
        status, output = self.backend.poll(job.backend_id)  # type: ignore[arg-type]
        if status is JobStatus.RUNNING:
            return SchedState.RUNNING
        job.output = output
        job.state = SchedState.DONE if status is JobStatus.DONE else SchedState.FAILED
        if job.request.domain == "local":
            self.pool.release(job.request)
        return job.state

    def _admit(self) -> None:
        pending = sorted(
            (j for j in self._jobs.values() if j.state is SchedState.QUEUED),
            key=lambda j: (-j.request.priority, j.seq),
        )
        for job in pending:
            # Delegated domains (e.g. slurm) are not gated by our local pool.
            if job.request.domain != "local":
                self._launch(job)
                continue
            # Backfill: skip jobs that don't fit, keep scanning smaller ones.
            if self.pool.fits(job.request):
                self.pool.acquire(job.request)
                self._launch(job)

    def _launch(self, job: ScheduledJob) -> None:
        job.backend_id = self.backend.submit(job.kind, job.command)
        job.state = SchedState.RUNNING

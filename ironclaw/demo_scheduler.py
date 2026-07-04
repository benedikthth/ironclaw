"""Slice 3 demo: a CPU job is not held hostage by contending GPU jobs.

One GPU, two GPU jobs, one CPU job. Backfill admission runs the CPU job
immediately alongside the first GPU job, while the second GPU job waits for the
lease — exactly the "no reason for a CPU user to wait behind a GPU user" case.

    python -m ironclaw.demo_scheduler
"""

from __future__ import annotations

from .jobs import FakeJobBackend
from .scheduler import ResourcePool, ResourceRequest, Scheduler


def main() -> None:
    # FakeJobBackend keeps the focus on admission state (no real subprocesses).
    sched = Scheduler(FakeJobBackend(), ResourcePool({"gpu": 1, "cpu": 8}))
    jobs = {
        "gpu-train-A": sched.submit("train", "train A", ResourceRequest({"gpu": 1})),
        "gpu-train-B": sched.submit("train", "train B", ResourceRequest({"gpu": 1})),
        "cpu-crawl-C": sched.submit("crawl", "crawl C", ResourceRequest({"cpu": 2})),
    }
    sched.tick()  # admission pass
    print(f"pool available after admission: {sched.pool.available()}\n")
    for label, jid in jobs.items():
        print(f"  {label:14s} -> {sched.state_of(jid).value}")
    print(
        "\nCPU job runs alongside the first GPU job; the second GPU job waits "
        "for the GPU lease (backfill, not head-of-line blocking)."
    )


if __name__ == "__main__":
    main()

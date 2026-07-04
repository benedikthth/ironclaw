"""Slice 3: resource-aware scheduling with backfill + PhD integration."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.phd import resume_phd, run_phd  # noqa: E402
from ironclaw.contracts import AcceptanceCriterion, CheckKind, Task, TaskStatus  # noqa: E402
from ironclaw.jobs import FakeJobBackend  # noqa: E402
from ironclaw.providers.base import AssistantTurn, ToolCall  # noqa: E402
from ironclaw.providers.fake import FakeProvider  # noqa: E402
from ironclaw.scheduler import (  # noqa: E402
    ResourcePool,
    ResourceRequest,
    ScheduledJob,
    SchedState,
    Scheduler,
)


def _backend_id(sched: Scheduler, jid: str) -> str:
    return sched._jobs[jid].backend_id  # test reaches into internals deliberately


class TestBackfill(unittest.TestCase):
    def test_cpu_job_not_blocked_behind_saturating_gpu_jobs(self):
        # The exact scenario: two GPU jobs contend for one GPU; a CPU job must not
        # wait behind them.
        sched = Scheduler(FakeJobBackend(), ResourcePool({"gpu": 1, "cpu": 4}))
        gpu_a = sched.submit("train", "a", ResourceRequest({"gpu": 1}))
        gpu_b = sched.submit("train", "b", ResourceRequest({"gpu": 1}))
        cpu_c = sched.submit("crawl", "c", ResourceRequest({"cpu": 1}))

        sched.tick()

        self.assertIs(sched.state_of(gpu_a), SchedState.RUNNING)
        self.assertIs(sched.state_of(gpu_b), SchedState.QUEUED)  # waits for the GPU
        self.assertIs(sched.state_of(cpu_c), SchedState.RUNNING)  # backfilled past B

    def test_lease_releases_and_next_gpu_job_admits(self):
        backend = FakeJobBackend()
        sched = Scheduler(backend, ResourcePool({"gpu": 1}))
        j1 = sched.submit("train", "a", ResourceRequest({"gpu": 1}))
        j2 = sched.submit("train", "b", ResourceRequest({"gpu": 1}))

        sched.tick()
        self.assertIs(sched.state_of(j1), SchedState.RUNNING)
        self.assertIs(sched.state_of(j2), SchedState.QUEUED)
        self.assertEqual(sched.pool.available()["gpu"], 0)

        backend.complete(_backend_id(sched, j1), "[exit 0]\ndone")
        finished = sched.tick()

        self.assertIn(j1, finished)
        self.assertIs(sched.state_of(j1), SchedState.DONE)
        self.assertIs(sched.state_of(j2), SchedState.RUNNING)  # lease freed -> admitted

    def test_priority_orders_admission(self):
        sched = Scheduler(FakeJobBackend(), ResourcePool({"gpu": 1}))
        low = sched.submit("train", "low", ResourceRequest({"gpu": 1}, priority=0))
        high = sched.submit("train", "high", ResourceRequest({"gpu": 1}, priority=10))
        sched.tick()
        self.assertIs(sched.state_of(high), SchedState.RUNNING)
        self.assertIs(sched.state_of(low), SchedState.QUEUED)

    def test_delegated_domain_is_not_gated_by_local_pool(self):
        # A slurm job should launch even with no local GPU capacity.
        sched = Scheduler(FakeJobBackend(), ResourcePool({"gpu": 0}))
        j = sched.submit("train", "x", ResourceRequest({"gpu": 8}, domain="slurm:gpu"))
        sched.tick()
        self.assertIs(sched.state_of(j), SchedState.RUNNING)


class TestPhdUnderScheduler(unittest.TestCase):
    def test_phd_suspends_while_resource_blocked_then_resumes(self):
        backend = FakeJobBackend()
        sched = Scheduler(backend, ResourcePool({"gpu": 1}))
        # Pre-occupy the only GPU with an unrelated job so the PhD's job must wait.
        hog = sched.submit("train", "hog", ResourceRequest({"gpu": 1}))
        sched.tick()
        self.assertIs(sched.state_of(hog), SchedState.RUNNING)

        task = Task(
            goal="Train and record the result.",
            acceptance=[
                AcceptanceCriterion("result exists", CheckKind.FILE_EXISTS, "result.txt"),
                AcceptanceCriterion("has value", CheckKind.COMMAND, "grep -q ok result.txt"),
            ],
        )
        script = [
            AssistantTurn(tool_calls=[ToolCall("s", "start_job", {"kind": "train", "command": "run", "resources": {"gpu": 1}})]),
            AssistantTurn(tool_calls=[ToolCall("w", "await_job", {"job_id": "sched_2"})]),
            AssistantTurn(tool_calls=[ToolCall("r", "write_file", {"path": "result.txt", "content": "ok\n"})]),
            AssistantTurn(tool_calls=[ToolCall("d", "submit_result", {"summary": "done", "artifacts": [{"path": "result.txt", "kind": "data"}]})]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state.json")
            out = run_phd(
                task=task,
                provider=FakeProvider(script),
                workspace=tmp,
                jobs=sched,
                state_path=state_path,
            )
            # PhD's job is QUEUED (GPU held by hog) -> PhD suspended.
            self.assertEqual(out.result.status, TaskStatus.WAITING)
            sched.tick()  # GPU still held; PhD job stays queued
            self.assertEqual(
                resume_phd(state_path=state_path, provider=FakeProvider(script), jobs=sched).result.status,
                TaskStatus.WAITING,
            )

            # Free the GPU; scheduler admits the PhD's job.
            backend.complete(_backend_id(sched, hog), "done")
            sched.tick()
            self.assertIs(sched.state_of("sched_2"), SchedState.RUNNING)

            # Job finishes; PhD resumes to completion.
            backend.complete(_backend_id(sched, "sched_2"), "[exit 0]\ntrained")
            sched.tick()
            final = resume_phd(state_path=state_path, provider=FakeProvider(script), jobs=sched)
            self.assertEqual(final.result.status, TaskStatus.PASSED)


if __name__ == "__main__":
    unittest.main()

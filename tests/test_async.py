"""Slice 2: durable background jobs and PhD suspend/resume."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.phd import resume_phd, run_phd  # noqa: E402
from ironclaw.contracts import AcceptanceCriterion, CheckKind, Task, TaskStatus  # noqa: E402
from ironclaw.jobs import FakeJobBackend, JobStatus, LocalProcessBackend  # noqa: E402
from ironclaw.providers.base import AssistantTurn, ToolCall  # noqa: E402
from ironclaw.providers.fake import FakeProvider  # noqa: E402

TASK = Task(
    goal="Run the background job and record its result.",
    acceptance=[
        AcceptanceCriterion("result exists", CheckKind.FILE_EXISTS, "result.txt"),
        AcceptanceCriterion("has the value", CheckKind.COMMAND, "grep -q 42 result.txt"),
    ],
)


def _script():
    # Stateless script: indexed by count of assistant turns in the log, so it
    # survives suspend/resume (the provider is rebuilt on resume).
    return [
        AssistantTurn(tool_calls=[ToolCall("s", "start_job", {"kind": "compute", "command": "echo 42"})]),
        AssistantTurn(tool_calls=[ToolCall("w", "await_job", {"job_id": "fake_1"})]),
        AssistantTurn(tool_calls=[ToolCall("r", "write_file", {"path": "result.txt", "content": "42\n"})]),
        AssistantTurn(
            tool_calls=[
                ToolCall("d", "submit_result", {"summary": "recorded 42", "artifacts": [{"path": "result.txt", "kind": "data"}]})
            ]
        ),
    ]


class TestSuspendResume(unittest.TestCase):
    def test_phd_suspends_on_pending_job_then_resumes_to_pass(self):
        backend = FakeJobBackend()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state.json")
            # First pass: submit job, await -> pending -> suspend.
            out1 = run_phd(
                task=TASK,
                provider=FakeProvider(_script()),
                workspace=tmp,
                jobs=backend,
                state_path=state_path,
            )
            self.assertEqual(out1.result.status, TaskStatus.WAITING)
            self.assertEqual(out1.waiting_on, "fake_1")
            self.assertTrue(os.path.exists(state_path))

            # Resuming while still pending must stay WAITING (idempotent).
            still = resume_phd(state_path=state_path, provider=FakeProvider(_script()), jobs=backend)
            self.assertEqual(still.result.status, TaskStatus.WAITING)

            # Job completes; resume drives the task to completion.
            backend.complete("fake_1", "[exit 0]\n42")
            out2 = resume_phd(state_path=state_path, provider=FakeProvider(_script()), jobs=backend)
            self.assertEqual(out2.result.status, TaskStatus.PASSED)
            self.assertTrue(all(c.passed for c in out2.result.checks))


class TestLocalProcessBackend(unittest.TestCase):
    def test_job_completes_and_is_restart_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalProcessBackend(tmp)
            job_id = backend.submit("compute", "echo hello")
            # Poll with a bounded wait; a fresh backend object (restart) must see
            # completion via the on-disk sentinel, not any live handle.
            fresh = LocalProcessBackend(tmp)
            status = JobStatus.RUNNING
            for _ in range(100):
                status, output = fresh.poll(job_id)
                if status is not JobStatus.RUNNING:
                    break
                time.sleep(0.02)
            self.assertIs(status, JobStatus.DONE)
            self.assertIn("hello", output)

    def test_failed_job_reports_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalProcessBackend(tmp)
            job_id = backend.submit("compute", "exit 3")
            status = JobStatus.RUNNING
            for _ in range(100):
                status, output = backend.poll(job_id)
                if status is not JobStatus.RUNNING:
                    break
                time.sleep(0.02)
            self.assertIs(status, JobStatus.FAILED)
            self.assertIn("exit 3", output)


if __name__ == "__main__":
    unittest.main()

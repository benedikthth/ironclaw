"""End-to-end tests for the PhD vertical slice (stdlib unittest, no deps)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw import demo  # noqa: E402
from ironclaw.agents.phd import run_phd  # noqa: E402
from ironclaw.contracts import AcceptanceCriterion, CheckKind, Task, TaskStatus  # noqa: E402
from ironclaw.providers.base import AssistantTurn, ToolCall  # noqa: E402
from ironclaw.providers.fake import FakeProvider  # noqa: E402


class TestPhdSlice(unittest.TestCase):
    def test_happy_path_passes_and_reuses_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = demo.run(tmp)
        self.assertEqual(result.status, TaskStatus.PASSED)
        self.assertTrue(all(c.passed for c in result.checks))
        self.assertEqual({a.path for a in result.artifacts}, {"region_totals.csv", "report.md"})

    def test_failed_verification_triggers_retry_then_passes(self):
        # First submit forgets to create the file; harness must report the failing
        # check and let the PhD fix it on a later turn.
        skills_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
        task = Task(
            goal="Create result.txt containing OK.",
            acceptance=[
                AcceptanceCriterion("result.txt exists", CheckKind.FILE_EXISTS, "result.txt"),
                AcceptanceCriterion(
                    "contains OK", CheckKind.COMMAND, "grep -q OK result.txt"
                ),
            ],
        )
        script = [
            # Premature submission — nothing written yet.
            AssistantTurn(tool_calls=[ToolCall("a", "submit_result", {"summary": "done", "artifacts": []})]),
            # After failure feedback, actually do the work.
            AssistantTurn(tool_calls=[ToolCall("b", "write_file", {"path": "result.txt", "content": "OK\n"})]),
            AssistantTurn(tool_calls=[ToolCall("c", "submit_result", {"summary": "done", "artifacts": [{"path": "result.txt", "kind": "data"}]})]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_phd(
                task=task,
                provider=FakeProvider(script),
                workspace=tmp,
                skills_root=skills_root,
            )
        self.assertEqual(outcome.result.status, TaskStatus.PASSED)
        self.assertEqual(outcome.result.iterations, 3)

    def test_iteration_budget_escalates(self):
        from ironclaw.runtime.loop import LoopConfig

        task = Task(
            goal="Impossible under budget.",
            acceptance=[AcceptanceCriterion("never", CheckKind.FILE_EXISTS, "nope.txt")],
        )
        # Model keeps emitting no-op text; loop must terminate as ESCALATED.
        script = [AssistantTurn(text="thinking...") for _ in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_phd(
                task=task,
                provider=FakeProvider(script),
                workspace=tmp,
                config=LoopConfig(max_iterations=3),
            )
        self.assertEqual(outcome.result.status, TaskStatus.ESCALATED)

    def test_workspace_escape_is_blocked(self):
        from ironclaw.tools.base import ToolContext, ToolError
        from ironclaw.tools.files import WriteFile

        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(workspace=tmp)
            with self.assertRaises(ToolError):
                WriteFile().run({"path": "../escape.txt", "content": "x"}, ctx)


if __name__ == "__main__":
    unittest.main()

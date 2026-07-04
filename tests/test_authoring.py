"""Tests for the LLM-authoring seams (deterministic stand-ins for the LLM)."""

import os
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import ProjectSpec, run_lab  # noqa: E402
from ironclaw.agents.senior import supervise  # noqa: E402
from ironclaw.authoring import author_problem  # noqa: E402
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.observability import Recorder  # noqa: E402


def _cat(*ids) -> AgentCatalog:
    specs = [
        AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1),
        AgentSpec("moderate", "anthropic", "claude-sonnet-5", 2),
    ]
    return AgentCatalog(specs, {Role.PHD: list(ids)})


class FakeAuthor:
    def reformulate(self, task, diagnosis):
        return replace(task, goal="reformulated")

    def split(self, task, diagnosis):
        return [
            Task(goal="part one", acceptance=[], id="sub1"),
            Task(goal="part two", acceptance=[], id="sub2"),
        ]


class TestDecomposition(unittest.TestCase):
    def test_author_problem_turns_a_sentence_into_a_lab(self):
        def decomposer(statement):
            return [ProjectSpec("p1", "do part 1", [Task(goal="t", acceptance=[], id="t1")])]

        rec = Recorder()
        from ironclaw.observability import using

        with using(rec):
            problem = author_problem("Solve world hunger, tastefully.", decomposer)
        self.assertEqual(problem.statement, "Solve world hunger, tastefully.")
        self.assertEqual([p.id for p in problem.projects], ["p1"])
        self.assertIn("pi.decompose", [e.kind for e in rec.events])


class TestReformulate(unittest.TestCase):
    def test_reformulate_rewrites_the_task_then_it_passes(self):
        # Only 'cheapo' whitelisted -> no stronger agent -> BUDGET_EXHAUSTED maps
        # to REFORMULATE. The author rewrites the goal; the rewrite passes.
        def runner(task, agent, budget, ws):
            ok = task.goal == "reformulated"
            return TaskResult(task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED, iterations=budget)

        rec = Recorder()
        from ironclaw.observability import using

        with tempfile.TemporaryDirectory() as tmp, using(rec):
            res = supervise(
                task=Task(goal="original", acceptance=[], id="t"),
                catalog=_cat("cheapo"), runner=runner, workspace=tmp,
                phd_agent_id="cheapo", author=FakeAuthor(), max_relaunches=2,
            )
        self.assertIs(res.status, TaskStatus.PASSED)
        self.assertIn("reformulate", [e.kind for e in rec.events])


class TestSplit(unittest.TestCase):
    def _runner(self, subtasks_pass: bool):
        def runner(task, agent, budget, ws):
            if task.id in ("sub1", "sub2"):
                ok = subtasks_pass or task.id == "sub1"  # sub1 always ok; sub2 gated
                return TaskResult(task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED,
                                  failure_kind=None if ok else "postdoc_rejected", iterations=budget)
            # the parent task always comes back as a postdoc rejection -> SPLIT
            return TaskResult(task.id, TaskStatus.ESCALATED, failure_kind="postdoc_rejected", iterations=budget)

        return runner

    def test_split_into_subtasks_all_pass(self):
        rec = Recorder()
        from ironclaw.observability import using

        with tempfile.TemporaryDirectory() as tmp, using(rec):
            res = supervise(
                task=Task(goal="too big", acceptance=[], id="big"),
                catalog=_cat("cheapo"), runner=self._runner(True), workspace=tmp,
                phd_agent_id="cheapo", author=FakeAuthor(), max_relaunches=1,
            )
        self.assertIs(res.status, TaskStatus.PASSED)
        self.assertEqual([s.task_id for s in res.subtasks], ["sub1", "sub2"])
        self.assertIn("split", [e.kind for e in rec.events])

    def test_split_escalates_to_pi_when_a_subtask_is_stuck(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = supervise(
                task=Task(goal="too big", acceptance=[], id="big"),
                catalog=_cat("cheapo"), runner=self._runner(False), workspace=tmp,
                phd_agent_id="cheapo", author=FakeAuthor(), max_relaunches=1,
            )
        self.assertIs(res.status, TaskStatus.ESCALATED)
        self.assertIsNotNone(res.escalation)
        self.assertIs(res.escalation.to_role, Role.PI)
        # sub2 could not be solved and was not split again (depth bound).
        stuck = [s for s in res.subtasks if s.status is TaskStatus.ESCALATED]
        self.assertEqual([s.task_id for s in stuck], ["sub2"])


if __name__ == "__main__":
    unittest.main()

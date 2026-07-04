"""Tests for the Senior supervision loop (deterministic runner, no LLM)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.senior import supervise  # noqa: E402
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import (  # noqa: E402
    AgentCatalog,
    AgentSpec,
    Disposition,
    Role,
)

TASK = Task(goal="do a thing", acceptance=[])


def _catalog() -> AgentCatalog:
    agents = [
        AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", cost_tier=1),
        AgentSpec("moderate", "anthropic", "claude-sonnet-5", cost_tier=2),
    ]
    return AgentCatalog(agents, {Role.PHD: ["cheapo", "moderate"]})


class TestSupervise(unittest.TestCase):
    def test_passes_first_try(self):
        def runner(task, agent, budget, ws):
            return TaskResult(task.id, TaskStatus.PASSED, iterations=3)

        with tempfile.TemporaryDirectory() as tmp:
            res = supervise(
                task=TASK, catalog=_catalog(), runner=runner, workspace=tmp, phd_agent_id="cheapo"
            )
        self.assertIs(res.status, TaskStatus.PASSED)
        self.assertEqual([a.agent_id for a in res.attempts], ["cheapo"])
        self.assertIsNone(res.escalation)

    def test_cheap_fails_then_relaunches_stronger_and_passes(self):
        def runner(task, agent, budget, ws):
            # cheapo can't do it; the stronger agent can.
            if agent.id == "cheapo":
                return TaskResult(task.id, TaskStatus.ESCALATED, iterations=budget)
            return TaskResult(task.id, TaskStatus.PASSED, iterations=5)

        with tempfile.TemporaryDirectory() as tmp:
            res = supervise(
                task=TASK, catalog=_catalog(), runner=runner, workspace=tmp, phd_agent_id="cheapo"
            )
        self.assertIs(res.status, TaskStatus.PASSED)
        self.assertEqual([a.agent_id for a in res.attempts], ["cheapo", "moderate"])
        self.assertIs(res.attempts[0].disposition, Disposition.RELAUNCH_STRONGER)
        # The relaunch got a larger budget than the first attempt.
        self.assertGreater(res.attempts[1].budget, res.attempts[0].budget)

    def test_persistent_failure_is_bounded_and_escalates_to_pi(self):
        calls = []

        def runner(task, agent, budget, ws):
            calls.append(agent.id)
            return TaskResult(task.id, TaskStatus.ESCALATED, iterations=budget)

        with tempfile.TemporaryDirectory() as tmp:
            res = supervise(
                task=TASK,
                catalog=_catalog(),
                runner=runner,
                workspace=tmp,
                phd_agent_id="cheapo",
                max_relaunches=2,
            )
        # Bounded: cheapo -> moderate -> moderate, then up to the PI.
        self.assertIs(res.status, TaskStatus.ESCALATED)
        self.assertIsNotNone(res.escalation)
        self.assertIs(res.escalation.to_role, Role.PI)
        self.assertIs(res.escalation.diagnosis.suggested_disposition, Disposition.ESCALATE_TO_PI)
        self.assertEqual(len(res.attempts), 3)
        self.assertEqual(calls, ["cheapo", "moderate", "moderate"])

    def test_attempts_share_the_task_workspace(self):
        # A relaunch on a stronger agent should build on what the weaker one
        # produced, so all attempts run in the same (shared) workspace.
        seen = []

        def runner(task, agent, budget, ws):
            seen.append(ws)
            return TaskResult(task.id, TaskStatus.ESCALATED, iterations=budget)

        with tempfile.TemporaryDirectory() as tmp:
            supervise(
                task=TASK, catalog=_catalog(), runner=runner, workspace=tmp, phd_agent_id="cheapo"
            )
        self.assertTrue(len(seen) > 1)
        self.assertEqual(set(seen), {tmp})  # all the same shared dir


if __name__ == "__main__":
    unittest.main()

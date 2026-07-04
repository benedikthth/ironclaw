"""Tests for the PI lab lifecycle (deterministic runner, no LLM)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import Problem, ProjectSpec, run_lab  # noqa: E402
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402


def _catalog() -> AgentCatalog:
    return AgentCatalog(
        [
            AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1),
            AgentSpec("moderate", "anthropic", "claude-sonnet-5", 2),
        ],
        {Role.PHD: ["cheapo", "moderate"]},
    )


def _problem() -> Problem:
    return Problem(
        statement="Solve the big thing.",
        projects=[
            ProjectSpec("proj_a", "part A", [Task(goal="a1", acceptance=[], id="t_a1")]),
            ProjectSpec(
                "proj_b",
                "part B",
                [Task(goal="b1", acceptance=[], id="t_b1"), Task(goal="b2", acceptance=[], id="t_b2")],
            ),
        ],
    )


class TestRunLab(unittest.TestCase):
    def test_all_pass_lab_passes(self):
        def runner(task, agent, budget, ws):
            return TaskResult(task.id, TaskStatus.PASSED, iterations=3)

        with tempfile.TemporaryDirectory() as tmp:
            lab = run_lab(
                problem=_problem(),
                catalog=_catalog(),
                runner=runner,
                workspace=tmp,
                phd_agent_id="cheapo",
            )
        self.assertIs(lab.status, TaskStatus.PASSED)
        self.assertEqual([p.status for p in lab.projects], [TaskStatus.PASSED, TaskStatus.PASSED])
        self.assertEqual(lab.escalations(), [])

    def test_one_task_stuck_bubbles_to_lab_escalation(self):
        # Task b2 can never pass -> its Senior escalates to the PI -> project b
        # fails -> the lab is ESCALATED, and the stuck task shows up in the PI's
        # action list.
        def runner(task, agent, budget, ws):
            ok = task.id != "t_b2"
            return TaskResult(
                task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED, iterations=budget
            )

        with tempfile.TemporaryDirectory() as tmp:
            lab = run_lab(
                problem=_problem(),
                catalog=_catalog(),
                runner=runner,
                workspace=tmp,
                phd_agent_id="cheapo",
                max_relaunches=1,
            )
        self.assertIs(lab.status, TaskStatus.ESCALATED)
        proj_status = {p.project_id: p.status for p in lab.projects}
        self.assertIs(proj_status["proj_a"], TaskStatus.PASSED)
        self.assertIs(proj_status["proj_b"], TaskStatus.ESCALATED)
        stuck = lab.escalations()
        self.assertEqual([s.task_id for s in stuck], ["t_b2"])
        self.assertIsNotNone(stuck[0].escalation)
        self.assertIs(stuck[0].escalation.to_role, Role.PI)


if __name__ == "__main__":
    unittest.main()

"""Tests for the TUI: the event-stream fold and the renderer."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import Problem, ProjectSpec, run_lab  # noqa: E402
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.observability import Recorder  # noqa: E402
from ironclaw.tui import LabModel, LiveDashboard, build_model, render  # noqa: E402


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
        "Characterize and stress-test.",
        [
            ProjectSpec("chars", "measure", [Task(goal="m", acceptance=[], id="measure", project_id="chars")]),
            ProjectSpec(
                "stress", "push",
                [
                    Task(goal="hard", acceptance=[], id="hard", project_id="stress"),
                    Task(goal="imp", acceptance=[], id="impossible", project_id="stress"),
                ],
            ),
        ],
    )


class TestFold(unittest.TestCase):
    def test_dashboard_reconstructs_the_lab_tree(self):
        # measure passes; hard needs 'moderate' (relaunch ladder); impossible never.
        def runner(task, agent, budget, ws):
            ok = {"measure": True, "hard": agent.id == "moderate", "impossible": False}[task.id]
            return TaskResult(task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED, iterations=budget)

        dash = LiveDashboard()
        rec = Recorder(sinks=[dash.as_sink()])
        with tempfile.TemporaryDirectory() as tmp:
            run_lab(problem=_problem(), catalog=_catalog(), runner=runner, workspace=tmp,
                    phd_agent_id="cheapo", max_relaunches=1, recorder=rec)
        m = dash.model
        self.assertEqual(m.status, "escalated")
        self.assertEqual(m.projects["chars"].status, "passed")
        self.assertEqual(m.projects["stress"].status, "escalated")
        self.assertEqual(m.tasks["measure"].status, "passed")
        self.assertEqual(m.tasks["hard"].status, "passed")
        self.assertEqual(m.tasks["hard"].agents, ["cheapo", "moderate"])  # the ladder
        self.assertEqual(m.tasks["impossible"].status, "escalated")
        self.assertIn(("impossible", "accept"), m.pi_actions)
        self.assertEqual(m.tasks["hard"].project_id, "stress")

    def test_build_model_from_replayed_events_matches_live(self):
        def runner(task, agent, budget, ws):
            return TaskResult(task.id, TaskStatus.PASSED, iterations=3)

        rec = Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            run_lab(problem=_problem(), catalog=_catalog(), runner=runner, workspace=tmp,
                    phd_agent_id="cheapo", recorder=rec)
        m = build_model(rec.events)  # replay the log
        self.assertEqual(m.status, "passed")
        self.assertTrue(all(t.status == "passed" for t in m.tasks.values()))


class TestRender(unittest.TestCase):
    def test_render_nests_tasks_under_projects_and_shows_cost(self):
        m = LabModel(statement="S", status="passed")
        from ironclaw.observability import Event

        for ev in [
            Event("project.start", project_id="p1", message="do it"),
            Event("attempt", task_id="t1", project_id="p1", agent_id="cheapo"),
            Event("task.done", task_id="t1", project_id="p1", data={"status": "passed"}),
            Event("usage", data={"input_tokens": 120, "output_tokens": 30}),
        ]:
            m.apply(ev)
        out = render(m)
        self.assertIn("project p1", out)
        self.assertIn("t1", out)
        self.assertIn("via cheapo", out)
        self.assertIn("120 in / 30 out", out)

    def test_usage_folds_into_cost(self):
        from ironclaw.observability import Event

        m = LabModel()
        m.apply(Event("usage", data={"input_tokens": 10, "output_tokens": 2}))
        m.apply(Event("usage", data={"input_tokens": 5, "output_tokens": 1}))
        self.assertEqual(m.usage, {"input_tokens": 15, "output_tokens": 3, "turns": 2})


if __name__ == "__main__":
    unittest.main()

"""Tests for shared project workspace (#1) and task dependencies (#2)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import Problem, ProjectSpec, run_lab  # noqa: E402
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.observability import Recorder  # noqa: E402


def _catalog() -> AgentCatalog:
    return AgentCatalog(
        [AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1)], {Role.PHD: ["cheapo"]}
    )


class TestDependencies(unittest.TestCase):
    def test_dependency_ordering_runs_producer_before_consumer(self):
        # 'verify' is declared first but depends on 'compute' -> compute runs first.
        problem = Problem(
            "P",
            [ProjectSpec("p", "g", [
                Task(goal="verify", acceptance=[], id="verify", project_id="p", depends_on=["compute"]),
                Task(goal="compute", acceptance=[], id="compute", project_id="p"),
            ])],
        )
        order = []

        def runner(task, agent, budget, ws):
            order.append(task.id)
            return TaskResult(task.id, TaskStatus.PASSED, iterations=3)

        with tempfile.TemporaryDirectory() as tmp:
            run_lab(problem=problem, catalog=_catalog(), runner=runner, workspace=tmp, phd_agent_id="cheapo")
        self.assertEqual(order, ["compute", "verify"])

    def test_shared_workspace_lets_a_task_see_a_dependency_artifact(self):
        # 'compute' writes a file; 'verify' passes only if it can read it -> proves
        # the two tasks share a workspace (the original live-run bug, fixed).
        problem = Problem(
            "P",
            [ProjectSpec("p", "g", [
                Task(goal="compute", acceptance=[], id="compute", project_id="p"),
                Task(goal="verify", acceptance=[], id="verify", project_id="p", depends_on=["compute"]),
            ])],
        )

        def runner(task, agent, budget, ws):
            if task.id == "compute":
                with open(os.path.join(ws, "result.txt"), "w") as fh:
                    fh.write("mean=50.5\n")
                return TaskResult(task.id, TaskStatus.PASSED, iterations=3)
            # 'verify' can only pass if compute's file is visible in the shared ws
            ok = os.path.exists(os.path.join(ws, "result.txt"))
            return TaskResult(task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED, iterations=budget)

        with tempfile.TemporaryDirectory() as tmp:
            lab = run_lab(problem=problem, catalog=_catalog(), runner=runner, workspace=tmp, phd_agent_id="cheapo")
        self.assertIs(lab.status, TaskStatus.PASSED)

    def test_task_is_blocked_when_a_dependency_fails(self):
        problem = Problem(
            "P",
            [ProjectSpec("p", "g", [
                Task(goal="compute", acceptance=[], id="compute", project_id="p"),
                Task(goal="verify", acceptance=[], id="verify", project_id="p", depends_on=["compute"]),
            ])],
        )
        ran = []

        def runner(task, agent, budget, ws):
            ran.append(task.id)
            return TaskResult(task.id, TaskStatus.ESCALATED, iterations=budget)  # compute fails

        rec = Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            lab = run_lab(problem=problem, catalog=_catalog(), runner=runner, workspace=tmp,
                          phd_agent_id="cheapo", max_relaunches=0, recorder=rec)
        # 'verify' never ran — its dependency didn't pass.
        self.assertEqual(ran, ["compute"])
        self.assertIn("task.blocked", [e.kind for e in rec.events])
        blocked = [e for e in rec.events if e.kind == "task.blocked"][0]
        self.assertEqual(blocked.data["unmet"], ["compute"])
        self.assertIs(lab.status, TaskStatus.ESCALATED)


if __name__ == "__main__":
    unittest.main()

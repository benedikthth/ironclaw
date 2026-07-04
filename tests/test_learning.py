"""Tests for the self-improving skill loop (learn from success)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import Problem, ProjectSpec, run_lab  # noqa: E402
from ironclaw.agents.senior import SupervisionResult  # noqa: E402
from ironclaw.contracts import Artifact, Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.infra import InfrastructureManager  # noqa: E402
from ironclaw.learning import SkillLearner  # noqa: E402
from ironclaw.observability import Recorder, active_recorder  # noqa: E402

SKILL = {"worth_keeping": True, "name": "aggregate_csv",
         "description": "Aggregate a CSV column by a key.",
         "body": "# Aggregate\n\nRead the CSV, group by key, sum the value column.\n"}


class _FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def structured(self, system, prompt, schema, *, max_tokens=2048):
        self.calls += 1
        return self.response


def _im() -> InfrastructureManager:
    return InfrastructureManager(os.path.join(tempfile.mkdtemp(), "skills"))


def _passed(task_id="t", artifacts=None) -> SupervisionResult:
    return SupervisionResult(
        task_id, TaskStatus.PASSED,
        TaskResult(task_id, TaskStatus.PASSED, artifacts=artifacts or [], summary="done"),
    )


class TestSkillLearner(unittest.TestCase):
    def test_learns_a_skill_from_a_nontrivial_success(self):
        im = _im()
        learner = SkillLearner(im, _FakeProvider(SKILL), min_tool_calls=5)
        with tempfile.TemporaryDirectory() as ws:
            with open(os.path.join(ws, "agg.py"), "w") as fh:
                fh.write("import csv\n# real procedure\n")
            task = Task(goal="aggregate sales.csv by region", acceptance=[], id="t")
            name = learner.consider(task, _passed("t", [Artifact("agg.py", "code")]), ws, tool_calls=7)
        self.assertEqual(name, "aggregate_csv")
        self.assertIn("aggregate_csv", im.list_skills())

    def test_below_threshold_learns_nothing(self):
        im = _im()
        learner = SkillLearner(im, _FakeProvider(SKILL), min_tool_calls=5)
        with tempfile.TemporaryDirectory() as ws:
            out = learner.consider(_passed().__class__ and Task(goal="g", acceptance=[], id="t"),
                                   _passed(), ws, tool_calls=2)
        self.assertIsNone(out)
        self.assertEqual(im.list_skills(), [])

    def test_worth_keeping_false_is_dropped(self):
        im = _im()
        prov = _FakeProvider({"worth_keeping": False, "name": "x", "description": "y", "body": "z" * 30})
        learner = SkillLearner(im, prov, min_tool_calls=1)
        with tempfile.TemporaryDirectory() as ws:
            out = learner.consider(Task(goal="g", acceptance=[], id="t"), _passed(), ws, tool_calls=9)
        self.assertIsNone(out)
        self.assertEqual(im.list_skills(), [])

    def test_failed_task_is_not_learned_from(self):
        im = _im()
        learner = SkillLearner(im, _FakeProvider(SKILL), min_tool_calls=1)
        sup = SupervisionResult("t", TaskStatus.ESCALATED, TaskResult("t", TaskStatus.ESCALATED))
        with tempfile.TemporaryDirectory() as ws:
            self.assertIsNone(learner.consider(Task(goal="g", acceptance=[], id="t"), sup, ws, tool_calls=9))


class TestLearningWiring(unittest.TestCase):
    def test_run_lab_learns_from_an_effortful_task(self):
        im = _im()
        learner = SkillLearner(im, _FakeProvider(SKILL), min_tool_calls=5)
        problem = Problem(
            "P", [ProjectSpec("p", "g", [Task(goal="aggregate the data", acceptance=[], id="t1", project_id="p")])]
        )

        def runner(task, agent, budget, ws):
            # Simulate a real PhD doing real work: emit tool.call events + an artifact.
            for _ in range(6):
                active_recorder().emit("tool.call", task_id=task.id)
            with open(os.path.join(ws, "out.py"), "w") as fh:
                fh.write("print('ok')\n")
            return TaskResult(task.id, TaskStatus.PASSED, artifacts=[Artifact("out.py", "code")], summary="done")

        rec = Recorder()
        catalog = AgentCatalog([AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1)], {Role.PHD: ["cheapo"]})
        with tempfile.TemporaryDirectory() as tmp:
            run_lab(problem=problem, catalog=catalog, runner=runner, workspace=tmp,
                    phd_agent_id="cheapo", recorder=rec, learner=learner)
        self.assertIn("aggregate_csv", im.list_skills())
        self.assertIn("skill.learned", [e.kind for e in rec.events])


if __name__ == "__main__":
    unittest.main()

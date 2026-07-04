"""Tests for the event stream, cost aggregation, and the PI overseer seam."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import (  # noqa: E402
    AutoOverseer,
    PIDecision,
    Problem,
    ProjectSpec,
    run_lab,
)
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.observability import (  # noqa: E402
    Recorder,
    active_recorder,
    jsonl_sink,
    total_usage,
    usage_by_agent,
    using,
)


def _catalog() -> AgentCatalog:
    return AgentCatalog(
        [
            AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1),
            AgentSpec("moderate", "anthropic", "claude-sonnet-5", 2),
        ],
        {Role.PHD: ["cheapo", "moderate"]},
    )


class TestEventStream(unittest.TestCase):
    def test_lab_run_emits_a_coherent_stream(self):
        problem = Problem(
            statement="P",
            projects=[ProjectSpec("proj", "g", [Task(goal="t", acceptance=[], id="t1")])],
        )
        rec = Recorder()

        def runner(task, agent, budget, ws):
            return TaskResult(task.id, TaskStatus.PASSED, iterations=3)

        with tempfile.TemporaryDirectory() as tmp:
            run_lab(
                problem=problem, catalog=_catalog(), runner=runner, workspace=tmp,
                phd_agent_id="cheapo", recorder=rec,
            )
        kinds = [e.kind for e in rec.events]
        for expected in ["lab.start", "project.start", "attempt", "project.result", "lab.result"]:
            self.assertIn(expected, kinds)

    def test_recorder_is_ambient_within_using(self):
        rec = Recorder()
        with using(rec):
            active_recorder().emit("x")
        self.assertEqual([e.kind for e in rec.events], ["x"])
        # Outside the block the ambient recorder is the null one (no storage).
        self.assertIsNot(active_recorder(), rec)

    def test_jsonl_sink_writes_one_line_per_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            rec = Recorder(sinks=[jsonl_sink(path)])
            rec.emit("a", task_id="t")
            rec.emit("b", data={"n": 1})
            with open(path, encoding="utf-8") as fh:
                lines = [json.loads(ln) for ln in fh]
        self.assertEqual([l["kind"] for l in lines], ["a", "b"])
        self.assertEqual(lines[1]["data"], {"n": 1})


class TestCostAggregation(unittest.TestCase):
    def test_totals_and_by_agent(self):
        rec = Recorder()
        rec.emit("usage", agent_id="cheapo", data={"input_tokens": 100, "output_tokens": 20})
        rec.emit("usage", agent_id="cheapo", data={"input_tokens": 50, "output_tokens": 10})
        rec.emit("usage", agent_id="moderate", data={"input_tokens": 200, "output_tokens": 40})
        rec.emit("task.result")  # non-usage events ignored

        self.assertEqual(
            total_usage(rec.events), {"input_tokens": 350, "output_tokens": 70, "turns": 3}
        )
        by = usage_by_agent(rec.events)
        self.assertEqual(by["cheapo"]["input_tokens"], 150)
        self.assertEqual(by["moderate"]["turns"], 1)


class TestOverseerInteraction(unittest.TestCase):
    def _stuck_problem(self):
        return Problem(
            statement="P",
            projects=[
                ProjectSpec(
                    "proj",
                    "g",
                    [Task(goal="a", acceptance=[], id="a"), Task(goal="b", acceptance=[], id="b")],
                )
            ],
        )

    def test_retry_then_accept_reruns_the_task(self):
        calls = {"a": 0}

        def runner(task, agent, budget, ws):
            if task.id == "a":
                calls["a"] += 1
            return TaskResult(task.id, TaskStatus.ESCALATED, iterations=budget)

        class RetryOnce:
            def __init__(self):
                self.n = 0

            def on_escalation(self, sup):
                self.n += 1
                return PIDecision.RETRY if self.n == 1 else PIDecision.ACCEPT

        rec = Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            run_lab(
                problem=self._stuck_problem(), catalog=_catalog(), runner=runner, workspace=tmp,
                phd_agent_id="cheapo", overseer=RetryOnce(), recorder=rec, max_relaunches=0,
            )
        # Task 'a' escalated, PI retried once (RETRY) then accepted -> two runs.
        decisions = [e.message for e in rec.events if e.kind == "pi.decision" and e.task_id == "a"]
        self.assertEqual(decisions, ["retry", "accept"])

    def test_abandon_skips_remaining_project_tasks(self):
        ran = []

        def runner(task, agent, budget, ws):
            ran.append(task.id)
            return TaskResult(task.id, TaskStatus.ESCALATED, iterations=budget)

        class Abandon:
            def on_escalation(self, sup):
                return PIDecision.ABANDON

        with tempfile.TemporaryDirectory() as tmp:
            lab = run_lab(
                problem=self._stuck_problem(), catalog=_catalog(), runner=runner, workspace=tmp,
                phd_agent_id="cheapo", overseer=Abandon(), max_relaunches=0,
            )
        self.assertEqual(ran, ["a"])  # 'b' never ran — project abandoned on 'a'
        self.assertIs(lab.status, TaskStatus.ESCALATED)
        self.assertEqual(len(lab.projects[0].tasks), 1)

    def test_auto_overseer_accepts(self):
        self.assertIs(
            AutoOverseer().on_escalation.__call__(None), PIDecision.ACCEPT  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    unittest.main()

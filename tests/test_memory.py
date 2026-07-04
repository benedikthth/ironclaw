"""Tests for cross-lab memory: recording, retrieval, persistence, wiring."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import Problem, ProjectSpec, run_lab  # noqa: E402
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.memory import MemoryStore  # noqa: E402


def _catalog() -> AgentCatalog:
    return AgentCatalog(
        [AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1)], {Role.PHD: ["cheapo"]}
    )


class TestMemoryStore(unittest.TestCase):
    def test_add_and_search_by_keyword(self):
        m = MemoryStore()
        m.add("finding", "computed the mean of a list of integers", ref={"task_id": "t1"})
        m.add("finding", "crawled a website and stored JSONL", ref={"task_id": "t2"})
        hits = m.search("mean of integers")
        self.assertEqual(hits[0].ref["task_id"], "t1")

    def test_search_empty_when_no_overlap(self):
        m = MemoryStore()
        m.add("finding", "trained a neural net", ref={})
        self.assertEqual(m.search("quantum chromodynamics"), [])

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mem.json")
            m1 = MemoryStore(path)
            m1.add("lab", "solve the widget problem", tags=["passed"])
            # A fresh store over the same file sees the record.
            m2 = MemoryStore(path)
            self.assertEqual(len(m2.records), 1)
            self.assertEqual(m2.records[0].text, "solve the widget problem")
            # seq continues, not restarts.
            self.assertEqual(m2.add("finding", "x").seq, 2)

    def test_context_for_builds_prior_work_block(self):
        m = MemoryStore()
        m.add("finding", "computed statistics for integers", tags=[])
        block = m.context_for("compute statistics")
        self.assertIn("Prior related work", block)
        self.assertIn("computed statistics", block)
        self.assertEqual(m.context_for("totally unrelated topic xyz"), "")


class TestMemoryWiring(unittest.TestCase):
    def test_run_lab_records_lab_and_findings(self):
        problem = Problem(
            "P", [ProjectSpec("p", "g", [Task(goal="do the thing", acceptance=[], id="t1", project_id="p")])]
        )

        def runner(task, agent, budget, ws):
            return TaskResult(task.id, TaskStatus.PASSED, summary="did the thing well", iterations=3)

        store = MemoryStore()
        with tempfile.TemporaryDirectory() as tmp:
            run_lab(problem=problem, catalog=_catalog(), runner=runner, workspace=tmp,
                    phd_agent_id="cheapo", memory=store)
        kinds = [r.kind for r in store.records]
        self.assertIn("finding", kinds)
        self.assertIn("lab", kinds)
        finding = next(r for r in store.records if r.kind == "finding")
        self.assertIn("do the thing", finding.text)

    def test_decomposer_injects_prior_work(self):
        class _Capturing:
            def __init__(self, response):
                self.response = response
                self.prompt = None

            def structured(self, system, prompt, schema, *, max_tokens=2048):
                self.prompt = prompt
                return self.response

        from ironclaw.authoring import llm_decomposer

        store = MemoryStore()
        store.add("finding", "computed the mean of integers 1..100", tags=[])
        payload = {"projects": [{"id": "p", "goal": "g", "tasks": [
            {"id": "t", "goal": "do", "acceptance": [], "depends_on": []}]}]}
        prov = _Capturing(payload)
        decompose = llm_decomposer(prov, memory=store)
        decompose("compute the mean of some integers")
        self.assertIn("Prior related work", prov.prompt)
        self.assertIn("computed the mean", prov.prompt)


if __name__ == "__main__":
    unittest.main()

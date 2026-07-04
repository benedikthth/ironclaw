"""Cross-lab memory + self-improving skills — what carries between labs.

Lab 1 runs, records its finding, and (because it took real effort) the institute
learns a reusable skill from the success. Then we show that a *second* lab would
start informed: the memory recalls the prior finding, and the learned skill is
now in the shared skills folder. Deterministic, no API key (a fake provider
stands in for the skill-distilling LLM).

    python -m ironclaw.demo_memory
"""

from __future__ import annotations

import os
import tempfile

from .agents.pi import Problem, ProjectSpec, run_lab
from .contracts import Artifact, Task, TaskResult, TaskStatus
from .control import AgentCatalog, AgentSpec, Role
from .infra import InfrastructureManager
from .learning import SkillLearner
from .memory import MemoryStore
from .observability import Recorder, console_sink, using


class _FakeDistiller:
    def structured(self, system, prompt, schema, *, max_tokens=2048):
        return {
            "worth_keeping": True,
            "name": "aggregate by key",
            "description": "Aggregate a CSV numeric column grouped by a key column.",
            "body": "# Aggregate by key\n\nRead the CSV, group rows by the key column, "
                    "sum the value column, write a two-column result. Generalizes the "
                    "sales-by-region pattern.\n",
        }


def main() -> None:
    catalog = AgentCatalog(
        [AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1)], {Role.PHD: ["cheapo"]}
    )

    def runner(task, agent, budget, ws):
        from .observability import active_recorder
        for _ in range(6):  # a real PhD's tool calls (over the learning threshold)
            active_recorder().emit("tool.call", task_id=task.id)
        with open(os.path.join(ws, "totals.py"), "w") as fh:
            fh.write("import csv, collections  # aggregation procedure\n")
        return TaskResult(task.id, TaskStatus.PASSED, artifacts=[Artifact("totals.py", "code")],
                          summary="Aggregated sales.csv by region.")

    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryStore(os.path.join(tmp, "memory.json"))
        im = InfrastructureManager(os.path.join(tmp, "skills"))
        learner = SkillLearner(im, _FakeDistiller(), min_tool_calls=5)
        rec = Recorder(sinks=[console_sink()])

        lab1 = Problem("Aggregate the regional sales data.", [
            ProjectSpec("agg", "aggregate", [
                Task(goal="aggregate sales.csv by region", acceptance=[], id="agg1", project_id="agg")])])

        print("=== LAB 1 (learns + remembers) ===")
        with using(rec):
            run_lab(problem=lab1, catalog=catalog, runner=runner, workspace=tmp,
                    phd_agent_id="cheapo", memory=memory, learner=learner, recorder=rec)

        print("\n=== what carried across to future labs ===")
        print("memory records:", [(r.kind, r.text.split(chr(10))[0][:40]) for r in memory.records])
        print("learned skills:", im.list_skills())
        print("\nLAB 2 would open its PI decomposition with:")
        print(memory.context_for("aggregate quarterly sales by region"))


if __name__ == "__main__":
    main()

"""Full-institute demo: PI -> Senior -> (Postdoc) -> PhD, top to bottom.

A deterministic runner stands in for the PhD so the whole lifecycle is visible
with no API key. It shows all three behaviors in one lab: a task solved outright,
a task the Senior relaunches onto a stronger agent, and a task nobody can solve
that bubbles all the way up to the PI's action list.

    python -m ironclaw.demo_lab
"""

from __future__ import annotations

import tempfile

from .agents.pi import Problem, ProjectSpec, run_lab
from .contracts import Task, TaskResult, TaskStatus
from .control import AgentCatalog, AgentSpec, Role


def main() -> None:
    catalog = AgentCatalog(
        [
            AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1),
            AgentSpec("moderate", "anthropic", "claude-sonnet-5", 2),
        ],
        {Role.PHD: ["cheapo", "moderate"]},
    )

    problem = Problem(
        statement="Characterize the widget and stress-test the impossible case.",
        projects=[
            ProjectSpec("characterize", "measure it", [Task(goal="measure", acceptance=[], id="measure")]),
            ProjectSpec(
                "stress",
                "push the limits",
                [
                    Task(goal="hard but doable", acceptance=[], id="hard"),
                    Task(goal="impossible", acceptance=[], id="impossible"),
                ],
            ),
        ],
    )

    # measure: any agent. hard: needs 'moderate'. impossible: nobody.
    def runner(task, agent, budget, ws):
        ok = {
            "measure": True,
            "hard": agent.id == "moderate",
            "impossible": False,
        }[task.id]
        return TaskResult(
            task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED, iterations=budget
        )

    with tempfile.TemporaryDirectory() as tmp:
        lab = run_lab(
            problem=problem,
            catalog=catalog,
            runner=runner,
            workspace=tmp,
            phd_agent_id="cheapo",
            max_relaunches=1,
        )

    print(f"LAB: {lab.statement}")
    print(f"status: {lab.status.value}\n")
    for proj in lab.projects:
        print(f"  project {proj.project_id} [{proj.status.value}]")
        for sup in proj.tasks:
            ladder = " -> ".join(a.agent_id for a in sup.attempts)
            print(f"    task {sup.task_id:11s} [{sup.status.value:9s}] via {ladder}")
    stuck = lab.escalations()
    if stuck:
        print("\nPI action list (Seniors gave up):")
        for s in stuck:
            print(f"  - {s.task_id}: {s.escalation.diagnosis.suggested_disposition.value}")


if __name__ == "__main__":
    main()

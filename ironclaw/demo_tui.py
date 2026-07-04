"""Sentence -> lab -> rendered tree: the authoring seam + the TUI, together.

A deterministic decomposer stands in for the LLM PI (swap in
``authoring.llm_decomposer`` for the real thing), and a deterministic runner
stands in for the PhDs, so the whole flow runs with no API key. The lab is folded
into a live model via the dashboard sink and rendered as a tree.

    python -m ironclaw.demo_tui
"""

from __future__ import annotations

import tempfile

from .agents.pi import Problem, ProjectSpec, run_lab
from .authoring import author_problem
from .contracts import Task, TaskResult, TaskStatus
from .control import AgentCatalog, AgentSpec, Role
from .observability import Recorder, using
from .tui import LiveDashboard, render


def fake_decomposer(statement: str):
    # An LLM PI would author these from the statement; here they're fixed.
    return [
        ProjectSpec("characterize", "measure the widget",
                    [Task(goal="measure throughput", acceptance=[], id="measure", project_id="characterize")]),
        ProjectSpec("stress", "find the breaking point", [
            Task(goal="load test to 2x", acceptance=[], id="loadtest", project_id="stress"),
            Task(goal="prove the impossible bound", acceptance=[], id="impossible", project_id="stress"),
        ]),
    ]


def main() -> None:
    catalog = AgentCatalog(
        [
            AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1),
            AgentSpec("moderate", "anthropic", "claude-sonnet-5", 2),
        ],
        {Role.PHD: ["cheapo", "moderate"]},
    )

    def runner(task, agent, budget, ws):
        ok = {"measure": True, "loadtest": agent.id == "moderate", "impossible": False}[task.id]
        return TaskResult(task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED, iterations=budget)

    dash = LiveDashboard()  # LiveDashboard(live=True) redraws the terminal each event
    rec = Recorder(sinks=[dash.as_sink()])

    with tempfile.TemporaryDirectory() as tmp, using(rec):
        problem: Problem = author_problem(
            "Characterize the widget and stress-test the impossible case.", fake_decomposer
        )
        run_lab(
            problem=problem, catalog=catalog, runner=runner, workspace=tmp,
            phd_agent_id="cheapo", max_relaunches=1, recorder=rec,
        )

    print(render(dash.model))


if __name__ == "__main__":
    main()

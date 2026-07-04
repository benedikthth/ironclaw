"""Senior supervision demo: the failure loop, closed.

A deterministic runner stands in for the PhD so the control flow is visible with
no API key: the cheap agent fails, the Senior relaunches on a stronger
whitelisted agent with more budget, and it passes. Swap the runner for
`anthropic_phd_runner(...)` to drive real PhDs.

    python -m ironclaw.demo_senior
"""

from __future__ import annotations

import tempfile

from .agents.senior import supervise
from .contracts import Task, TaskResult, TaskStatus
from .control import AgentCatalog, AgentSpec, Role


def main() -> None:
    catalog = AgentCatalog(
        [
            AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", cost_tier=1),
            AgentSpec("moderate", "anthropic", "claude-sonnet-5", cost_tier=2),
            AgentSpec("strong", "anthropic", "claude-opus-4-8", cost_tier=3),
        ],
        {Role.PHD: ["cheapo", "moderate", "strong"]},
    )

    # The task is only solved once the Senior reaches the 'strong' agent.
    def runner(task, agent, budget, ws):
        ok = agent.id == "strong"
        return TaskResult(
            task.id, TaskStatus.PASSED if ok else TaskStatus.ESCALATED, iterations=budget
        )

    task = Task(goal="A hard task only the strong agent can finish.", acceptance=[])
    with tempfile.TemporaryDirectory() as tmp:
        res = supervise(
            task=task, catalog=catalog, runner=runner, workspace=tmp, phd_agent_id="cheapo"
        )

    print("Supervision ladder:")
    for i, a in enumerate(res.attempts):
        d = f" -> {a.disposition.value}" if a.disposition else ""
        print(f"  attempt {i}: agent={a.agent_id:9s} budget={a.budget:3d} {a.status.value}{d}")
    print(f"\nfinal: {res.status.value}")
    if res.escalation:
        print(f"escalated to {res.escalation.to_role.value}: {res.escalation.diagnosis.notes}")


if __name__ == "__main__":
    main()

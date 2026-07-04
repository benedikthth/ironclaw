"""A terminal UI over the observability stream — watch and steer a live lab.

The whole UI is a *fold* over events: ``LabModel`` accumulates the event stream
into a lab -> project -> task tree with per-task ladders, review verdicts, cost,
and the PI action list. ``render`` turns that model into text. ``LiveDashboard``
is a sink that re-renders on each event. Because it's a pure fold + pure render,
it is fully testable without a terminal — and a web/Telegram UI is the same fold
over the same stream, just a different renderer.

``TUIOverseer`` is the steering half: the PI consults it on an escalation, and it
shows the live tree before prompting for accept / retry / abandon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .agents.pi import PIDecision
from .observability import Event, Recorder, Sink

_ICON = {"passed": "✓", "escalated": "!", "running": "·", "abandoned": "✗"}


@dataclass
class TaskView:
    task_id: str
    project_id: str | None = None
    status: str = "running"
    agents: list[str] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)
    note: str = ""  # reformulate / split marker


@dataclass
class ProjectView:
    project_id: str
    goal: str = ""
    status: str = "running"


@dataclass
class LabModel:
    statement: str = ""
    status: str = "running"
    projects: dict[str, ProjectView] = field(default_factory=dict)
    tasks: dict[str, TaskView] = field(default_factory=dict)
    task_order: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "turns": 0})
    pi_actions: list[tuple[str, str]] = field(default_factory=list)

    def _task(self, task_id: str, project_id: str | None = None) -> TaskView:
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskView(task_id, project_id)
            self.task_order.append(task_id)
        tv = self.tasks[task_id]
        if project_id and not tv.project_id:
            tv.project_id = project_id
        return tv

    def apply(self, ev: Event) -> None:
        k = ev.kind
        if k in ("lab.start", "pi.decompose"):
            self.statement = ev.lab or self.statement
        elif k == "project.start":
            self.projects[ev.project_id] = ProjectView(ev.project_id, goal=ev.message)
        elif k == "project.result":
            if ev.project_id in self.projects:
                self.projects[ev.project_id].status = ev.message
        elif k == "lab.result":
            self.status = ev.message
        elif k == "attempt":
            self._task(ev.task_id, ev.project_id).agents.append(ev.agent_id or "?")
        elif k == "review":
            self._task(ev.task_id, ev.project_id).reviews.append(ev.message)
        elif k in ("reformulate", "split"):
            self._task(ev.task_id, ev.project_id).note = k
        elif k == "task.done":
            self._task(ev.task_id, ev.project_id).status = ev.data.get("status", ev.message)
        elif k == "pi.decision":
            self.pi_actions.append((ev.task_id or "?", ev.message))
        elif k == "usage":
            self.usage["input_tokens"] += int(ev.data.get("input_tokens", 0))
            self.usage["output_tokens"] += int(ev.data.get("output_tokens", 0))
            self.usage["turns"] += 1


def render(model: LabModel) -> str:
    lines = [f"LAB [{_ICON.get(model.status, '·')} {model.status}] {model.statement}"]

    def render_task(tv: TaskView, indent: str) -> None:
        ladder = " → ".join(tv.agents) if tv.agents else "-"
        row = f"{indent}{_ICON.get(tv.status, '·')} {tv.task_id:<14} [{tv.status}] via {ladder}"
        if tv.reviews:
            row += f"  review:{','.join(tv.reviews)}"
        if tv.note:
            row += f"  ({tv.note})"
        lines.append(row)

    placed = set()
    for pid, pv in model.projects.items():
        lines.append(f"  {_ICON.get(pv.status, '·')} project {pid} [{pv.status}]  {pv.goal}")
        for tid in model.task_order:
            tv = model.tasks[tid]
            if tv.project_id == pid:
                render_task(tv, "      ")
                placed.add(tid)
    loose = [model.tasks[t] for t in model.task_order if t not in placed]
    if loose:
        lines.append("  (unassigned tasks)")
        for tv in loose:
            render_task(tv, "      ")

    if model.pi_actions:
        lines.append("  PI actions:")
        for task_id, decision in model.pi_actions:
            lines.append(f"      {task_id} → {decision}")
    u = model.usage
    lines.append(f"  cost: {u['turns']} turns, {u['input_tokens']} in / {u['output_tokens']} out tokens")
    return "\n".join(lines)


class LiveDashboard:
    """A sink that folds events into a model and (optionally) redraws each time."""

    def __init__(self, *, live: bool = False) -> None:
        self.model = LabModel()
        self.live = live

    def __call__(self, ev: Event) -> None:
        self.model.apply(ev)
        if self.live:
            os.system("clear" if os.name == "posix" else "cls")
            print(render(self.model))

    def as_sink(self) -> Sink:
        return self


def build_model(events: list[Event]) -> LabModel:
    """Reconstruct the tree from a recorded/replayed event log — the same fold a
    UI applies live, used for after-the-fact rendering (e.g. a JSONL log)."""
    model = LabModel()
    for ev in events:
        model.apply(ev)
    return model


class TUIOverseer:
    """Steering: show the live tree, then prompt on each PI escalation."""

    def __init__(self, dashboard: LiveDashboard) -> None:
        self.dashboard = dashboard

    def on_escalation(self, supervision) -> PIDecision:
        print("\n" + render(self.dashboard.model))
        esc = supervision.escalation
        note = esc.diagnosis.notes if esc else ""
        ans = input(f"\n[PI] {supervision.task_id} stuck ({note}). [a]ccept/[r]etry/a[b]andon? ")
        return {"r": PIDecision.RETRY, "b": PIDecision.ABANDON}.get(ans.strip().lower(), PIDecision.ACCEPT)

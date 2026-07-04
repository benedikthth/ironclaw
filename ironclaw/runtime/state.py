"""Durable task state: what a suspended PhD writes so it can be resumed later.

The provider and tools are intentionally *not* persisted — they are rebuilt on
resume. Only the irreducible state travels: the task, the running message log,
the iteration count, and which job the PhD is blocked on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ..contracts import Task, _encode


@dataclass
class PersistedTask:
    task: Task
    system: str
    messages: list[dict[str, Any]]
    iterations: int
    workspace: str
    max_iterations: int
    skills_root: str | None = None
    # {"job_id": ..., "call_id": ...} — the awaited job's id and the tool_use id
    # whose result must be filled in on resume.
    awaiting: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def save_state(path: str, state: PersistedTask) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "task": _encode(state.task),
        "system": state.system,
        "messages": state.messages,
        "iterations": state.iterations,
        "workspace": state.workspace,
        "max_iterations": state.max_iterations,
        "skills_root": state.skills_root,
        "awaiting": state.awaiting,
        "extra": state.extra,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_state(path: str) -> PersistedTask:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return PersistedTask(
        task=Task.from_dict(d["task"]),
        system=d["system"],
        messages=d["messages"],
        iterations=d["iterations"],
        workspace=d["workspace"],
        max_iterations=d["max_iterations"],
        skills_root=d.get("skills_root"),
        awaiting=d.get("awaiting"),
        extra=d.get("extra", {}),
    )

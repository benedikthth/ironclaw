"""Core data contracts that flow between the org layers.

A task is authored by a Senior Researcher and handed *down* to a PhD. The PhD's
obligation is defined entirely by ``acceptance`` — the concrete, checkable
definition of done. This is the contract that survives the strict PI->Senior->PhD
layering intact: the acceptance criteria are carried verbatim, so a PhD (running
on a cheap model) is graded against a fixed target rather than a re-interpretation
of the PI's intent.

Results flow *up* as a ``TaskResult`` referencing durable ``Artifact`` files —
never raw transcripts.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CheckKind(str, Enum):
    """How a single acceptance criterion is verified."""

    FILE_EXISTS = "file_exists"  # spec = workspace-relative path
    COMMAND = "command"  # spec = shell command; pass iff exit code 0


@dataclass
class AcceptanceCriterion:
    """One checkable condition. The conjunction of all criteria == "done"."""

    description: str
    kind: CheckKind
    spec: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AcceptanceCriterion":
        return AcceptanceCriterion(
            description=d["description"],
            kind=CheckKind(d["kind"]),
            spec=d["spec"],
        )


@dataclass
class Artifact:
    """A durable result: a report, code, or queryable data file in the workspace."""

    path: str  # workspace-relative
    kind: str  # "report" | "code" | "data" | ...
    description: str = ""


@dataclass
class Task:
    """A unit of work owned by exactly one PhD."""

    goal: str
    acceptance: list[AcceptanceCriterion]
    inputs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    project_id: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Task":
        return Task(
            goal=d["goal"],
            acceptance=[AcceptanceCriterion.from_dict(c) for c in d.get("acceptance", [])],
            inputs=d.get("inputs", {}),
            id=d.get("id", f"task_{uuid.uuid4().hex[:8]}"),
            project_id=d.get("project_id"),
        )


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"  # blocked on a durable background job (slice 2)
    PASSED = "passed"
    FAILED = "failed"  # budget/iterations exhausted or verification failed
    ESCALATED = "escalated"  # handed back up to the Senior


@dataclass
class CheckResult:
    criterion: AcceptanceCriterion
    passed: bool
    detail: str = ""


@dataclass
class TaskResult:
    """What flows upward when a PhD finishes (or gives up)."""

    task_id: str
    status: TaskStatus
    artifacts: list[Artifact] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    summary: str = ""
    iterations: int = 0
    # Optional control-model FailureKind value (a plain string to keep contracts
    # free of a control.py import). Set by a reviewer/gate to tell a supervisor
    # *why* this failed — e.g. "postdoc_rejected" vs the default budget inference.
    failure_kind: str | None = None

    def to_json(self) -> str:
        return json.dumps(_encode(self), indent=2, sort_keys=True)


def _encode(obj: Any) -> Any:
    """Dataclass/enum-aware encoder for stable serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _encode(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


def load_task(path: str) -> Task:
    with open(path, "r", encoding="utf-8") as fh:
        return Task.from_dict(json.load(fh))


def workspace_path(workspace: str, rel: str) -> str:
    """Resolve a workspace-relative path, refusing escapes outside the workspace."""
    root = os.path.realpath(workspace)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"path {rel!r} escapes workspace")
    return full

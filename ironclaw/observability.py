"""Observability: a structured event stream the whole institute emits to.

Every layer (PhD loop, Postdoc, Senior, PI) emits typed ``Event``s to an ambient
``Recorder``. This is the single substrate that all three planned UIs (web, TUI,
Telegram) consume, that cost/usage tracking rides on, and that the human overseer
reads to decide when to step in.

The recorder is *ambient* via a context var, so deep code — even a tool — can
``active_recorder().emit(...)`` without every function signature growing a
parameter. An orchestration entry point wraps its run in ``using(recorder)``;
everything underneath emits into it. With no recorder set, emits hit a cheap
no-op sink.
"""

from __future__ import annotations

import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class Event:
    kind: str  # "task.start", "tool.call", "usage", "disposition", "lab.result", ...
    seq: int = 0
    ts: float = 0.0
    role: str | None = None
    lab: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


Sink = Callable[[Event], None]


class Recorder:
    """Collects events, fans them out to sinks. ``store=False`` for the no-op."""

    def __init__(self, sinks: list[Sink] | None = None, *, store: bool = True) -> None:
        self.sinks = sinks or []
        self.store = store
        self.events: list[Event] = []
        self._seq = 0

    def emit(self, kind: str, **fields: Any) -> Event:
        self._seq += 1
        data = fields.pop("data", None) or {}
        ev = Event(kind=kind, seq=self._seq, ts=time.time(), data=data, **fields)
        if self.store:
            self.events.append(ev)
        for sink in self.sinks:
            sink(ev)
        return ev


NULL_RECORDER = Recorder(store=False)

_current: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
    "ironclaw_recorder", default=None
)


def active_recorder() -> Recorder:
    return _current.get() or NULL_RECORDER


@contextmanager
def using(recorder: Recorder | None) -> Iterator[Recorder]:
    """Make ``recorder`` the ambient recorder for the duration. None -> no-op."""
    rec = recorder or NULL_RECORDER
    token = _current.set(rec)
    try:
        yield rec
    finally:
        _current.reset(token)


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #


def jsonl_sink(path: str) -> Sink:
    """Append each event as one JSON line — the durable audit log / UI feed."""

    def sink(ev: Event) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(ev), sort_keys=True) + "\n")

    return sink


def console_sink(prefix: str = "") -> Sink:
    """Human-readable live progress."""

    def sink(ev: Event) -> None:
        loc = ev.task_id or ev.project_id or ev.lab or "-"
        extra = " ".join(f"{k}={v}" for k, v in ev.data.items())
        line = f"{prefix}[{ev.seq:>3}] {ev.kind:<16} {loc:<14}"
        if ev.message:
            line += f" {ev.message}"
        if extra:
            line += f"  ({extra})"
        print(line)

    return sink


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #


def total_usage(events: list[Event]) -> dict[str, int]:
    """Aggregate token spend across all ``usage`` events — the cost view."""
    totals = {"input_tokens": 0, "output_tokens": 0, "turns": 0}
    for ev in events:
        if ev.kind == "usage":
            totals["input_tokens"] += int(ev.data.get("input_tokens", 0))
            totals["output_tokens"] += int(ev.data.get("output_tokens", 0))
            totals["turns"] += 1
    return totals


def usage_by_agent(events: list[Event]) -> dict[str, dict[str, int]]:
    """Token spend grouped by agent id — where the cost actually went."""
    out: dict[str, dict[str, int]] = {}
    for ev in events:
        if ev.kind == "usage":
            agent = ev.agent_id or ev.data.get("model", "unknown")
            bucket = out.setdefault(agent, {"input_tokens": 0, "output_tokens": 0, "turns": 0})
            bucket["input_tokens"] += int(ev.data.get("input_tokens", 0))
            bucket["output_tokens"] += int(ev.data.get("output_tokens", 0))
            bucket["turns"] += 1
    return out

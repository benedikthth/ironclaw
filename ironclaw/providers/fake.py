"""Deterministic provider for tests and demos.

Drives a scripted sequence of ``AssistantTurn``s so the entire PhD harness can be
exercised end-to-end without a network, API key, or spent token. This is also how
we regression-test harness reliability independent of any model's whims.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import AssistantTurn, LLMProvider, ToolSpec

# A step is either a fixed AssistantTurn or a callable that inspects the running
# message list and returns one (so a script can react to earlier tool results).
Step = AssistantTurn | Callable[[list[dict[str, Any]]], AssistantTurn]


class FakeProvider(LLMProvider):
    def __init__(self, script: list[Step], name: str = "fake") -> None:
        self.name = name
        self._script = list(script)

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        # Stateless indexing: the next step is chosen by how many assistant turns
        # already exist in the log. This mirrors a real (stateless) provider and
        # makes suspend/resume work — a rebuilt FakeProvider picks up exactly
        # where the persisted message log left off.
        idx = sum(1 for m in messages if m.get("role") == "assistant")
        if idx >= len(self._script):
            # Exhausting the script means the harness looped more than expected;
            # surface it loudly rather than hanging.
            raise AssertionError("FakeProvider script exhausted")
        step = self._script[idx]
        return step(messages) if callable(step) else step

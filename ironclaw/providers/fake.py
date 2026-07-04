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
        self._i = 0

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        if self._i >= len(self._script):
            # Exhausting the script means the harness looped more than expected;
            # surface it loudly rather than hanging.
            raise AssertionError("FakeProvider script exhausted")
        step = self._script[self._i]
        self._i += 1
        return step(messages) if callable(step) else step

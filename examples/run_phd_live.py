"""Run the PhD slice against a real Claude model — no scripted provider.

This is the thesis test: a *cheap* model (Haiku by default) driving a real PhD,
with the harness (skills, tools, forced verification) carrying reliability. The
model must decide on its own to consult the skill, write the transform, and
submit — then the harness grades it against the acceptance contract.

    ANTHROPIC_API_KEY=... python examples/run_phd_live.py [model]

Model defaults to claude-haiku-4-5. Pass e.g. claude-opus-4-8 to compare tiers.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw import demo  # noqa: E402
from ironclaw.agents.phd import run_phd  # noqa: E402
from ironclaw.observability import Recorder, console_sink, total_usage, using  # noqa: E402
from ironclaw.providers.anthropic import AnthropicProvider  # noqa: E402


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5"
    provider = AnthropicProvider(model=model)
    skills_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
    recorder = Recorder(sinks=[console_sink()])
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "sales.csv"), "w", encoding="utf-8") as fh:
            fh.write(demo.SALES_CSV)
        # `using` makes the recorder ambient, so the PhD loop streams its events.
        with using(recorder):
            outcome = run_phd(
                task=demo.TASK, provider=provider, workspace=tmp, skills_root=skills_root
            )
        print(f"\n=== {model} ===")
        print(outcome.result.to_json())
        u = total_usage(recorder.events)
        print(
            f"\ncost: {u['turns']} turns, {u['input_tokens']} in / {u['output_tokens']} out tokens"
        )


if __name__ == "__main__":
    main()

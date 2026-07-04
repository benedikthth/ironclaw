"""End-to-end LIVE lab: a sentence -> real institute -> rendered tree.

Wires the real LLM seams together against actual Claude models:
  PI (llm_decomposer)  ->  Senior (relaunch ladder + llm_task_author)
      ->  Postdoc (llm_reviewer, blocking gate)  ->  PhD (real, on the assigned model)
Everything streams through the TUI fold, and real token cost is reported.

    ANTHROPIC_API_KEY=... python examples/run_lab_live.py "<problem statement>"
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.pi import run_lab  # noqa: E402
from ironclaw.agents.postdoc import reviewed_phd_runner, llm_reviewer  # noqa: E402
from ironclaw.authoring import author_problem, llm_decomposer, llm_task_author  # noqa: E402
from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.observability import Recorder, console_sink, total_usage, using  # noqa: E402
from ironclaw.providers.registry import ProviderConfig, default_registry  # noqa: E402
from ironclaw.tui import LiveDashboard, render  # noqa: E402


def main() -> None:
    statement = sys.argv[1] if len(sys.argv) > 1 else (
        "Compute basic statistics for the integers 1..100 and write them to a report."
    )
    key = os.environ.get("ANTHROPIC_API_KEY")
    skills_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")

    # Per-role model whitelist: cheap PhDs, a stronger Postdoc/Senior/PI.
    catalog = AgentCatalog(
        [
            AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1),
            AgentSpec("moderate", "anthropic", "claude-sonnet-5", 2),
        ],
        {Role.PHD: ["cheapo", "moderate"]},
    )

    # One registry wires every role; swap any AgentSpec.provider / .build(...) call
    # below to "openai" / "openrouter" / "local" to run that role elsewhere.
    registry = default_registry()
    if key:
        registry.register(ProviderConfig("anthropic", "anthropic", api_key=key))

    # Reviewer / authors are now provider-agnostic — built from the registry.
    reviewer = llm_reviewer(registry.build("anthropic", "claude-sonnet-5"))
    author = llm_task_author(registry.build("anthropic", "claude-opus-4-8"))
    decomposer = llm_decomposer(registry.build("anthropic", "claude-opus-4-8"))
    # Postdoc-gated real-PhD runner: each PhD run is reviewed before it counts.
    runner = reviewed_phd_runner(reviewer, skills_root=skills_root, registry=registry, max_rounds=1)

    dash = LiveDashboard()
    rec = Recorder(sinks=[dash.as_sink(), console_sink()])

    with tempfile.TemporaryDirectory() as tmp, using(rec):
        print(f"=== decomposing: {statement}\n")
        problem = author_problem(statement, decomposer)
        run_lab(
            problem=problem, catalog=catalog, runner=runner, workspace=tmp,
            phd_agent_id="cheapo", max_relaunches=1, base_budget=12,
            author=author, recorder=rec,
        )

    print("\n=== final tree ===")
    print(render(dash.model))
    u = total_usage(rec.events)
    print(f"\ntotal cost: {u['turns']} model turns, {u['input_tokens']} in / {u['output_tokens']} out tokens")


if __name__ == "__main__":
    main()

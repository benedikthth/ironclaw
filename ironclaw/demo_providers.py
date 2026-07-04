"""Mixed-provider catalog: any models, any number of providers.

Shows a single PhD whitelist spanning four providers, and the registry building
the right client for each — deterministically, no keys or network (clients are
constructed lazily, so this just proves the routing).

    python -m ironclaw.demo_providers
"""

from __future__ import annotations

from .control import AgentCatalog, AgentSpec, Role
from .providers.registry import default_registry


def main() -> None:
    # A PhD pool mixing four providers; the Senior picks among them by cost_tier,
    # relaunching up the ladder on failure. Postdoc/Senior can use their own mix.
    catalog = AgentCatalog(
        [
            AgentSpec("local-llama", "local", "llama3.1:8b", cost_tier=1),
            AgentSpec("gpt-mini", "openai", "gpt-4o-mini", cost_tier=1),
            AgentSpec("haiku", "anthropic", "claude-haiku-4-5", cost_tier=2),
            AgentSpec("or-70b", "openrouter", "meta-llama/llama-3.1-70b-instruct", cost_tier=2),
            AgentSpec("opus", "anthropic", "claude-opus-4-8", cost_tier=3),
        ],
        {Role.PHD: ["local-llama", "gpt-mini", "haiku", "or-70b", "opus"]},
    )

    registry = default_registry()
    print("PhD ladder (weakest → strongest), each on its own provider:\n")
    for spec in catalog.allowed(Role.PHD):
        provider = registry.build(spec.provider, spec.model)
        endpoint = provider.base_url or "(provider default)"
        print(f"  tier {spec.cost_tier}  {spec.id:<12} {spec.provider:<10} {spec.model:<34} -> {endpoint}")

    print(f"\nregistered providers: {registry.names()}")
    print("keys resolve from env: ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY;")
    print("local defaults to Ollama (LOCAL_AI_BASE_URL to point at vLLM/LM Studio).")


if __name__ == "__main__":
    main()

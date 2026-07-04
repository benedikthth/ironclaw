"""Provider registry — mix any models across any number of providers.

An ``AgentSpec`` says *which provider* and *which model* an agent runs on
(control.AgentSpec.provider / .model). The registry maps a provider name to a
concrete client config and builds the right adapter on demand. So a role's
whitelist can freely mix, e.g. a cheap local Llama, GPT-4o-mini via OpenAI, and
Haiku via Anthropic — the Senior picks among them, the registry wires each up.

Two wire protocols cover everything: ``anthropic`` and ``openai`` (the latter
also serves OpenRouter and any self-hosted OpenAI-compatible endpoint via
``base_url``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .anthropic import AnthropicProvider
from .base import LLMProvider
from .openai import OpenAICompatibleProvider


@dataclass
class ProviderConfig:
    name: str  # how AgentSpec.provider refers to it, e.g. "openrouter"
    kind: str  # wire protocol: "anthropic" | "openai"
    base_url: str | None = None
    api_key: str | None = None  # explicit key (wins over env)
    api_key_env: str | None = None  # else read the key from this env var
    default_headers: dict | None = None

    def resolve_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class ProviderRegistry:
    def __init__(self, configs: list[ProviderConfig] | None = None) -> None:
        self._by_name: dict[str, ProviderConfig] = {c.name: c for c in (configs or [])}

    def register(self, config: ProviderConfig) -> "ProviderRegistry":
        self._by_name[config.name] = config
        return self

    def names(self) -> list[str]:
        return list(self._by_name)

    def build(self, provider_name: str, model: str, *, max_tokens: int = 4096) -> LLMProvider:
        cfg = self._by_name.get(provider_name)
        if cfg is None:
            raise KeyError(f"unknown provider {provider_name!r}; registered: {self.names()}")
        key = cfg.resolve_key()
        if cfg.kind == "anthropic":
            return AnthropicProvider(model, api_key=key, base_url=cfg.base_url, max_tokens=max_tokens)
        if cfg.kind == "openai":
            return OpenAICompatibleProvider(
                model, base_url=cfg.base_url, api_key=key, max_tokens=max_tokens,
                default_headers=cfg.default_headers, name=cfg.name,
            )
        raise ValueError(f"unknown provider kind {cfg.kind!r} for {provider_name!r}")


def structured_provider(provider=None, *, model: str = "claude-opus-4-8", api_key: str | None = None):
    """Resolve a provider for a one-shot structured call (reviewer/authoring).
    Pass a built provider to run on any endpoint; omit it for the Anthropic
    convenience path (back-compat with the ``model``/``api_key`` signature)."""
    if provider is not None:
        return provider
    return AnthropicProvider(model, api_key=api_key)


def default_registry() -> ProviderRegistry:
    """Sensible presets for the four common providers, keyed off standard env
    vars. ``local`` defaults to Ollama's OpenAI-compatible endpoint; override
    ``LOCAL_AI_BASE_URL`` for vLLM/LM Studio/llama.cpp."""
    return ProviderRegistry(
        [
            ProviderConfig("anthropic", "anthropic", api_key_env="ANTHROPIC_API_KEY"),
            ProviderConfig("openai", "openai", api_key_env="OPENAI_API_KEY"),
            ProviderConfig(
                "openrouter", "openai",
                base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
            ),
            ProviderConfig(
                "local", "openai",
                base_url=os.environ.get("LOCAL_AI_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("LOCAL_AI_API_KEY", "ollama"),
            ),
        ]
    )

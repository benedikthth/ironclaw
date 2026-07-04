"""Tests for multi-provider support: OpenAI translation + registry routing."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.control import AgentCatalog, AgentSpec, Role  # noqa: E402
from ironclaw.providers.anthropic import AnthropicProvider  # noqa: E402
from ironclaw.providers.base import (  # noqa: E402
    AssistantTurn,
    ToolCall,
    ToolSpec,
    assistant_message,
    tool_result,
    user_text,
)
from ironclaw.providers.openai import OpenAICompatibleProvider  # noqa: E402
from ironclaw.providers.registry import ProviderRegistry, default_registry  # noqa: E402


class _FakeOpenAI:
    """Records the request payload and returns a scripted response."""

    def __init__(self, response):
        self.response = response
        self.captured = None
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.captured = kw
                return outer.response

        self.chat = SimpleNamespace(completions=_Completions())


def _tool_response():
    tc = SimpleNamespace(
        id="c9", function=SimpleNamespace(name="python_exec", arguments='{"code": "print(1)"}')
    )
    msg = SimpleNamespace(content="running it", tool_calls=[tc])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


class TestOpenAITranslation(unittest.TestCase):
    def test_neutral_conversation_maps_to_chat_completions(self):
        fake = _FakeOpenAI(_tool_response())
        provider = OpenAICompatibleProvider("gpt-4o-mini", client=fake, name="openai")
        messages = [
            user_text("do the task"),
            assistant_message(AssistantTurn(text="ok", tool_calls=[ToolCall("call1", "write_file", {"path": "x"})])),
            tool_result("call1", "wrote 1 byte"),
        ]
        tools = [ToolSpec("write_file", "write a file", {"type": "object", "properties": {}})]

        turn = provider.complete("SYSTEM", messages, tools)

        sent = fake.captured["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "tool"])
        self.assertEqual(sent[0]["content"], "SYSTEM")
        # the neutral tool_use becomes an OpenAI tool_calls entry ...
        self.assertEqual(sent[2]["tool_calls"][0]["function"]["name"], "write_file")
        # ... and the neutral tool_result becomes a standalone role:tool message.
        self.assertEqual(sent[3], {"role": "tool", "tool_call_id": "call1", "content": "wrote 1 byte"})
        self.assertEqual(fake.captured["tools"][0]["function"]["name"], "write_file")

        # response parsing: tool call + usage
        self.assertEqual(turn.text, "running it")
        self.assertEqual(turn.tool_calls[0].name, "python_exec")
        self.assertEqual(turn.tool_calls[0].arguments, {"code": "print(1)"})
        self.assertEqual(turn.usage["input_tokens"], 100)
        self.assertEqual(turn.usage["output_tokens"], 20)

    def test_no_tools_response_parses(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )
        provider = OpenAICompatibleProvider("m", client=_FakeOpenAI(resp))
        turn = provider.complete("", [user_text("hi")], [])
        self.assertEqual(turn.text, "hi")
        self.assertEqual(turn.tool_calls, [])


class TestRegistryRouting(unittest.TestCase):
    def test_default_registry_routes_each_provider(self):
        reg = default_registry()
        a = reg.build("anthropic", "claude-haiku-4-5")
        self.assertIsInstance(a, AnthropicProvider)
        self.assertEqual(a.model, "claude-haiku-4-5")

        o = reg.build("openai", "gpt-4o-mini")
        self.assertIsInstance(o, OpenAICompatibleProvider)
        self.assertIsNone(o.base_url)

        r = reg.build("openrouter", "meta-llama/llama-3.1-70b")
        self.assertEqual(r.base_url, "https://openrouter.ai/api/v1")

        local = reg.build("local", "llama3.1:8b")
        self.assertIn("localhost", local.base_url)

    def test_unknown_provider_raises(self):
        with self.assertRaises(KeyError):
            default_registry().build("nope", "m")

    def test_explicit_key_beats_env(self):
        from ironclaw.providers.registry import ProviderConfig

        reg = ProviderRegistry([ProviderConfig("x", "openai", api_key="explicit")])
        self.assertEqual(reg._by_name["x"].resolve_key(), "explicit")

    def test_mixed_provider_catalog_builds_each_agent(self):
        # "any combination of models for any number of providers"
        catalog = AgentCatalog(
            [
                AgentSpec("anth", "anthropic", "claude-haiku-4-5", 1),
                AgentSpec("oai", "openai", "gpt-4o-mini", 1),
                AgentSpec("orouter", "openrouter", "x/y-70b", 2),
                AgentSpec("llama", "local", "llama3.1:8b", 1),
            ],
            {Role.PHD: ["anth", "oai", "orouter", "llama"]},
        )
        reg = default_registry()
        built = {s.id: reg.build(s.provider, s.model) for s in catalog.allowed(Role.PHD)}
        self.assertIsInstance(built["anth"], AnthropicProvider)
        self.assertIsInstance(built["oai"], OpenAICompatibleProvider)
        self.assertEqual(built["orouter"].base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(built["llama"].model, "llama3.1:8b")


class _FakeStructured:
    """A minimal provider exposing only .structured — proves the reviewer/authors
    are provider-agnostic (any provider with structured output works)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def structured(self, system, prompt, schema, *, max_tokens=2048):
        self.calls.append({"system": system, "schema": schema})
        return self.responses.pop(0)


class TestProviderAgnosticSeams(unittest.TestCase):
    def test_reviewer_runs_on_any_provider(self):
        from ironclaw.agents.postdoc import llm_reviewer
        from ironclaw.contracts import TaskResult, TaskStatus
        from ironclaw.control import FlagKind, ReviewVerdict

        fake = _FakeStructured([{"verdict": "reject", "feedback": "needs work", "flag": "misspecified"}])
        review = llm_reviewer(fake)
        from ironclaw.contracts import Task

        out = review(Task(goal="g", acceptance=[]), TaskResult("t", TaskStatus.PASSED), "/tmp")
        self.assertIs(out.verdict, ReviewVerdict.REJECT)
        self.assertIs(out.flag, FlagKind.MISSPECIFIED)
        self.assertTrue(fake.calls)  # went through provider.structured

    def test_decomposer_runs_on_any_provider(self):
        from ironclaw.authoring import llm_decomposer

        payload = {"projects": [{"id": "p1", "goal": "g", "tasks": [
            {"id": "t1", "goal": "do", "acceptance": [
                {"description": "exists", "kind": "file_exists", "spec": "out.txt"}], "depends_on": []}]}]}
        decompose = llm_decomposer(_FakeStructured([payload]))
        projects = decompose("some problem")
        self.assertEqual(projects[0].id, "p1")
        self.assertEqual(projects[0].tasks[0].acceptance[0].spec, "out.txt")

    def test_task_author_runs_on_any_provider(self):
        from ironclaw.authoring import llm_task_author
        from ironclaw.contracts import Task

        reformulated = {"goal": "clearer goal", "acceptance": []}
        split = {"subtasks": [
            {"id": "s1", "goal": "a", "acceptance": []},
            {"id": "s2", "goal": "b", "acceptance": []},
        ]}
        author = llm_task_author(_FakeStructured([reformulated, split]))
        t = Task(goal="vague", acceptance=[], id="x")
        self.assertEqual(author.reformulate(t, None).goal, "clearer goal")
        self.assertEqual([s.id for s in author.split(t, None)], ["s1", "s2"])


if __name__ == "__main__":
    unittest.main()

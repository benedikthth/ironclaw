"""LLM-authoring seams: turn plain language into the institute's structured work.

Two authoring roles, both pluggable (LLM-backed in production, deterministic
stand-ins in tests):

- ``Decomposer`` — the PI turning a raw problem *statement* into projects and
  tasks (each with checkable acceptance criteria). This is what lets a lab start
  from a sentence instead of a hand-built ``Problem``.
- ``TaskAuthor`` — the Senior rewriting a failed task (REFORMULATE) or splitting
  it into subtasks (SPLIT). The deterministic supervision loop already routes to
  these dispositions; the author supplies the new task *content*.

The LLM implementations use structured outputs so the model returns validated
JSON, not prose to parse.
"""

from __future__ import annotations

from typing import Protocol

from .agents.pi import Problem, ProjectSpec
from .contracts import AcceptanceCriterion, CheckKind, Task
from .observability import active_recorder


class Decomposer(Protocol):
    def __call__(self, statement: str) -> list[ProjectSpec]: ...


class TaskAuthor(Protocol):
    def reformulate(self, task: Task, diagnosis) -> Task: ...
    def split(self, task: Task, diagnosis) -> list[Task]: ...


def author_problem(statement: str, decomposer: Decomposer) -> Problem:
    """PI entry point: a sentence in, a structured Problem out."""
    projects = decomposer(statement)
    active_recorder().emit(
        "pi.decompose", role="pi", lab=statement,
        data={"projects": [p.id for p in projects],
              "tasks": sum(len(p.tasks) for p in projects)},
    )
    return Problem(statement=statement, projects=projects)


# --------------------------------------------------------------------------- #
# Structured-output schemas
# --------------------------------------------------------------------------- #

_ACCEPTANCE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "kind": {"type": "string", "enum": ["file_exists", "command"]},
            "spec": {"type": "string"},
        },
        "required": ["description", "kind", "spec"],
        "additionalProperties": False,
    },
}

_DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "goal": {"type": "string"},
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "goal": {"type": "string"},
                                "acceptance": _ACCEPTANCE,
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["id", "goal", "acceptance", "depends_on"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "goal", "tasks"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["projects"],
    "additionalProperties": False,
}


def _to_acceptance(items: list[dict]) -> list[AcceptanceCriterion]:
    return [AcceptanceCriterion(i["description"], CheckKind(i["kind"]), i["spec"]) for i in items]


def llm_decomposer(provider=None, *, memory=None, model: str = "claude-opus-4-8", api_key: str | None = None) -> Decomposer:
    """PI decomposition, provider-agnostic: pass any provider (built from the
    registry) or omit it for the Anthropic convenience path. A strong model is
    recommended — the plan sets up everything below it. Pass a ``MemoryStore`` to
    inform the plan with prior related labs/findings (cross-lab sharing)."""
    from .providers.registry import structured_provider

    prov = structured_provider(provider, model=model, api_key=api_key)

    def decompose(statement: str) -> list[ProjectSpec]:
        prior = memory.context_for(statement) if memory is not None else ""
        prior_block = (prior + "\n\n") if prior else ""
        prompt = (
            f"You are the PI of a research lab. Decompose this problem into "
            f"projects, each with concrete tasks. Every task must have "
            f"mechanically-checkable acceptance criteria (a file that must exist, "
            f"or a shell command that must exit 0). Tasks in a project share one "
            f"workspace directory. If a task needs a sibling task's output, list "
            f"that sibling's id in depends_on (else use an empty list); the "
            f"dependency's files will already be present when the task runs.\n\n"
            f"{prior_block}Problem:\n\n{statement}"
        )
        data = prov.structured(
            system="You are the PI of a research lab.", prompt=prompt, schema=_DECOMPOSE_SCHEMA, max_tokens=4096
        )
        return [
            ProjectSpec(
                id=p["id"],
                goal=p["goal"],
                tasks=[
                    Task(
                        goal=t["goal"],
                        acceptance=_to_acceptance(t["acceptance"]),
                        id=t["id"],
                        project_id=p["id"],
                        depends_on=list(t.get("depends_on", [])),
                    )
                    for t in p["tasks"]
                ],
            )
            for p in data["projects"]
        ]

    return decompose


def llm_task_author(provider=None, *, model: str = "claude-opus-4-8", api_key: str | None = None) -> TaskAuthor:
    """Senior authoring, provider-agnostic: rewrite a stuck task, or split it into
    subtasks. Pass any provider (built from the registry) or omit for Anthropic."""
    from .providers.registry import structured_provider

    prov = structured_provider(provider, model=model, api_key=api_key)

    class _Author:
        def reformulate(self, task: Task, diagnosis) -> Task:
            schema = {
                "type": "object",
                "properties": {"goal": {"type": "string"}, "acceptance": _ACCEPTANCE},
                "required": ["goal", "acceptance"],
                "additionalProperties": False,
            }
            prompt = (
                f"A PhD could not complete this task:\n\nGoal: {task.goal}\n"
                f"Why it failed: {getattr(diagnosis, 'notes', '')}\n\n"
                "Rewrite it to be clearer and more achievable, keeping the same "
                "intent. Return a new goal and mechanically-checkable acceptance criteria."
            )
            data = prov.structured(system="You are a senior researcher.", prompt=prompt, schema=schema, max_tokens=2048)
            return Task(goal=data["goal"], acceptance=_to_acceptance(data["acceptance"]),
                        inputs=task.inputs, project_id=task.project_id)

        def split(self, task: Task, diagnosis) -> list[Task]:
            schema = {
                "type": "object",
                "properties": {
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}, "goal": {"type": "string"}, "acceptance": _ACCEPTANCE},
                            "required": ["id", "goal", "acceptance"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["subtasks"],
                "additionalProperties": False,
            }
            prompt = (
                f"This task is too big for one PhD:\n\nGoal: {task.goal}\n"
                f"Trouble: {getattr(diagnosis, 'notes', '')}\n\n"
                "Split it into 2-4 smaller, independently-checkable subtasks."
            )
            data = prov.structured(system="You are a senior researcher.", prompt=prompt, schema=schema, max_tokens=2048)
            return [
                Task(goal=s["goal"], acceptance=_to_acceptance(s["acceptance"]), id=s["id"], project_id=task.project_id)
                for s in data["subtasks"]
            ]

    return _Author()

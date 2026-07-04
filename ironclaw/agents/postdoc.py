"""The Postdoc: the submission-quality gate (the control model's first gate).

The Postdoc reviews a PhD's *mechanically-passing* submission for the quality the
acceptance criteria failed to encode. It is blocking but iteration-boxed:

- APPROVE -> the submission flows up unchanged (PASSED).
- REJECT + feedback -> the PhD re-runs with the feedback injected; bounded by
  ``max_rounds``.
- A narrow typed FLAG (misspecified / suspected-intractable) -> straight up to
  the Senior, no more PhD rounds.
- ``max_rounds`` rejections without approval -> up to the Senior as a
  POSTDOC_REJECTED failure (the Senior's policy then suggests SPLIT).

Composition: ``reviewed_phd_runner`` produces a Senior-compatible runner, so the
full stack is PI -> Senior -> **Postdoc** -> PhD, each layer unaware of the ones
below it. A PhD result that doesn't even pass mechanical verification is not a
quality question — it flows straight up to the Senior untouched.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Protocol

from ..contracts import Task, TaskResult, TaskStatus
from ..control import FailureKind, FlagKind, PostdocReview, ReviewVerdict
from ..observability import active_recorder


class Reviewer(Protocol):
    """Judges a PhD submission. The default is Claude-backed; tests inject a
    deterministic stand-in."""

    def __call__(self, task: Task, result: TaskResult, workspace: str) -> PostdocReview: ...


class PhDAttempt(Protocol):
    def __call__(self, task: Task, workspace: str) -> TaskResult: ...


def reviewed_run(
    *,
    task: Task,
    phd_run: PhDAttempt,
    reviewer: Reviewer,
    workspace: str,
    max_rounds: int = 2,
) -> TaskResult:
    feedback: str | None = None
    result: TaskResult | None = None

    for round_i in range(max_rounds):
        attempt_task = _with_feedback(task, feedback) if feedback else task
        ws = os.path.join(workspace, f"round_{round_i}")
        result = phd_run(attempt_task, ws)

        if result.status is not TaskStatus.PASSED:
            # Didn't clear the objective floor — a Senior matter, not a quality
            # review. Pass it up untouched (budget exhaustion etc.).
            return result

        review = reviewer(task, result, ws)
        active_recorder().emit(
            "review", role="postdoc", task_id=task.id, message=review.verdict.value,
            data={"round": round_i, "flag": review.flag.value if review.flag else None},
        )

        if review.flag is not None:
            result.status = TaskStatus.ESCALATED
            result.failure_kind = (
                FailureKind.SUSPECTED_INTRACTABLE.value
                if review.flag is FlagKind.SUSPECTED_INTRACTABLE
                else FailureKind.MISSPECIFIED.value
            )
            result.summary = f"Postdoc flag [{review.flag.value}]: {review.feedback}"
            return result

        if review.verdict is ReviewVerdict.APPROVE:
            result.summary = (result.summary + " | postdoc: approved").strip(" |")
            return result

        feedback = review.feedback  # REJECT: carry it into the next round

    # Box exhausted: persistent quality rejection -> up to the Senior.
    assert result is not None
    result.status = TaskStatus.ESCALATED
    result.failure_kind = FailureKind.POSTDOC_REJECTED.value
    result.summary = f"Postdoc rejected after {max_rounds} rounds: {feedback}"
    return result


def _with_feedback(task: Task, feedback: str) -> Task:
    inputs = dict(task.inputs)
    inputs["postdoc_feedback"] = feedback
    return replace(task, inputs=inputs)


def reviewed_phd_runner(
    reviewer: Reviewer,
    *,
    skills_root: str | None = None,
    api_key: str | None = None,
    max_rounds: int = 2,
):
    """A Senior-compatible runner (task, agent, budget, workspace) that gates each
    real PhD run through the Postdoc."""

    def run(task: Task, agent, budget_iterations: int, workspace: str) -> TaskResult:
        from ..providers.anthropic import AnthropicProvider
        from ..runtime.loop import LoopConfig
        from .phd import run_phd

        def phd_run(t: Task, ws: str) -> TaskResult:
            provider = AnthropicProvider(model=agent.model, api_key=api_key)
            return run_phd(
                task=t,
                provider=provider,
                workspace=ws,
                skills_root=skills_root,
                config=LoopConfig(max_iterations=budget_iterations),
            ).result

        return reviewed_run(
            task=task, phd_run=phd_run, reviewer=reviewer, workspace=workspace, max_rounds=max_rounds
        )

    return run


def llm_reviewer(model: str = "claude-sonnet-5", *, api_key: str | None = None) -> Reviewer:
    """A Claude-backed Postdoc. Reads the artifacts and returns a structured
    verdict. Postdocs use a stronger model than PhDs by policy — quality review is
    where you don't skimp."""
    import json

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approve", "reject"]},
            "feedback": {"type": "string"},
            "flag": {"type": "string", "enum": ["none", "misspecified", "suspected_intractable"]},
        },
        "required": ["verdict", "feedback", "flag"],
        "additionalProperties": False,
    }

    def review(task: Task, result: TaskResult, workspace: str) -> PostdocReview:
        listing = []
        for art in result.artifacts:
            try:
                with open(os.path.join(workspace, art.path), "r", encoding="utf-8") as fh:
                    listing.append(f"### {art.path} ({art.kind})\n{fh.read()[:4000]}")
            except OSError:
                listing.append(f"### {art.path} (unreadable)")
        prompt = (
            f"Task goal:\n{task.goal}\n\nPhD summary:\n{result.summary}\n\n"
            f"Artifacts:\n" + "\n\n".join(listing) + "\n\n"
            "Review this submission for correctness and quality the acceptance "
            "criteria may not have captured. Approve if it genuinely satisfies the "
            "goal. Reject with specific, actionable feedback otherwise. Use flag="
            "'misspecified' if the task's criteria are themselves wrong, "
            "'suspected_intractable' if the goal appears infeasible, else 'none'."
        )
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        data = json.loads(next(b.text for b in resp.content if b.type == "text"))
        flag = None if data["flag"] == "none" else FlagKind(data["flag"])
        return PostdocReview(
            verdict=ReviewVerdict(data["verdict"]), feedback=data["feedback"], flag=flag
        )

    return review

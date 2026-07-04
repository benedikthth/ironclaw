"""Self-improving skills — learn from success, not just repair on failure.

The Infrastructure Manager already harvests skills a PhD explicitly left behind
and quarantines/repairs broken ones. This adds Hermes's missing half: when a PhD
*succeeds* at something non-trivial, distill the approach into a reusable skill
automatically, so the next PhD doesn't start from zero.

Threshold-triggered like Hermes (only skill-worthy work): the task passed AND it
took real effort (>= ``min_tool_calls`` tool calls). The distillation is
provider-agnostic (``provider.structured``), reading the artifacts the PhD
produced so the skill captures the actual procedure, generalized. Dedup in the IM
keeps the folder clean when similar work recurs.
"""

from __future__ import annotations

import os

from .contracts import TaskStatus
from .infra import SkillCandidate
from .observability import active_recorder

_SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "worth_keeping": {"type": "boolean"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["worth_keeping", "name", "description", "body"],
    "additionalProperties": False,
}


class SkillLearner:
    def __init__(self, im, provider, *, min_tool_calls: int = 5) -> None:
        self.im = im
        self.provider = provider
        self.min_tool_calls = min_tool_calls

    def consider(self, task, supervision, workspace: str, tool_calls: int) -> str | None:
        """If a passed task cleared the effort threshold, distill + register a
        skill. Returns the registered skill name, or None."""
        if supervision.status is not TaskStatus.PASSED:
            return None
        if tool_calls < self.min_tool_calls:
            return None

        artifacts = []
        for art in getattr(supervision.result, "artifacts", []):
            path = os.path.join(workspace, art.path)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    artifacts.append(f"### {art.path}\n{fh.read()[:3000]}")
            except OSError:
                continue

        prompt = (
            f"A PhD solved this task in {tool_calls} steps:\n\nGoal: {task.goal}\n\n"
            f"Artifacts produced:\n" + ("\n\n".join(artifacts) or "(none)") + "\n\n"
            "If the *general approach* is reusable for future similar tasks, distill "
            "it into a skill: a short name, a one-line description, and a Markdown "
            "procedure body. Generalize — strip task-specific data/paths. Set "
            "worth_keeping=false for trivial or one-off work."
        )
        data = self.provider.structured(
            system="You distill successful work into reusable, general skills.",
            prompt=prompt, schema=_SKILL_SCHEMA, max_tokens=1024,
        )
        if not data.get("worth_keeping"):
            return None

        reg = self.im.register_skill(
            SkillCandidate(data["name"], data["description"], data["body"], source=f"learned:{task.id}")
        )
        if reg.get("status") == "registered":
            active_recorder().emit(
                "skill.learned", role="infra", message=data["name"],
                data={"from": task.id, "tool_calls": tool_calls},
            )
            return reg.get("name")
        return None

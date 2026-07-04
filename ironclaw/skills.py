"""Skills: reusable, versioned capability recipes.

A skill is a folder containing ``SKILL.md`` (frontmatter with ``name`` and
``description``, then the procedure) plus optional helper files. This is the
same progressive-disclosure shape most harnesses use: the registry advertises
only name+description cheaply, and the full procedure is pulled on demand.

Skills are the answer to "PhDs should not start from zero when touching
infrastructure." The Infrastructure Manager agent is the custodian of this
folder; PhDs consult the registry rather than reinventing the enqueue/poll/
download/transform wheel every time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: str


def _parse_skill_md(text: str, path: str) -> Skill:
    name = os.path.basename(os.path.dirname(path))
    description = ""
    body = text
    # Minimal frontmatter parse: a leading '---' fenced block of key: value lines.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            for line in front.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip().lower()
                    val = val.strip()
                    if key == "name" and val:
                        name = val
                    elif key == "description":
                        description = val
    return Skill(name=name, description=description, body=body, path=path)


class SkillRegistry:
    """Indexes a skills root directory. Custodian surface for the Infra Manager."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        self._skills.clear()
        if not os.path.isdir(self.root):
            return
        for entry in sorted(os.listdir(self.root)):
            md = os.path.join(self.root, entry, "SKILL.md")
            if os.path.isfile(md):
                with open(md, "r", encoding="utf-8") as fh:
                    skill = _parse_skill_md(fh.read(), md)
                self._skills[skill.name] = skill

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def catalog(self) -> str:
        """Cheap advertisement injected into the PhD's system prompt."""
        if not self._skills:
            return "(no skills registered)"
        return "\n".join(f"- {s.name}: {s.description}" for s in self.list())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

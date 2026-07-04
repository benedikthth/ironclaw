"""The submit_result tool: the PhD's signal that it believes the task is done.

Submitting does not itself declare success — it hands control back to the
harness, which runs verification against the acceptance contract. This keeps the
pass/fail decision out of the model's hands.
"""

from __future__ import annotations

from typing import Any

from ..contracts import Artifact
from .base import Tool, ToolContext


class SubmitResult(Tool):
    name = "submit_result"
    description = (
        "Declare the task complete and hand your artifacts up for verification. "
        "List every artifact you produced. The harness will check them against "
        "the acceptance criteria; if a check fails you will be asked to continue."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "One-paragraph result summary."},
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "kind": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["path", "kind"],
                },
            },
        },
        "required": ["summary", "artifacts"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        artifacts = [
            Artifact(path=a["path"], kind=a["kind"], description=a.get("description", ""))
            for a in args.get("artifacts", [])
        ]
        ctx.scratch["submission"] = {
            "summary": args.get("summary", ""),
            "artifacts": artifacts,
        }
        return "submission received; running verification"

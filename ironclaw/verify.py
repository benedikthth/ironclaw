"""Verify a PhD's output against the task's acceptance criteria.

Verification is deliberately mechanical: the Senior authored concrete,
checkable criteria, and the PhD is graded against them rather than against a
reviewer's mood. This is what makes "done" objective and keeps a cheap model
from talking its way to a false pass.
"""

from __future__ import annotations

import os
import subprocess

from .contracts import (
    AcceptanceCriterion,
    CheckKind,
    CheckResult,
    Task,
    workspace_path,
)


def check_one(criterion: AcceptanceCriterion, workspace: str) -> CheckResult:
    if criterion.kind is CheckKind.FILE_EXISTS:
        try:
            full = workspace_path(workspace, criterion.spec)
        except ValueError as e:
            return CheckResult(criterion, False, str(e))
        ok = os.path.exists(full)
        return CheckResult(criterion, ok, "found" if ok else "missing")

    if criterion.kind is CheckKind.COMMAND:
        try:
            proc = subprocess.run(
                criterion.spec,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(criterion, False, "timeout")
        detail = ((proc.stdout or "") + (proc.stderr or "")).strip()[:500]
        return CheckResult(criterion, proc.returncode == 0, f"[exit {proc.returncode}] {detail}")

    return CheckResult(criterion, False, f"unknown check kind {criterion.kind}")


def verify(task: Task, workspace: str) -> list[CheckResult]:
    return [check_one(c, workspace) for c in task.acceptance]


def all_passed(results: list[CheckResult]) -> bool:
    return bool(results) and all(r.passed for r in results)

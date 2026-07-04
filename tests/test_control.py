"""Tests for the control model: agent whitelists and failure disposition."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.contracts import TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import (  # noqa: E402
    AgentAssignment,
    AgentCatalog,
    AgentSpec,
    Disposition,
    FailureKind,
    Role,
    escalate,
    suggest_disposition,
)


def _catalog() -> AgentCatalog:
    agents = [
        AgentSpec("cheapo", "openrouter", "cheap-model", cost_tier=1),
        AgentSpec("moderate", "anthropic", "mid-model", cost_tier=2),
        AgentSpec("x", "anthropic", "strong-model", cost_tier=3),
    ]
    whitelists = {
        Role.PHD: ["cheapo", "moderate"],
        Role.POSTDOC: ["x"],
        Role.SENIOR: ["moderate", "x"],
    }
    return AgentCatalog(agents, whitelists)


class TestWhitelist(unittest.TestCase):
    def test_allowed_is_ordered_weakest_first(self):
        cat = _catalog()
        self.assertEqual([a.id for a in cat.allowed(Role.PHD)], ["cheapo", "moderate"])

    def test_next_stronger_bumps_up_then_stops(self):
        cat = _catalog()
        self.assertEqual(cat.next_stronger(Role.PHD, "cheapo").id, "moderate")
        self.assertIsNone(cat.next_stronger(Role.PHD, "moderate"))

    def test_assignment_validate_rejects_non_whitelisted(self):
        cat = _catalog()
        AgentAssignment(Role.PHD, "cheapo").validate(cat)  # ok
        with self.assertRaises(ValueError):
            AgentAssignment(Role.PHD, "x").validate(cat)  # x not allowed for PhD


class TestDisposition(unittest.TestCase):
    def test_budget_exhausted_bumps_agent_when_possible(self):
        d = suggest_disposition(
            failure_kind=FailureKind.BUDGET_EXHAUSTED,
            role=Role.PHD,
            agent_id="cheapo",
            catalog=_catalog(),
        )
        self.assertIs(d, Disposition.RELAUNCH_STRONGER)

    def test_budget_exhausted_reformulates_when_already_strongest(self):
        d = suggest_disposition(
            failure_kind=FailureKind.BUDGET_EXHAUSTED,
            role=Role.PHD,
            agent_id="moderate",  # top of PhD whitelist
            catalog=_catalog(),
        )
        self.assertIs(d, Disposition.REFORMULATE)

    def test_intractable_jumps_to_pi(self):
        d = suggest_disposition(
            failure_kind=FailureKind.SUSPECTED_INTRACTABLE,
            role=Role.PHD,
            agent_id="cheapo",
            catalog=_catalog(),
        )
        self.assertIs(d, Disposition.ESCALATE_TO_PI)

    def test_senior_cannot_tirespin_forever(self):
        d = suggest_disposition(
            failure_kind=FailureKind.BUDGET_EXHAUSTED,
            role=Role.PHD,
            agent_id="cheapo",
            catalog=_catalog(),
            prior_relaunches=2,
            max_relaunches=2,
        )
        self.assertIs(d, Disposition.ESCALATE_TO_PI)


class TestEscalate(unittest.TestCase):
    def test_escalation_carries_reason_and_suggestion(self):
        result = TaskResult(
            task_id="task_1",
            status=TaskStatus.ESCALATED,
            summary="stuck",
            iterations=12,
        )
        esc = escalate(
            result=result,
            from_role=Role.PHD,
            to_role=Role.SENIOR,
            catalog=_catalog(),
            agent_id="cheapo",
        )
        self.assertEqual(esc.to_role, Role.SENIOR)
        self.assertIs(esc.diagnosis.failure_kind, FailureKind.BUDGET_EXHAUSTED)
        self.assertIs(esc.diagnosis.suggested_disposition, Disposition.RELAUNCH_STRONGER)

    def test_postdoc_intractable_flag_routes_to_pi(self):
        result = TaskResult(task_id="t", status=TaskStatus.ESCALATED, iterations=3)
        esc = escalate(
            result=result,
            from_role=Role.SENIOR,
            to_role=Role.PI,
            catalog=_catalog(),
            agent_id="moderate",
            intractable=True,
        )
        self.assertIs(esc.diagnosis.suggested_disposition, Disposition.ESCALATE_TO_PI)


if __name__ == "__main__":
    unittest.main()

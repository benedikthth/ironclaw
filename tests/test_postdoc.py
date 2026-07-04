"""Tests for the Postdoc review gate (deterministic reviewer + PhD stand-in)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.agents.postdoc import reviewed_run  # noqa: E402
from ironclaw.contracts import Task, TaskResult, TaskStatus  # noqa: E402
from ironclaw.control import FlagKind, PostdocReview, ReviewVerdict  # noqa: E402

TASK = Task(goal="produce a good artifact", acceptance=[])


def _passing_phd(task, ws):
    return TaskResult(task.id, TaskStatus.PASSED, summary="done", iterations=3)


class TestPostdoc(unittest.TestCase):
    def test_approve_passes_through(self):
        reviewer = lambda t, r, ws: PostdocReview(ReviewVerdict.APPROVE)
        with tempfile.TemporaryDirectory() as tmp:
            res = reviewed_run(task=TASK, phd_run=_passing_phd, reviewer=reviewer, workspace=tmp)
        self.assertIs(res.status, TaskStatus.PASSED)
        self.assertIn("postdoc: approved", res.summary)

    def test_reject_then_approve_carries_feedback(self):
        seen_feedback = []

        def phd(task, ws):
            seen_feedback.append(task.inputs.get("postdoc_feedback"))
            return TaskResult(task.id, TaskStatus.PASSED, iterations=3)

        rounds = iter(
            [
                PostdocReview(ReviewVerdict.REJECT, feedback="tighten the totals"),
                PostdocReview(ReviewVerdict.APPROVE),
            ]
        )
        reviewer = lambda t, r, ws: next(rounds)
        with tempfile.TemporaryDirectory() as tmp:
            res = reviewed_run(task=TASK, phd_run=phd, reviewer=reviewer, workspace=tmp)
        self.assertIs(res.status, TaskStatus.PASSED)
        # Round 0 had no feedback; round 1 saw the postdoc's note.
        self.assertEqual(seen_feedback, [None, "tighten the totals"])

    def test_persistent_reject_escalates_as_postdoc_rejected(self):
        reviewer = lambda t, r, ws: PostdocReview(ReviewVerdict.REJECT, feedback="still wrong")
        with tempfile.TemporaryDirectory() as tmp:
            res = reviewed_run(
                task=TASK, phd_run=_passing_phd, reviewer=reviewer, workspace=tmp, max_rounds=2
            )
        self.assertIs(res.status, TaskStatus.ESCALATED)
        self.assertEqual(res.failure_kind, "postdoc_rejected")

    def test_intractable_flag_goes_straight_up(self):
        reviewer = lambda t, r, ws: PostdocReview(
            ReviewVerdict.REJECT, feedback="can't be done", flag=FlagKind.SUSPECTED_INTRACTABLE
        )
        with tempfile.TemporaryDirectory() as tmp:
            res = reviewed_run(task=TASK, phd_run=_passing_phd, reviewer=reviewer, workspace=tmp)
        self.assertIs(res.status, TaskStatus.ESCALATED)
        self.assertEqual(res.failure_kind, "suspected_intractable")

    def test_phd_failure_is_not_reviewed(self):
        # A PhD that doesn't pass mechanical verification never reaches the postdoc.
        reviewed = []

        def reviewer(t, r, ws):
            reviewed.append(True)
            return PostdocReview(ReviewVerdict.APPROVE)

        def failing_phd(task, ws):
            return TaskResult(task.id, TaskStatus.ESCALATED, iterations=16)

        with tempfile.TemporaryDirectory() as tmp:
            res = reviewed_run(task=TASK, phd_run=failing_phd, reviewer=reviewer, workspace=tmp)
        self.assertIs(res.status, TaskStatus.ESCALATED)
        self.assertEqual(reviewed, [])  # reviewer never called


class TestPostdocSeniorComposition(unittest.TestCase):
    def test_postdoc_rejection_drives_senior_to_split(self):
        from ironclaw.control import Disposition, Role, escalate
        from ironclaw.control import AgentCatalog, AgentSpec

        catalog = AgentCatalog(
            [AgentSpec("cheapo", "anthropic", "claude-haiku-4-5", 1)], {Role.PHD: ["cheapo"]}
        )
        rejected = TaskResult(
            "t", TaskStatus.ESCALATED, failure_kind="postdoc_rejected", iterations=16
        )
        esc = escalate(
            result=rejected,
            from_role=Role.PHD,
            to_role=Role.SENIOR,
            catalog=catalog,
            agent_id="cheapo",
        )
        self.assertIs(esc.diagnosis.suggested_disposition, Disposition.SPLIT)


if __name__ == "__main__":
    unittest.main()

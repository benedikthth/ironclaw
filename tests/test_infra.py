"""Tests for the Infrastructure Manager (custodian of resources + skills)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironclaw.infra import (  # noqa: E402
    InfrastructureManager,
    ProbeResult,
    SkillCandidate,
)

BODY = "# How to\n\nStep 1: do the thing.\nStep 2: verify it worked.\n"


def _im() -> "tuple[InfrastructureManager, str, object]":
    tmp = tempfile.mkdtemp()
    return InfrastructureManager(os.path.join(tmp, "skills")), tmp


class TestSurvey(unittest.TestCase):
    def test_survey_inventories_and_registers_local_compute(self):
        im, tmp = _im()
        report = im.survey(paths=[tmp], disk_threshold=1.1)  # never trips a guardrail
        self.assertGreaterEqual(report["cpu"], 1)
        self.assertIn("local", {r.id for r in im.list_resources()})
        self.assertEqual(im.guardrails, [])

    def test_near_full_partition_becomes_a_guardrail(self):
        im, tmp = _im()
        im.survey(paths=[tmp], disk_threshold=0.0)  # any usage trips it
        self.assertTrue(im.guardrails)
        self.assertIn("do not write", im.guardrails[0].reason)

    def test_resource_pool_reflects_surveyed_compute(self):
        im, tmp = _im()
        im.survey(paths=[tmp], disk_threshold=1.1)
        pool = im.resource_pool()
        self.assertIn("cpu", pool.capacity)
        self.assertGreaterEqual(pool.capacity["cpu"], 1)


class TestOnboard(unittest.TestCase):
    def test_successful_probe_packages_a_skill_and_hides_secrets(self):
        im, _ = _im()
        descriptor = {"id": "cluster1", "kind": "slurm", "host": "hpc.example", "user": "me", "password": "hunter2"}

        def prober(d):
            return ProbeResult(
                ok=True,
                detail="sinfo returned 3 partitions",
                capacity={"slurm_slots": 128},
                skill=SkillCandidate(
                    "slurm cluster1",
                    "Enqueue and poll jobs on cluster1 via the IM credential handle.",
                    BODY,
                    source="onboard:slurm",
                ),
            )

        res = im.onboard_resource(descriptor, prober, credentials={"password": "hunter2"})
        self.assertTrue(res["ok"])
        self.assertIn("slurm_cluster1", im.list_skills())
        # Secret is retrievable only via the credential handle, never in spec/skill.
        self.assertEqual(im.credential_handle("cluster1"), {"password": "hunter2"})
        self.assertNotIn("password", im.resources["cluster1"].spec)
        with open(os.path.join(im.skills_root, "slurm_cluster1", "SKILL.md"), encoding="utf-8") as fh:
            self.assertNotIn("hunter2", fh.read())

    def test_failed_probe_registers_nothing(self):
        im, _ = _im()
        res = im.onboard_resource(
            {"id": "bad", "kind": "slurm"}, lambda d: ProbeResult(ok=False, detail="auth failed")
        )
        self.assertFalse(res["ok"])
        self.assertEqual(im.list_skills(), [])
        self.assertNotIn("bad", im.resources)


class TestSkillCustody(unittest.TestCase):
    def test_validation_rejects_thin_skills(self):
        im, _ = _im()
        reg = im.register_skill(SkillCandidate("x", "desc", "too short"))
        self.assertFalse(reg["ok"])

    def test_identical_duplicate_is_skipped_conflict_is_rejected(self):
        im, _ = _im()
        c = SkillCandidate("my skill", "does a thing", BODY)
        self.assertEqual(im.register_skill(c)["status"], "registered")
        self.assertEqual(im.register_skill(c)["status"], "duplicate")  # dedup
        conflict = SkillCandidate("my skill", "does a thing", BODY + "\ndifferent")
        self.assertFalse(im.register_skill(conflict)["ok"])  # name conflict

    def test_harvest_picks_up_phd_skill_and_dedupes(self):
        im, tmp = _im()
        ws = os.path.join(tmp, "phd_ws", "candidate_skills", "crawl")
        os.makedirs(ws)
        with open(os.path.join(ws, "SKILL.md"), "w", encoding="utf-8") as fh:
            fh.write("---\nname: crawl_site\ndescription: crawl a site politely\n---\n" + BODY)

        first = im.harvest(os.path.join(tmp, "phd_ws"), source="phd:t1")
        self.assertEqual(first["registered"], ["crawl_site"])
        second = im.harvest(os.path.join(tmp, "phd_ws"), source="phd:t1")  # over-notify
        self.assertEqual(second["registered"], [])
        self.assertEqual(second["skipped"], ["crawl_site"])


class TestSkillFailure(unittest.TestCase):
    def test_failure_quarantines_and_removes_from_listing(self):
        im, _ = _im()
        im.register_skill(SkillCandidate("flaky", "sometimes works", BODY))
        self.assertIn("flaky", im.list_skills())
        out = im.report_failure("flaky", "command not found: sbatch")
        self.assertEqual(out["status"], "quarantined")
        self.assertNotIn("flaky", im.list_skills())  # no PhD gets the broken recipe

    def test_fixer_repairs_and_republishes(self):
        im, _ = _im()
        im.register_skill(SkillCandidate("flaky", "sometimes works", BODY))
        fixed_body = BODY + "\nStep 3: install sbatch first.\n"
        out = im.report_failure("flaky", "no sbatch", fixer=lambda name, body, err: fixed_body)
        self.assertEqual(out["status"], "repaired")
        self.assertIn("flaky", im.list_skills())
        self.assertIn("install sbatch", im.registry.get("flaky").body)


if __name__ == "__main__":
    unittest.main()

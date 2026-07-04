"""Infrastructure Manager demo: wake, onboard, harvest, repair — all observable.

Talking to the custodian is just calling its methods (a CLI / Telegram / LLM
front-end maps utterances onto these). Runs with no API key and no real cluster:
the slurm probe is faked so the flow is visible end to end.

    python -m ironclaw.demo_infra
"""

from __future__ import annotations

import os
import tempfile

from .infra import InfrastructureManager, ProbeResult, SkillCandidate
from .observability import Recorder, console_sink

SLURM_SKILL = """# Run a job on cluster1 (slurm)

Ask the Infrastructure Manager for the `cluster1` credential handle; never inline
the password. Submit with `start_job(kind='slurm', command='sbatch ...', domain='slurm:gpu')`
and poll with `await_job`. The IM verified `sinfo` works on onboarding.
"""

PHD_SKILL = """---
name: polite_crawl
description: Crawl a site with rate limiting and a descriptive user agent.
---
# Polite crawl

Use one request/second, set a descriptive User-Agent, and honor robots.txt.
Persist results as JSONL so downstream tasks can query them.
"""


def main() -> None:
    rec = Recorder(sinks=[console_sink()])
    from .observability import using

    with tempfile.TemporaryDirectory() as tmp, using(rec):
        im = InfrastructureManager(os.path.join(tmp, "skills"))

        print("--- 1. wake & survey ---")
        im.survey(paths=[tmp], disk_threshold=0.0)  # force a guardrail for the demo

        print("\n--- 2. onboard: 'here is a slurm host, user, pw' ---")
        im.onboard_resource(
            {"id": "cluster1", "kind": "slurm", "host": "hpc.example", "user": "me", "password": "s3cret"},
            prober=lambda d: ProbeResult(
                ok=True,
                detail="ssh + sinfo ok, 3 partitions",
                capacity={"slurm_slots": 128},
                skill=SkillCandidate("slurm cluster1", "Enqueue/poll jobs on cluster1.", SLURM_SKILL, "onboard:slurm"),
            ),
            credentials={"password": "s3cret"},
        )

        print("\n--- 3. harvest a skill a PhD left behind (no notification) ---")
        ws = os.path.join(tmp, "phd_task_42", "candidate_skills", "crawl")
        os.makedirs(ws)
        with open(os.path.join(ws, "SKILL.md"), "w", encoding="utf-8") as fh:
            fh.write(PHD_SKILL)
        im.harvest(os.path.join(tmp, "phd_task_42"), source="phd:task_42")

        print("\n--- 4. a skill fails in use -> quarantine + repair ---")
        im.report_failure(
            "slurm cluster1",
            "sbatch: command not found",
            fixer=lambda name, body, err: body + "\n(Repair: load the slurm module first: `module load slurm`.)",
        )

        print("\n--- state ---")
        print("skills:   ", im.list_skills())
        print("resources:", [f"{r.id}({r.kind})" for r in im.list_resources()])
        print("guardrails:", [g.reason for g in im.guardrails])


if __name__ == "__main__":
    main()

"""The Infrastructure Manager — custodian of resources and skills.

Unlike the org hierarchy (PI/Senior/Postdoc/PhD, strict layer comms), the IM is a
sibling service everyone consults, and the *human talks to it directly*. Its
public methods are that admin channel; a CLI / Telegram / LLM front-end just maps
utterances onto them.

Responsibilities:

- **Wake & survey** the host on first run: inventory CPU/mem/disk/GPU, and flag
  what must not be touched (a near-full partition becomes a guardrail).
- **Onboard a resource by conversation**: given e.g. a slurm host + credentials,
  *probe* it to learn how it behaves, then **package a clean skill** so PhDs never
  start from zero. Secrets are held as credential handles, never written into the
  skill folder.
- **Custody of the skills folder**: validate + dedupe on every registration (the
  folder must stay clean), proactively **harvest** skills a PhD produced (the
  dumb ones won't notify; the eager ones over-notify — dedup handles both), and
  on a **skill failure**, quarantine it and drive a repair.
- **Declare resources** to the scheduler (`resource_pool()`), closing the loop:
  the IM is custodian of what exists, the scheduler enforces admission.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable

from .observability import active_recorder
from .scheduler import ResourcePool
from .skills import SkillRegistry, _parse_skill_md

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    s = _SLUG.sub("_", name.strip().lower()).strip("_")
    return s or "skill"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class Resource:
    id: str
    kind: str  # "compute" | "gpu" | "storage" | "slurm" | ...
    capacity: dict[str, float] = field(default_factory=dict)  # feeds the scheduler
    spec: dict = field(default_factory=dict)  # non-secret connection details
    status: str = "ok"


@dataclass
class Guardrail:
    target: str  # a path or resource id
    reason: str


@dataclass
class SkillCandidate:
    name: str
    description: str
    body: str
    source: str = "manual"  # "survey" | "onboard:slurm" | "phd:<task_id>" | ...


@dataclass
class ProbeResult:
    """What a prober reports back after testing a resource."""

    ok: bool
    detail: str = ""
    capacity: dict[str, float] = field(default_factory=dict)
    skill: SkillCandidate | None = None  # the skill to package on success


Prober = Callable[[dict], ProbeResult]
# A fixer takes (name, current_body, error) and returns a repaired body, or None.
Fixer = Callable[[str, str, str], "str | None"]


# --------------------------------------------------------------------------- #
# Infrastructure Manager
# --------------------------------------------------------------------------- #


class InfrastructureManager:
    def __init__(self, skills_root: str) -> None:
        self.skills_root = skills_root
        os.makedirs(skills_root, exist_ok=True)
        self.registry = SkillRegistry(skills_root)
        self.resources: dict[str, Resource] = {}
        self.guardrails: list[Guardrail] = []
        self._credentials: dict[str, dict] = {}  # resource_id -> secrets (in-memory only)

    # -- wake & survey ------------------------------------------------------ #

    def survey(self, paths: list[str] | None = None, *, disk_threshold: float = 0.9) -> dict:
        """First-wake look around: inventory compute + storage, raise guardrails
        for anything unsafe to touch (a near-full partition)."""
        rec = active_recorder()
        cpu = os.cpu_count() or 1
        mem_gb = _total_memory_gb()
        gpus = _detect_gpus()

        self.resources["local"] = Resource(
            id="local",
            kind="compute",
            capacity={"cpu": float(cpu), "mem_gb": mem_gb, "gpu": float(gpus)},
        )

        checked = []
        for path in paths or ["/"]:
            try:
                usage = shutil.disk_usage(path)
            except OSError:
                continue
            frac = usage.used / usage.total if usage.total else 0.0
            checked.append({"path": path, "used_frac": round(frac, 3)})
            if frac >= disk_threshold:
                self.guardrails.append(
                    Guardrail(path, f"{frac:.0%} full — do not write large artifacts here")
                )

        report = {
            "cpu": cpu,
            "mem_gb": mem_gb,
            "gpus": gpus,
            "disks": checked,
            "guardrails": [g.target for g in self.guardrails],
        }
        rec.emit("infra.survey", role="infra", message="host surveyed", data=report)
        return report

    # -- onboard a resource by conversation --------------------------------- #

    def onboard_resource(
        self, descriptor: dict, prober: Prober, *, credentials: dict | None = None
    ) -> dict:
        """Test an offered resource; on success, package a skill and register both
        the resource and the skill. Credentials are stored as a handle, never
        written into the skill folder."""
        rec = active_recorder()
        rid = descriptor.get("id") or descriptor.get("kind", "resource")
        probe = prober(descriptor)
        if not probe.ok:
            rec.emit("infra.onboard_failed", role="infra", message=rid, data={"detail": probe.detail})
            return {"ok": False, "detail": probe.detail}

        if credentials:
            self._credentials[rid] = credentials  # handle only; kept out of skills
        self.resources[rid] = Resource(
            id=rid, kind=descriptor.get("kind", "resource"), capacity=probe.capacity, spec=_no_secrets(descriptor)
        )
        rec.emit("infra.resource_added", role="infra", message=rid,
                 data={"kind": descriptor.get("kind"), "capacity": probe.capacity})

        registered = None
        if probe.skill is not None:
            reg = self.register_skill(probe.skill)
            registered = reg.get("name") if reg["ok"] else None
        return {"ok": True, "resource": rid, "skill": registered, "detail": probe.detail}

    def credential_handle(self, resource_id: str) -> dict | None:
        """PhDs never see raw secrets; they ask the IM to act with a handle."""
        return self._credentials.get(resource_id)

    # -- skill custody ------------------------------------------------------ #

    def register_skill(self, candidate: SkillCandidate, *, overwrite: bool = False) -> dict:
        """Validate, dedupe, and write a clean SKILL.md. Keeps the folder tidy:
        identical duplicates are skipped, name conflicts are rejected."""
        rec = active_recorder()
        slug = slugify(candidate.name)
        problem = _validate(candidate, slug)
        if problem:
            rec.emit("skill.rejected", role="infra", message=slug, data={"reason": problem})
            return {"ok": False, "reason": problem}

        existing = self.registry.get(slug)
        if existing is not None and not overwrite:
            if existing.body.strip() == candidate.body.strip():
                rec.emit("skill.duplicate", role="infra", message=slug)
                return {"ok": True, "name": slug, "status": "duplicate"}
            rec.emit("skill.rejected", role="infra", message=slug, data={"reason": "name conflict"})
            return {"ok": False, "reason": "name conflict"}

        _write_skill(self.skills_root, slug, candidate)
        self.registry.reload()
        rec.emit("skill.registered", role="infra", message=slug, data={"source": candidate.source})
        return {"ok": True, "name": slug, "status": "registered"}

    def harvest(self, workspace: str, *, source: str = "harvest") -> dict:
        """Scan a PhD workspace for skill-able artifacts (SKILL.md files) and
        register the good ones. Proactive: the dumb PhD never had to notify us;
        dedup absorbs the eager PhD that notified three times."""
        found = _find_skill_files(workspace)
        registered, skipped = [], []
        for path in found:
            with open(path, "r", encoding="utf-8") as fh:
                parsed = _parse_skill_md(fh.read(), path)
            cand = SkillCandidate(parsed.name, parsed.description, parsed.body, source=source)
            reg = self.register_skill(cand)
            (registered if reg.get("status") == "registered" else skipped).append(slugify(cand.name))
        active_recorder().emit(
            "skill.harvested", role="infra", message=workspace,
            data={"registered": registered, "skipped": skipped},
        )
        return {"registered": registered, "skipped": skipped}

    def report_failure(self, skill_name: str, error: str, *, fixer: Fixer | None = None) -> dict:
        """A skill failed at use time. Quarantine it (so no PhD is handed a broken
        recipe), notify, and drive a repair. The custodian must fix it — a fixer
        (LLM or human) may return a repaired body; otherwise it stays quarantined
        as an open repair task."""
        rec = active_recorder()
        slug = slugify(skill_name)
        skill = self.registry.get(slug)
        rec.emit("skill.failed", role="infra", message=slug, data={"error": error})
        if skill is None:
            return {"ok": False, "reason": "unknown skill"}

        body = skill.body
        _quarantine(self.skills_root, slug)
        self.registry.reload()
        rec.emit("skill.quarantined", role="infra", message=slug)

        if fixer is not None:
            repaired = fixer(slug, body, error)
            if repaired:
                _unquarantine(self.skills_root, slug)
                _write_skill(
                    self.skills_root, slug,
                    SkillCandidate(slug, skill.description, repaired, source="repair"),
                )
                self.registry.reload()
                rec.emit("skill.repaired", role="infra", message=slug)
                return {"ok": True, "status": "repaired"}

        rec.emit("skill.repair_needed", role="infra", message=slug, data={"error": error})
        return {"ok": True, "status": "quarantined"}

    # -- views / scheduler bridge ------------------------------------------ #

    def list_skills(self) -> list[str]:
        return [s.name for s in self.registry.list()]

    def list_resources(self) -> list[Resource]:
        return list(self.resources.values())

    def resource_pool(self) -> ResourcePool:
        """Declare compute capacity to the scheduler — the IM owns *what exists*,
        the scheduler enforces *what runs*."""
        cap: dict[str, float] = {}
        for r in self.resources.values():
            if r.kind == "compute":
                for k, v in r.capacity.items():
                    cap[k] = cap.get(k, 0.0) + v
        return ResourcePool(cap)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_SECRET_KEYS = {"password", "pw", "token", "secret", "key", "passphrase"}


def _no_secrets(descriptor: dict) -> dict:
    return {k: v for k, v in descriptor.items() if k.lower() not in _SECRET_KEYS}


def _validate(candidate: SkillCandidate, slug: str) -> str | None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", slug):
        return "invalid name"
    if not candidate.description.strip():
        return "missing description"
    if len(candidate.body.strip()) < 20:
        return "body too thin to be a real skill"
    return None


def _write_skill(root: str, slug: str, candidate: SkillCandidate) -> None:
    folder = os.path.join(root, slug)
    os.makedirs(folder, exist_ok=True)
    front = f"---\nname: {slug}\ndescription: {candidate.description.strip()}\n---\n\n"
    tmp = os.path.join(folder, "SKILL.md.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(front + candidate.body.strip() + "\n")
    os.replace(tmp, os.path.join(folder, "SKILL.md"))  # atomic — no half-written skills


def _quarantine(root: str, slug: str) -> None:
    qdir = os.path.join(root, ".quarantine")
    os.makedirs(qdir, exist_ok=True)
    src = os.path.join(root, slug)
    if os.path.isdir(src):
        dst = os.path.join(qdir, slug)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)


def _unquarantine(root: str, slug: str) -> None:
    src = os.path.join(root, ".quarantine", slug)
    if os.path.isdir(src):
        shutil.rmtree(src)  # drop the broken copy; caller rewrites a clean one


def _find_skill_files(workspace: str) -> list[str]:
    hits = []
    for dirpath, dirnames, filenames in os.walk(workspace):
        if ".quarantine" in dirpath:
            continue
        if "SKILL.md" in filenames:
            hits.append(os.path.join(dirpath, "SKILL.md"))
    return sorted(hits)


def _total_memory_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def _detect_gpus() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return sum(1 for line in out.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return 0

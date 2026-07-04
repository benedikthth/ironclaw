# Ironclaw

A multi-agent **research institute** for tackling difficult, long-running
projects that need many different compute approaches (training, simulation,
crawling, data transforms, queued/HPC jobs). Each hard problem gets a **lab**.

## The organization

Every layer talks only to its direct superior and its direct subordinate. The
PI and a PhD never speak — intent is carried down as an explicit, checkable
**contract**, and results flow up as **artifacts** (Markdown reports, code,
queryable data), never raw transcripts.

```
Lab (one hard problem)
  PI ───────────────── decomposes the problem into projects; adds/kills projects
   └─ Senior Researcher ─ owns 1..n projects; authors tasks with acceptance criteria
       └─ PhD ─────────── owns one task; does the actual work; self-verifies
           └─ Postdoc ─── reviews/mentors the PhD (1..n PhDs)

  Infrastructure Manager (sibling service) ── custodian of skills + resources
                                              (compute, storage, GPU, slurm)
```

**Why a hierarchy of cheap models can be reliable:** reliability lives in the
*harness*, not the model. Bounded loops, structured tools, reusable **skills**
that encode a working procedure once, a forced **verification gate**, and hard
iteration budgets turn "tirespinning" into a definite escalation instead of an
infinite spend.

## Status

Implemented and tested end-to-end with **no API key or network** — a
deterministic `FakeProvider` drives the harness, so the mechanics are
CI-testable and token-free. 16 tests, stdlib-only.

### PhD vertical slice ✅

The PhD is ~90% of the real work, so it landed first.

- Provider-neutral LLM interface (`ironclaw/providers/`) — Anthropic / OpenAI /
  OpenRouter / self-hosted adapters slot in behind one shape.
- The PhD agent loop (`ironclaw/runtime/loop.py`): model ⇄ tools → submit →
  **verify against acceptance criteria** → retry or escalate.
- Tools (`ironclaw/tools/`): `write_file`, `read_file`, `python_exec`,
  `invoke_skill`, `start_job`, `await_job`, `submit_result` — sandboxed to a
  task workspace.
- Skills registry (`ironclaw/skills.py`): progressive-disclosure `SKILL.md`
  folders; the surface the Infrastructure Manager will be custodian of.
- Mechanical verification (`ironclaw/verify.py`): `file_exists` + `command`
  checks; "done" is objective.

### Durable async (Slice 2) ✅

The long-running case (train / slurm-enqueue / poll / sleep) can't live in a
chat loop. So the PhD **suspends** on a background job — full state written to
disk — and is **resumed** when the job finishes, surviving process restarts.

- `ironclaw/jobs.py`: `LocalProcessBackend` is restart-safe via sentinel files
  (a spawned job records its own output + exit code, discoverable by any later
  process with no live handle). Real slurm/pufferlib backends implement the same
  tiny `submit`/`poll` protocol. `FakeJobBackend` makes the suspend/resume path
  deterministic in tests.
- `ironclaw/runtime/state.py`: the irreducible state a suspended PhD persists
  (task, message log, iteration count, awaited job) — provider and tools are
  rebuilt on resume.
- The demo suspends/resumes several times across a real 1s job, then passes.

### Control model ✅

The institute's supervision rules as typed contracts + a deterministic baseline
policy (`ironclaw/control.py`):

- **Three gates by ability-to-act** — Postdoc owns *submission* accept/reject,
  Senior owns *task* disposition (reformulate / split / relaunch-stronger /
  escalate), PI owns *project* stop.
- **Per-role agent whitelist** — supervisors assign a subordinate's agent from
  its whitelist; `next_stronger()` implements "relaunch with a smarter agent".
- **Structured failure** — `Diagnosis` (why + suggested disposition) wrapped in
  an `Escalation`; a Senior never receives a bare `FAILED`.
- **Bounded** — after N relaunches the only move left is up to the PI, so a
  Senior can't tirespin.

### Run it

```bash
python -m ironclaw.demo         # PhD slice: skill reuse → transform → verify
python -m ironclaw.demo_async   # durable suspend/resume across a real job
python -m unittest discover -s tests
```

## Roadmap

- Real provider adapters (Anthropic first) behind the neutral interface.
- Wire the control model into a running **Senior** loop (consume `Escalation`,
  act on the `Disposition`) and a **Postdoc** review gate.
- **PI** decomposition (problem → projects) and the lab lifecycle.
- The **Infrastructure Manager** agent + resource registry (GPU/storage/slurm
  leases) on top of the skills custodian surface.
- A backgrounded daemon that owns the durable task store and drives resume;
  then UIs: web, TUI, Telegram over one API.

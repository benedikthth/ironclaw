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
CI-testable and token-free. 40 tests, stdlib-only (the `anthropic` adapter is an
optional extra). The full org — PI → Senior → Postdoc → PhD — runs top to bottom,
emitting a live event stream the whole way.

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

### Resource-aware scheduler (Slice 3) ✅

Two agents shouldn't both grab 100% of one GPU — but a CPU task shouldn't wait
behind a GPU task either. Because a PhD suspends to disk while a job runs, the
scheduler doesn't gate reasoning loops (cheap, I/O-bound); it gates the heavy
*jobs* against a declared resource pool (`ironclaw/scheduler.py`):

- **Backfill admission** — scans the pending queue in priority order and admits
  every job that fits, *skipping* (not stopping at) ones that don't. That's the
  fix for head-of-line blocking: a CPU job runs immediately while a GPU job sits
  queued for a lease.
- **Leases** against a `ResourcePool` `{gpu, cpu, mem_gb, ...}`; released on
  completion so the next queued job admits.
- **Resource domains** — `local` jobs are gated by our pool; `slurm:<partition>`
  jobs are handed to the cluster's own scheduler (we don't reimplement Slurm).
- Drop-in for the PhD's `await_job`: "queued" and "running" both read as
  not-done, so the PhD just stays suspended. `tick()` drives admission + reaps
  completions and returns which jobs finished (so a daemon knows whom to resume).
- v2: preemption and fair-share weighting (backfill can starve a large job;
  Slurm-style reservations are the intended fix).

### Real provider adapter ✅

`ironclaw/providers/anthropic.py` drives a live Claude model behind the neutral
interface — deliberately thin (one Messages call per turn, no sampling params,
no thinking config) so a cheap PhD model and an expensive PI model use identical
code. Proven end-to-end: **Haiku 4.5 drives the PhD slice to a verified pass** —
consulting the skill, writing the transform, and submitting on its own — with no
scripted provider (`examples/run_phd_live.py`).

> Running it live surfaced a real harness bug the fake provider never could: the
> model did the work correctly but hit the iteration budget before calling
> `submit_result`, and the loop escalated a *finished* task. Fixed — mechanical
> verification, not the model's submit call, is the source of truth: at budget
> exhaustion the loop passes work that satisfies the acceptance contract.

### Observability & human-in-the-loop ✅

Every layer emits typed events to an **ambient event stream**
(`ironclaw/observability.py`) — the single substrate the planned web/TUI/Telegram
UIs will consume. The recorder is ambient via a context var, so even a tool can
`active_recorder().emit(...)` without threading a parameter through every
signature; an orchestration entry point wraps its run in `using(recorder)`.

- **Sinks**: `jsonl_sink` (durable audit log / UI feed), `console_sink` (live
  progress). The `demo_lab` streams the whole org as it runs — relaunch ladders,
  dispositions with their failure kind, the escalation to the PI, the PI's call.
- **Cost view**: providers report per-turn token usage on the stream;
  `total_usage` / `usage_by_agent` aggregate it — the cheap-vs-strong spend is
  visible per agent. `examples/run_phd_live.py` prints real token cost.
- **Human-in-the-loop**: the PI consults an `Overseer` on every task a Senior
  gives up on — this is "the PI steps in if direction is drifting", made
  pluggable. `AutoOverseer` (default) runs a lab unattended; `ConsoleOverseer`
  prompts to accept / retry / abandon. A UI plugs in at the same seam.

### Full org: PI → Senior → Postdoc → PhD ✅

The whole institute runs top to bottom. The **PI** (`ironclaw/agents/pi.py`)
decomposes a problem into projects, hands each project's tasks to the Senior
layer, aggregates results, and owns the top gate — a task a Senior gives up on
lands in the PI's action list. The **Postdoc** (`ironclaw/agents/postdoc.py`) is
the submission-quality gate between PhD and Senior: blocking, iteration-boxed,
reviewing for the quality the acceptance criteria couldn't encode, with the
narrow typed flag-up channel (misspecified / suspected-intractable).

Each layer is unaware of the ones below it — a PI hands the Senior a `runner`,
and in production that runner is the Postdoc-gated real-PhD runner, so **one call
drives the entire institute.** The `demo_lab` shows all three behaviors in one
lab: a task solved outright, one the Senior relaunches onto a stronger agent, and
an impossible one that bubbles to the PI:

```
project characterize [passed]
  task measure     [passed]    via cheapo
project stress [escalated]
  task hard        [passed]    via cheapo -> moderate
  task impossible  [escalated] via cheapo -> moderate
PI action list: impossible -> escalate_to_pi
```

### Senior supervision loop ✅

`ironclaw/agents/senior.py` ties the control model to the real executor and
closes the failure loop. The Senior runs a PhD; on failure it consumes the
structured `Escalation`, reads the suggested `Disposition`, and acts:

- **RELAUNCH_STRONGER** → re-assign the PhD to the next agent up its whitelist,
  with a larger iteration budget.
- **REFORMULATE / RETRY_SAME / SPLIT** → grant more budget and re-run (a
  mechanical fallback; content-authoring reformulate/split is the LLM-Senior
  seam, and slots in here).
- **ESCALATE_TO_PI** → hand the failure up to the PI — the only non-pass exit.

Termination is guaranteed, not hoped for: every non-pass attempt bumps the
relaunch count, and the baseline policy escalates to the PI once it hits the cap,
so **a Senior cannot tirespin — it escalates.** The demo shows the ladder:
`cheapo → moderate → strong → passed`, budget growing each rung. Fully tested
with a deterministic runner; `anthropic_phd_runner()` drives real PhDs.

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
python -m ironclaw.demo            # PhD slice: skill reuse → transform → verify
python -m ironclaw.demo_async      # durable suspend/resume across a real job
python -m ironclaw.demo_scheduler  # backfill: CPU job not blocked behind GPU jobs
python -m ironclaw.demo_senior     # Senior closes the failure loop (relaunch ladder)
python -m ironclaw.demo_lab        # full institute: PI -> Senior -> Postdoc -> PhD
python -m unittest discover -s tests

# live: a real cheap model drives the PhD slice (needs an API key)
ANTHROPIC_API_KEY=... python examples/run_phd_live.py claude-haiku-4-5
```

## Roadmap

- The **daemon**: an always-on service that owns the durable task store + the
  scheduler, ticks admission, and drives resume when jobs finish (today the
  demos play scheduler by hand). Persist the scheduler queue for cross-restart
  recovery.
- OpenAI / OpenRouter / self-hosted adapters behind the neutral interface (the
  Anthropic one is done), plus an inference-budget resource dimension
  (tokens/rate/$ per provider) — same lease pattern as compute.
- **LLM-driven authoring** at the seams (all wired, deterministic today): the PI
  authoring projects from a raw problem statement; the Senior authoring
  reformulated/split tasks; the Postdoc's `llm_reviewer` in the loop.
- The **Infrastructure Manager** agent + resource registry (GPU/storage/slurm
  leases) on top of the skills custodian surface.
- UIs: web, TUI, Telegram — all consume the `observability` event stream and
  drive the `Overseer` seam that already exist.

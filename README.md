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

## Status — PhD vertical slice ✅

The PhD is ~90% of the real work, so it lands first. Implemented and tested
end-to-end (no API key, no network — a deterministic `FakeProvider` drives the
harness):

- Provider-neutral LLM interface (`ironclaw/providers/`) — Anthropic / OpenAI /
  OpenRouter / self-hosted adapters slot in behind one shape.
- The PhD agent loop (`ironclaw/runtime/loop.py`): model ⇄ tools → submit →
  **verify against acceptance criteria** → retry or escalate.
- Tools: `write_file`, `read_file`, `python_exec`, `invoke_skill`,
  `submit_result` — all sandboxed to a task workspace.
- Skills registry (`ironclaw/skills.py`): progressive-disclosure `SKILL.md`
  folders; the surface the Infrastructure Manager will be custodian of.
- Mechanical verification (`ironclaw/verify.py`): `file_exists` + `command`
  checks; "done" is objective.

### Run it

```bash
python -m ironclaw.demo               # end-to-end slice, prints the TaskResult
python -m unittest discover -s tests  # the suite
```

## Roadmap

- **Slice 2 — durable async.** `start_job`/`await_job` with persisted state and
  resume-on-completion for slurm/train/poll/sleep (the long-running case).
- Real provider adapters + role→model cost policy.
- The management layers (Senior task authoring, Postdoc review gate, PI
  decomposition) wrapped around the PhD executor.
- The Infrastructure Manager agent + resource registry (GPU/storage/slurm leases).
- UIs: web, TUI, Telegram over one backend API.

### Open design decision

**Postdoc review** — recommended default is a *blocking gate*, iteration-boxed
(N rounds, then auto-escalate to the Senior), with the verdict logged so the
Senior sees an independent QA signal. Not yet wired in.

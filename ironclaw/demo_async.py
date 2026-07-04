"""Slice 2 demo: a PhD suspends on a real background job and is resumed.

Uses the real LocalProcessBackend with an actual `sleep` job. The PhD submits the
job, awaits it, and *suspends* (state written to disk). A driver loop then plays
the role of the scheduler: it polls and resumes with a **fresh** backend each
time, proving the resume path recovers purely from on-disk sentinels — i.e. it
survives the PhD process dying.

    python -m ironclaw.demo_async
"""

from __future__ import annotations

import os
import tempfile
import time

from .agents.phd import resume_phd, run_phd
from .contracts import AcceptanceCriterion, CheckKind, Task, TaskStatus
from .jobs import LocalProcessBackend
from .providers.base import AssistantTurn, ToolCall
from .providers.fake import FakeProvider

TASK = Task(
    goal="Compute the answer with a background job and record it.",
    acceptance=[
        AcceptanceCriterion("result exists", CheckKind.FILE_EXISTS, "result.txt"),
        AcceptanceCriterion("answer is 42", CheckKind.COMMAND, "grep -q 42 result.txt"),
    ],
)


def _script():
    return [
        AssistantTurn(
            text="Queuing the computation as a durable job.",
            tool_calls=[ToolCall("s", "start_job", {"kind": "compute", "command": "sleep 1; echo 42"})],
        ),
        AssistantTurn(
            text="Waiting for the job (I will suspend until it finishes).",
            tool_calls=[ToolCall("w", "await_job", {"job_id": "job_1"})],
        ),
        AssistantTurn(
            text="Job done; recording the result.",
            tool_calls=[ToolCall("r", "write_file", {"path": "result.txt", "content": "42\n"})],
        ),
        AssistantTurn(
            text="Submitting.",
            tool_calls=[
                ToolCall("d", "submit_result", {"summary": "computed 42 via a background job", "artifacts": [{"path": "result.txt", "kind": "data"}]})
            ],
        ),
    ]


def run(workspace: str):
    state_path = os.path.join(workspace, "state.json")
    backend = LocalProcessBackend(workspace)
    out = run_phd(
        task=TASK,
        provider=FakeProvider(_script()),
        workspace=workspace,
        jobs=backend,
        state_path=state_path,
    )
    suspends = 0
    while out.result.status is TaskStatus.WAITING:
        suspends += 1
        time.sleep(0.2)  # the scheduler's poll interval (the PhD itself is idle)
        # Fresh backend each resume => recovery is purely from on-disk state.
        out = resume_phd(state_path=state_path, provider=FakeProvider(_script()))
    print(f"(suspended/resumed {suspends}x while the job ran)\n")
    return out.result


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        print(run(tmp).to_json())


if __name__ == "__main__":
    main()

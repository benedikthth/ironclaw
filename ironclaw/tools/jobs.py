"""Job tools: start durable background work, and await it (suspending if pending).

``start_job`` submits and returns immediately. ``await_job`` is special — the
agent loop intercepts it: if the job is finished the loop feeds the result back
inline; if not, the loop *suspends* the whole PhD to disk and returns control, to
be resumed when the job completes. The AwaitJob object here exists to advertise
the tool's schema to the model; its ``run`` is never called by the loop.
"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolError


class StartJob(Tool):
    name = "start_job"
    description = (
        "Submit a long-running or queued shell command as a durable background "
        "job (training runs, slurm enqueue, crawls, sleeps). Returns a job_id "
        "immediately. Do NOT block on it inline — call await_job to wait."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "e.g. 'train', 'slurm', 'crawl'."},
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["kind", "command"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if ctx.jobs is None:
            raise ToolError("no job backend available")
        job_id = ctx.jobs.submit(args["kind"], args["command"])
        return f"submitted {args['kind']} job: {job_id}"


class AwaitJob(Tool):
    name = "await_job"
    description = (
        "Wait for a background job to finish and receive its output. If it is "
        "still running you will be suspended and resumed automatically when it "
        "completes — this is normal and costs nothing while you wait."
    )
    input_schema = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:  # pragma: no cover
        # The loop intercepts await_job; if this ever runs, jobs are misconfigured.
        raise ToolError("await_job must be handled by the agent loop")

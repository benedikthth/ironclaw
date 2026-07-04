"""The institute's control model: roles, agent assignment, and failure disposition.

This encodes the decisions that govern how work is supervised, independent of any
single agent's behavior:

- Three gates, each owned by the layer that can actually act on its judgment:
    submission -> Postdoc (accept / reject-with-reason)
    task       -> Senior  (reformulate / split / relaunch-stronger / escalate)
    project    -> PI       (kill / reshape / spawn)
  Authority follows ability-to-act: a reviewer that cannot split or reassign does
  not own the stop decision — it informs the layer that can.

- Every supervisor picks which agent to assign to a subordinate, bounded by a
  per-role whitelist. "Relaunch with a smarter agent" = the Senior re-assigns the
  PhD higher up its whitelist (and/or grants a bigger budget).

- Failure is never a bare status: it carries a structured Diagnosis so the Senior
  knows *why* and gets a suggested disposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .contracts import CheckResult, TaskResult, TaskStatus


class Role(str, Enum):
    PI = "pi"
    SENIOR = "senior"
    PHD = "phd"
    POSTDOC = "postdoc"


# --------------------------------------------------------------------------- #
# Agent assignment: supervisors pick from a per-role whitelist.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentSpec:
    """A concrete model an agent can run on. ``cost_tier`` orders "smarter"."""

    id: str
    provider: str  # "anthropic" | "openai" | "openrouter" | "selfhosted"
    model: str
    cost_tier: int  # higher == stronger/pricier; used to bump on relaunch


class AgentCatalog:
    """Known agents plus each role's whitelist. The supervisor's menu."""

    def __init__(
        self, agents: list[AgentSpec], whitelists: dict[Role, list[str]]
    ) -> None:
        self._by_id = {a.id: a for a in agents}
        self._whitelists = whitelists

    def get(self, agent_id: str) -> AgentSpec:
        return self._by_id[agent_id]

    def allowed(self, role: Role) -> list[AgentSpec]:
        """Whitelisted agents for a role, weakest first."""
        ids = self._whitelists.get(role, [])
        return sorted((self._by_id[i] for i in ids), key=lambda a: a.cost_tier)

    def is_allowed(self, role: Role, agent_id: str) -> bool:
        return agent_id in self._whitelists.get(role, [])

    def next_stronger(self, role: Role, current_id: str) -> AgentSpec | None:
        """The cheapest whitelisted agent strictly stronger than the current one."""
        current = self._by_id[current_id]
        stronger = [a for a in self.allowed(role) if a.cost_tier > current.cost_tier]
        return stronger[0] if stronger else None


@dataclass
class AgentAssignment:
    """A supervisor's choice of agent + budget for a subordinate task."""

    role: Role
    agent_id: str
    budget_iterations: int = 12

    def validate(self, catalog: AgentCatalog) -> None:
        if not catalog.is_allowed(self.role, self.agent_id):
            raise ValueError(
                f"agent {self.agent_id!r} not whitelisted for role {self.role.value}"
            )


# --------------------------------------------------------------------------- #
# Failure disposition.
# --------------------------------------------------------------------------- #


class FailureKind(str, Enum):
    BUDGET_EXHAUSTED = "budget_exhausted"
    VERIFICATION_FAILED = "verification_failed"
    POSTDOC_REJECTED = "postdoc_rejected"
    SUSPECTED_INTRACTABLE = "suspected_intractable"


class Disposition(str, Enum):
    ACCEPT = "accept"
    RETRY_SAME = "retry_same"
    REFORMULATE = "reformulate"
    SPLIT = "split"
    RELAUNCH_STRONGER = "relaunch_stronger"
    ESCALATE_TO_PI = "escalate_to_pi"


@dataclass
class Diagnosis:
    """Why a task failed, plus a suggested next move. Consumed by the Senior."""

    failure_kind: FailureKind
    attempts: int
    failing_checks: list[CheckResult] = field(default_factory=list)
    notes: str = ""
    suggested_disposition: Disposition = Disposition.RETRY_SAME


@dataclass
class Escalation:
    """A failure handed up the chain, from one role to its direct superior."""

    task_id: str
    from_role: Role
    to_role: Role
    diagnosis: Diagnosis


# --------------------------------------------------------------------------- #
# Postdoc review (axis 2): feedback goes to the PhD; a narrow flag goes up.
# --------------------------------------------------------------------------- #


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class FlagKind(str, Enum):
    """The only two reasons a postdoc interrupts sideways to the Senior."""

    MISSPECIFIED = "misspecified"  # acceptance criteria are insufficient/wrong
    SUSPECTED_INTRACTABLE = "suspected_intractable"


@dataclass
class PostdocReview:
    verdict: ReviewVerdict
    feedback: str = ""  # -> PhD
    flag: FlagKind | None = None  # -> Senior, optional


# --------------------------------------------------------------------------- #
# Policy: how a Senior turns a Diagnosis into a Disposition. A deterministic
# baseline the Senior agent can later override with judgment.
# --------------------------------------------------------------------------- #


def suggest_disposition(
    *,
    failure_kind: FailureKind,
    role: Role,
    agent_id: str,
    catalog: AgentCatalog,
    prior_relaunches: int = 0,
    max_relaunches: int = 2,
) -> Disposition:
    """Baseline Senior policy. Bounded so a Senior can't tirespin: after
    ``max_relaunches`` without success, the only move left is up to the PI."""
    if failure_kind is FailureKind.SUSPECTED_INTRACTABLE:
        return Disposition.ESCALATE_TO_PI
    if prior_relaunches >= max_relaunches:
        return Disposition.ESCALATE_TO_PI
    if failure_kind is FailureKind.BUDGET_EXHAUSTED:
        if catalog.next_stronger(role, agent_id) is not None:
            return Disposition.RELAUNCH_STRONGER
        return Disposition.REFORMULATE
    if failure_kind is FailureKind.VERIFICATION_FAILED:
        return Disposition.REFORMULATE
    if failure_kind is FailureKind.POSTDOC_REJECTED:
        return Disposition.SPLIT
    return Disposition.RETRY_SAME


def escalate(
    *,
    result: TaskResult,
    from_role: Role,
    to_role: Role,
    catalog: AgentCatalog,
    agent_id: str,
    prior_relaunches: int = 0,
    intractable: bool = False,
) -> Escalation:
    """Build the structured Escalation a supervisor receives on failure.

    ``intractable`` is set when a Postdoc flagged SUSPECTED_INTRACTABLE — the one
    signal that jumps straight to the PI regardless of budget.
    """
    failing = [c for c in result.checks if not c.passed]
    if intractable:
        failure_kind = FailureKind.SUSPECTED_INTRACTABLE
    elif result.status is TaskStatus.ESCALATED:
        failure_kind = FailureKind.BUDGET_EXHAUSTED
    else:
        failure_kind = FailureKind.VERIFICATION_FAILED

    disposition = suggest_disposition(
        failure_kind=failure_kind,
        role=from_role,
        agent_id=agent_id,
        catalog=catalog,
        prior_relaunches=prior_relaunches,
    )
    notes = result.summary or ""
    if failing:
        notes += " | unmet: " + "; ".join(c.criterion.description for c in failing)
    diagnosis = Diagnosis(
        failure_kind=failure_kind,
        attempts=result.iterations,
        failing_checks=failing,
        notes=notes.strip(" |"),
        suggested_disposition=disposition,
    )
    return Escalation(
        task_id=result.task_id, from_role=from_role, to_role=to_role, diagnosis=diagnosis
    )

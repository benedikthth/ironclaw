"""End-to-end PhD slice demo, driven by the deterministic FakeProvider.

Runs with no API key and no network: it exercises the real harness (loop, tools,
skills, verification) against a scripted model so the mechanics are observable
and CI-testable. Swap FakeProvider for a real adapter to run it for real.

    python -m ironclaw.demo
"""

from __future__ import annotations

import os
import tempfile

from .agents.phd import run_phd
from .contracts import AcceptanceCriterion, CheckKind, Task
from .providers.base import AssistantTurn, ToolCall
from .providers.fake import FakeProvider

SALES_CSV = "region,amount\nnorth,10\nsouth,5\nnorth,7\nsouth,3\n"

# Verifies the PhD actually computed the right aggregation, not just made a file.
_TOTALS_CHECK = (
    "python3 -c \""
    "import csv;"
    "d={r[0]:r[1] for r in list(csv.reader(open('region_totals.csv')))[1:]};"
    "assert d['north']=='17.0' and d['south']=='8.0', d\""
)

TASK = Task(
    goal="Aggregate the sales data in sales.csv by region and report the totals.",
    acceptance=[
        AcceptanceCriterion("region_totals.csv exists", CheckKind.FILE_EXISTS, "region_totals.csv"),
        AcceptanceCriterion("totals are correct", CheckKind.COMMAND, _TOTALS_CHECK),
        AcceptanceCriterion("a markdown report exists", CheckKind.FILE_EXISTS, "report.md"),
    ],
    inputs={"source": "sales.csv"},
)

_TRANSFORM_CODE = """
import csv, collections
totals = collections.defaultdict(float)
with open('sales.csv', newline='') as fh:
    for row in csv.DictReader(fh):
        totals[row['region']] += float(row['amount'])
with open('region_totals.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['key', 'total'])
    for k in sorted(totals):
        w.writerow([k, totals[k]])
print('wrote', dict(totals))
"""

REPORT = "# Sales by region\n\n- north: 17.0\n- south: 8.0\n"


def build_script() -> list:
    return [
        AssistantTurn(
            text="I'll reuse the transform_csv skill rather than improvise.",
            tool_calls=[ToolCall("c1", "invoke_skill", {"name": "transform_csv"})],
        ),
        AssistantTurn(
            text="Applying the recipe to sales.csv.",
            tool_calls=[ToolCall("c2", "python_exec", {"code": _TRANSFORM_CODE})],
        ),
        AssistantTurn(
            text="Writing the report.",
            tool_calls=[ToolCall("c3", "write_file", {"path": "report.md", "content": REPORT})],
        ),
        AssistantTurn(
            text="Done; submitting.",
            tool_calls=[
                ToolCall(
                    "c4",
                    "submit_result",
                    {
                        "summary": "Aggregated sales.csv by region (north=17.0, south=8.0).",
                        "artifacts": [
                            {"path": "region_totals.csv", "kind": "data", "description": "totals"},
                            {"path": "report.md", "kind": "report", "description": "summary"},
                        ],
                    },
                )
            ],
        ),
    ]


def run(workspace: str) -> "object":
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, "sales.csv"), "w", encoding="utf-8") as fh:
        fh.write(SALES_CSV)
    skills_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
    provider = FakeProvider(build_script())
    return run_phd(
        task=TASK, provider=provider, workspace=workspace, skills_root=skills_root
    ).result


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = run(tmp)
        print(result.to_json())


if __name__ == "__main__":
    main()

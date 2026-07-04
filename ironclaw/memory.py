"""Institute memory — what labs share across sessions.

Every ironclaw lab used to start from zero. This is the fix (Hermes's first
pillar): a durable, searchable record of past labs and their findings, so a new
lab starts informed by prior work. Skills are already institute-global via the
Infrastructure Manager; this adds the *knowledge* layer — statements attempted,
what passed, where the artifacts landed — and feeds it back into PI decomposition.

Stdlib-only: JSON-backed persistence + keyword-overlap retrieval. Swap the search
for SQLite FTS5 or embeddings later without touching callers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

from .observability import active_recorder

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class MemoryRecord:
    kind: str  # "lab" | "finding"
    text: str  # searchable content
    tags: list[str] = field(default_factory=list)
    ref: dict = field(default_factory=dict)  # structured payload (ids, artifacts, status)
    seq: int = 0


class MemoryStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.records: list[MemoryRecord] = []
        self._seq = 0
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for d in json.load(fh):
                    self.records.append(MemoryRecord(**d))
                    self._seq = max(self._seq, d.get("seq", 0))

    def add(self, kind: str, text: str, *, tags: list[str] | None = None, ref: dict | None = None) -> MemoryRecord:
        self._seq += 1
        rec = MemoryRecord(kind, text, list(tags or []), dict(ref or {}), self._seq)
        self.records.append(rec)
        self._persist()
        active_recorder().emit("memory.record", role="memory", message=kind, data={"seq": rec.seq})
        return rec

    def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Top-k prior records by keyword overlap (recent breaks ties)."""
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for r in self.records:
            overlap = len(q & _tokens(r.text + " " + " ".join(r.tags)))
            if overlap:
                scored.append((overlap, r.seq, r))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        hits = [r for _, _, r in scored[:k]]
        if hits:
            active_recorder().emit("memory.recall", role="memory", message=query[:60],
                                   data={"hits": len(hits)})
        return hits

    def context_for(self, query: str, k: int = 3) -> str:
        """A compact 'prior related work' block to prepend to a PI decomposition
        prompt. Empty string when there's nothing relevant."""
        hits = self.search(query, k)
        if not hits:
            return ""
        lines = ["## Prior related work in this institute (reuse where you can)"]
        for h in hits:
            status = f" [{', '.join(h.tags)}]" if h.tags else ""
            lines.append(f"- ({h.kind}){status} {h.text.strip()[:240]}")
        return "\n".join(lines)

    def _persist(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in self.records], fh, indent=2)
        os.replace(tmp, self.path)  # atomic

"""Contract blacklist scanner for manuscript text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class QAFinding:
    path: str
    line: int
    term: str
    category: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "term": self.term,
            "category": self.category,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sample_size_claim", re.compile(r"\b(?:n\s*=\s*)?(?:600|300)\s+(?:(?:simulated|synthetic|AI)\s+)?(?:learners?|students?)\b", re.I)),
    ("real_student_distribution", re.compile(r"\b(?:real|actual)\s+(?:student|learner)s?\s+distribution\b|\bdistribution\s+of\s+(?:real|actual)\s+(?:student|learner)s?\b|\brepresentative\s+of\s+(?:the\s+)?(?:real|actual)\s+(?:student|learner)\s+population\b|\b(?:reflects?|mirrors?)\s+(?:the\s+)?(?:real|actual)\s+(?:student|learner)\s+distribution\b", re.I)),
    ("human_or_teacher_validation", re.compile(r"\b(?:human|teacher|expert)[ -](?:gold|validated|validation|validation[- ]set|ground\s+truth)\b|\bvalidated\s+by\s+(?:human|teacher|expert)s?\b|\b(?:human|teacher|expert)\s+ground\s+truth\b", re.I)),
    ("learning_trajectory_simulation", re.compile(r"\blearning[ -]trajector(?:y|ies)(?:\s+simulation|\s+simulated|\s+study)?\b|\b(?:simulated?\s+)?learning\s+trajectories\b|\btrajectory\s+simulation\b", re.I)),
    ("four_state_persona", re.compile(r"\bfour[ -]state(?:s)?\s+persona(?:s)?\b|\bfour\s+states?\s+personas?\b", re.I)),
    ("novelty_claim", re.compile(r"\bfirst[ -]ever\b|\bfirst[ -]of[ -](?:(?:the|its|a)[ -])?kind\b", re.I)),
    ("ceai_index_claim", re.compile(r"(?:C&E:AI|Computers\s*&\s*Education:\s*Artificial\s+Intelligence)[^.]{0,120}\b(?:SCIE|SSCI|Q1|impact\s+factor|IF\s*23\.4|23\.4)\b|\b(?:SCIE|SSCI|Q1|impact\s+factor|IF\s*23\.4|23\.4)\b[^.]{0,120}(?:C&E:AI|Computers\s*&\s*Education:\s*Artificial\s+Intelligence)", re.I)),
)


def scan_manuscript_text(text: str, *, path: str = "<memory>") -> list[QAFinding]:
    if not isinstance(text, str):
        raise TypeError("manuscript text must be a string")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    scan_text = normalized.replace("\n", " ")
    findings: list[QAFinding] = []
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(scan_text):
            line_number = normalized.count("\n", 0, match.start()) + 1
            findings.append(
                QAFinding(
                    path=str(path),
                    line=line_number,
                    term=" ".join(match.group(0).split()),
                    category=category,
                )
            )
    findings.sort(key=lambda finding: (finding.line, finding.category, finding.term.lower()))
    return findings


def scan_manuscript(source: str | Path) -> list[QAFinding]:
    candidate = Path(source) if isinstance(source, (str, Path)) else None
    if candidate is not None and candidate.is_file():
        return scan_manuscript_text(candidate.read_text(encoding="utf-8"), path=str(candidate))
    return scan_manuscript_text(str(source), path="<memory>")


def scan_paths(paths: Iterable[str | Path]) -> list[QAFinding]:
    findings: list[QAFinding] = []
    for path in paths:
        findings.extend(scan_manuscript(path))
    return findings


__all__ = ["QAFinding", "scan_manuscript", "scan_manuscript_text", "scan_paths"]

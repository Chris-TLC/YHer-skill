"""Compatibility exports for manuscript contract QA."""

from .manuscript_qa import QAFinding, scan_manuscript, scan_manuscript_text, scan_paths


def scan_blacklist(source):
    return scan_manuscript(source)


__all__ = [
    "QAFinding",
    "scan_blacklist",
    "scan_manuscript",
    "scan_manuscript_text",
    "scan_paths",
]

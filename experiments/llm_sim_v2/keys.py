"""Canonical normalization for security-sensitive structured keys."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def canonical_key(value: Any) -> str:
    """Normalize snake/kebab/space/camel-case and Unicode separators alike."""

    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


__all__ = ["canonical_key"]

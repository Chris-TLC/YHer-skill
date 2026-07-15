"""Offline contracts for the Persona v2 dual-condition study.

The package intentionally contains no provider transport and no dependency on
the v1 experiment artifacts.
"""

from .grid import build_persona_grid, build_persona_rows, serialize_grid
from .mapping import normalize_target_option_map, validate_target_option_map
from .models import PersonaV2
from .panel import build_review_payload, select_calibration_items
from .manuscript_qa import scan_manuscript, scan_manuscript_text
from .store import V2Store

__all__ = [
    "PersonaV2",
    "V2Store",
    "build_persona_grid",
    "build_persona_rows",
    "serialize_grid",
    "select_calibration_items",
    "build_review_payload",
    "normalize_target_option_map",
    "validate_target_option_map",
    "scan_manuscript",
    "scan_manuscript_text",
]

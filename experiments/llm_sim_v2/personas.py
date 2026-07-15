"""Compatibility vocabulary for Persona v2 grid builders."""

from .grid import (
    ANCHOR_COUNT,
    NOISE_LEVELS,
    build_persona_grid,
    build_persona_rows,
    grid_sha256,
    serialize_grid,
)


def build_personas(*args, **kwargs):
    return build_persona_grid(*args, **kwargs)


__all__ = [
    "ANCHOR_COUNT",
    "NOISE_LEVELS",
    "build_personas",
    "build_persona_grid",
    "build_persona_rows",
    "serialize_grid",
    "grid_sha256",
]

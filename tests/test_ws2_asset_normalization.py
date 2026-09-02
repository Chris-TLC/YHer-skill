#!/usr/bin/env python3
"""Tests for WS2 asset normalization helpers."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def make_rgb_png(width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> bytes:
    rows = []
    for row in pixels:
        rows.append(b"\x00" + b"".join(bytes(rgb) for rgb in row))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(b"".join(rows))) + png_chunk(b"IEND", b"")


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def formula_block(media: str) -> dict:
    return {"type": "formula", "media": media}


def figure_block(media: str) -> dict:
    return {"type": "figure", "media": media}


def test_collect_stem_media_refs_resolves_assets_from_group_summaries(tmp_path: Path):
    from scripts.build_ws2_asset_manifest import collect_stem_media_refs

    batch = tmp_path / "batch"
    paper = batch / "paper_a"
    assets = paper / "assets"
    assets.mkdir(parents=True)
    (assets / "image1.wmf").write_bytes(b"wmf-data")
    (paper / "summary.json").write_text(json.dumps({"group_key": "测试卷A"}, ensure_ascii=False))
    (batch / "questions_deduped.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "group_key": "测试卷A",
                "stem_blocks": [{"para": [text_block("题干"), formula_block("image1.wmf")]}],
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    refs = collect_stem_media_refs(batch / "questions_deduped.jsonl", batch)

    assert len(refs) == 1
    assert refs[0].question_id == "q1"
    assert refs[0].asset_path == assets / "image1.wmf"
    assert refs[0].block_type == "formula"


def test_classification_uses_block_type_and_records_mixed_edge_case():
    from scripts.build_ws2_asset_manifest import classify_asset

    formula = classify_asset({"formula"}, (80, 20), ".wmf")
    mixed = classify_asset({"formula", "figure"}, (320, 240), ".png")

    assert formula["asset_class"] == "formula_image"
    assert "small_formula_like" in formula["edge_case_flags"]
    assert mixed["asset_class"] == "illustration"
    assert "mixed_formula_figure_refs" in mixed["edge_case_flags"]


def test_crop_png_whitespace_returns_content_bbox(tmp_path: Path):
    from scripts.build_ws2_asset_manifest import crop_png_whitespace, png_dimensions

    pixels = [
        [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
        [(255, 255, 255), (10, 10, 10), (20, 20, 20), (255, 255, 255)],
        [(255, 255, 255), (30, 30, 30), (40, 40, 40), (255, 255, 255)],
        [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
    ]
    src = tmp_path / "src.png"
    out = tmp_path / "cropped.png"
    src.write_bytes(make_rgb_png(4, 4, pixels))

    result = crop_png_whitespace(src, out, padding=0)

    assert result["bbox"] == [1, 1, 2, 2]
    assert png_dimensions(out) == (2, 2)

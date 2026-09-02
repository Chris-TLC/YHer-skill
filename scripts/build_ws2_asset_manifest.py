#!/usr/bin/env python3
"""Normalize WS2 stem assets for review.

The script resolves media referenced by v4 question stems, hash de-duplicates
them, converts WMF/EMF assets to SVG and cropped PNG via LibreOffice, and writes
only /tmp review artifacts by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import tempfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOFFICE = "/opt/homebrew/bin/soffice"
VECTOR_EXTS = {".wmf", ".emf"}


@dataclass(frozen=True)
class MediaRef:
    question_id: str
    group_key: str
    media: str
    block_type: str
    asset_path: Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_group_asset_dirs(batch_root: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for summary_path in batch_root.glob("*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        group_key = summary.get("group_key")
        if group_key:
            dirs[str(group_key)] = summary_path.parent / "assets"
    return dirs


def iter_media_blocks(node: object) -> Iterable[dict]:
    if isinstance(node, dict):
        if node.get("media") and node.get("type") in {"formula", "figure"}:
            yield node
        for value in node.values():
            yield from iter_media_blocks(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_media_blocks(value)


def collect_stem_media_refs(questions_path: Path, batch_root: Path) -> list[MediaRef]:
    group_dirs = load_group_asset_dirs(batch_root)
    refs: list[MediaRef] = []
    for question in load_jsonl(questions_path):
        group_key = str(question.get("group_key") or "")
        asset_dir = group_dirs.get(group_key)
        if not asset_dir:
            continue
        for block in iter_media_blocks(question.get("stem_blocks", [])):
            media = str(block.get("media") or "")
            refs.append(
                MediaRef(
                    question_id=str(question.get("question_id") or ""),
                    group_key=group_key,
                    media=media,
                    block_type=str(block.get("type") or "unknown"),
                    asset_path=asset_dir / media,
                )
            )
    return refs


def sha256_file(path: Path) -> str:
    h = hashlib_sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hashlib_sha256():
    import hashlib

    return hashlib.sha256()


def png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a png")
    chunks: list[tuple[bytes, bytes]] = []
    pos = 8
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        chunks.append((kind, payload))
        pos += 12 + length
        if kind == b"IEND":
            break
    return chunks


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    ihdr = next(payload for kind, payload in png_chunks(data) if kind == b"IHDR")
    return struct.unpack(">II", ihdr[:8])


def jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        return None
    pos = 2
    while pos + 9 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        pos += 2
        if marker in {0xD8, 0xD9}:
            continue
        length = struct.unpack(">H", data[pos : pos + 2])[0]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height, width = struct.unpack(">HH", data[pos + 3 : pos + 7])
            return width, height
        pos += length
    return None


def image_dimensions(path: Path) -> tuple[int, int] | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".png":
            return png_dimensions(path)
        if suffix in {".jpg", ".jpeg"}:
            return jpeg_dimensions(path)
    except Exception:
        return None
    return None


def paeth(left: int, up: int, up_left: int) -> int:
    p = left + up - up_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def png_decode_rows(data: bytes) -> tuple[int, int, int, int, list[bytearray]]:
    chunks = png_chunks(data)
    ihdr = next(payload for kind, payload in chunks if kind == b"IHDR")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise ValueError("unsupported png encoding")
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError("unsupported png color type")
    bpp = channels_by_type[color_type]
    raw = zlib.decompress(b"".join(payload for kind, payload in chunks if kind == b"IDAT"))
    stride = width * bpp
    rows: list[bytearray] = []
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        row = bytearray(raw[pos : pos + stride])
        pos += stride
        for i in range(stride):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filter_type == 1:
                row[i] = (row[i] + left) & 0xFF
            elif filter_type == 2:
                row[i] = (row[i] + up) & 0xFF
            elif filter_type == 3:
                row[i] = (row[i] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[i] = (row[i] + paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported png filter {filter_type}")
        rows.append(row)
        prev = row
    return width, height, color_type, bpp, rows


def png_encode_rows(width: int, height: int, color_type: int, rows: list[bytearray]) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def pixel_is_content(pixel: bytes, color_type: int) -> bool:
    if color_type == 0:
        return pixel[0] < 248
    if color_type == 2:
        return any(channel < 248 for channel in pixel[:3])
    if color_type == 4:
        return pixel[1] > 8 and pixel[0] < 248
    if color_type == 6:
        return pixel[3] > 8 and any(channel < 248 for channel in pixel[:3])
    return False


def crop_png_whitespace(src: Path, out: Path, padding: int = 4) -> dict:
    width, height, color_type, bpp, rows = png_decode_rows(src.read_bytes())
    min_x, min_y = width, height
    max_x, max_y = -1, -1
    for y, row in enumerate(rows):
        for x in range(width):
            pixel = bytes(row[x * bpp : (x + 1) * bpp])
            if pixel_is_content(pixel, color_type):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < 0:
        shutil.copy2(src, out)
        return {"bbox": None, "original_size": [width, height], "cropped_size": [width, height], "cropped": False}
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(width - 1, max_x + padding)
    max_y = min(height - 1, max_y + padding)
    cropped_rows = [row[min_x * bpp : (max_x + 1) * bpp] for row in rows[min_y : max_y + 1]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_encode_rows(max_x - min_x + 1, max_y - min_y + 1, color_type, cropped_rows))
    return {
        "bbox": [min_x, min_y, max_x, max_y],
        "original_size": [width, height],
        "cropped_size": [max_x - min_x + 1, max_y - min_y + 1],
        "cropped": [min_x, min_y, max_x, max_y] != [0, 0, width - 1, height - 1],
    }


def classify_asset(block_types: set[str], dimensions: tuple[int, int] | None, ext: str) -> dict:
    flags: list[str] = []
    if {"formula", "figure"}.issubset(block_types):
        flags.append("mixed_formula_figure_refs")
    if dimensions is None:
        flags.append("missing_dimensions")
    else:
        width, height = dimensions
        if width <= 160 or height <= 80:
            flags.append("small_formula_like")
        if "formula" in block_types and (width > 900 or height > 500):
            flags.append("large_formula_like")
        if "figure" in block_types and (width <= 160 or height <= 80):
            flags.append("small_figure_like")
    if "figure" in block_types:
        asset_class = "illustration"
    elif "formula" in block_types:
        asset_class = "formula_image"
    else:
        asset_class = "unknown"
    return {"asset_class": asset_class, "edge_case_flags": flags, "original_ext": ext.lower()}


def run_soffice_batch(paths: list[Path], out_dir: Path, target: str, chunk_size: int = 80) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start : start + chunk_size]
        subprocess.run(
            [SOFFICE, "--headless", "--convert-to", target, "--outdir", str(out_dir), *map(str, chunk)],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )


def build_asset_manifest(
    questions_path: Path,
    batch_root: Path,
    out_dir: Path,
    convert_vectors: bool = True,
) -> dict:
    refs = [ref for ref in collect_stem_media_refs(questions_path, batch_root) if ref.asset_path.exists()]
    by_hash: dict[str, list[MediaRef]] = defaultdict(list)
    for ref in refs:
        by_hash[sha256_file(ref.asset_path)].append(ref)

    out_dir.mkdir(parents=True, exist_ok=True)
    originals_dir = out_dir / "originals"
    normalized_dir = out_dir / "normalized"
    raw_png_dir = out_dir / "converted_raw_png"
    raw_svg_dir = out_dir / "converted_svg"
    originals_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    vector_inputs: list[Path] = []
    hash_to_temp_stem: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="yher_ws2_vectors_") as td:
        temp_input = Path(td)
        rows_prepared: list[tuple[str, list[MediaRef], Path, str]] = []
        for asset_hash, asset_refs in by_hash.items():
            representative = asset_refs[0].asset_path
            ext = representative.suffix.lower()
            original_copy = originals_dir / f"{asset_hash}{ext}"
            if not original_copy.exists():
                shutil.copy2(representative, original_copy)
            rows_prepared.append((asset_hash, asset_refs, original_copy, ext))
            if convert_vectors and ext in VECTOR_EXTS:
                temp_name = f"{asset_hash}{ext}"
                temp_path = temp_input / temp_name
                shutil.copy2(representative, temp_path)
                vector_inputs.append(temp_path)
                hash_to_temp_stem[asset_hash] = temp_path.stem

        if vector_inputs:
            run_soffice_batch(vector_inputs, raw_png_dir, "png")
            run_soffice_batch(vector_inputs, raw_svg_dir, "svg")

        manifest_rows: list[dict] = []
        conversion_failures = 0
        for asset_hash, asset_refs, original_copy, ext in rows_prepared:
            block_types = {ref.block_type for ref in asset_refs}
            normalized_png: Path | None = None
            normalized_svg: Path | None = None
            crop_result: dict | None = None
            dimensions: tuple[int, int] | None = None
            conversion_status = "not_required"

            if ext in VECTOR_EXTS:
                conversion_status = "converted"
                stem = hash_to_temp_stem.get(asset_hash, asset_hash)
                raw_png = raw_png_dir / f"{stem}.png"
                raw_svg = raw_svg_dir / f"{stem}.svg"
                if raw_png.exists():
                    normalized_png = normalized_dir / f"{asset_hash}.png"
                    try:
                        crop_result = crop_png_whitespace(raw_png, normalized_png)
                        dimensions = tuple(crop_result["cropped_size"])  # type: ignore[arg-type]
                    except Exception:
                        shutil.copy2(raw_png, normalized_png)
                        dimensions = image_dimensions(normalized_png)
                        crop_result = {"bbox": None, "cropped": False, "error": "png_crop_failed"}
                else:
                    conversion_status = "conversion_failed"
                    conversion_failures += 1
                if raw_svg.exists():
                    normalized_svg = normalized_dir / f"{asset_hash}.svg"
                    shutil.copy2(raw_svg, normalized_svg)
            else:
                normalized_path = normalized_dir / f"{asset_hash}{ext}"
                if not normalized_path.exists():
                    shutil.copy2(original_copy, normalized_path)
                if ext == ".png":
                    normalized_png = normalized_path
                dimensions = image_dimensions(normalized_path)

            classification = classify_asset(block_types, dimensions, ext)
            if conversion_status == "conversion_failed":
                classification["edge_case_flags"].append("conversion_failed")
            row = {
                "asset_hash": asset_hash,
                "original_ext": ext,
                "ref_count": len(asset_refs),
                "question_count": len({ref.question_id for ref in asset_refs}),
                "block_types": sorted(block_types),
                "asset_class": classification["asset_class"],
                "edge_case_flags": classification["edge_case_flags"],
                "dimensions": list(dimensions) if dimensions else None,
                "conversion_status": conversion_status,
                "normalized_png": str(normalized_png) if normalized_png else None,
                "normalized_svg": str(normalized_svg) if normalized_svg else None,
                "crop": crop_result,
                "sample_refs": [
                    {
                        "question_id": ref.question_id,
                        "group_key": ref.group_key,
                        "media": ref.media,
                        "asset_path": str(ref.asset_path),
                        "block_type": ref.block_type,
                    }
                    for ref in asset_refs[:5]
                ],
            }
            manifest_rows.append(row)

    manifest_rows.sort(key=lambda row: (row["asset_class"], row["asset_hash"]))
    write_jsonl(out_dir / "asset_manifest.jsonl", manifest_rows)

    summary = {
        "total_stem_refs": len(refs),
        "unique_asset_hashes": len(manifest_rows),
        "extension_counts_by_ref": dict(Counter(ref.asset_path.suffix.lower() for ref in refs)),
        "extension_counts_by_hash": dict(Counter(row["original_ext"] for row in manifest_rows)),
        "class_counts": dict(Counter(row["asset_class"] for row in manifest_rows)),
        "edge_case_counts": dict(Counter(flag for row in manifest_rows for flag in row["edge_case_flags"])),
        "conversion_failures": conversion_failures,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir / "ASSET_NORMALIZATION_REPORT.md", summary, manifest_rows)
    return summary


def write_report(path: Path, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# WS2 Asset Normalization Report",
        "",
        "## Summary",
        "",
        f"- Total stem media refs: {summary['total_stem_refs']}",
        f"- Unique asset hashes: {summary['unique_asset_hashes']}",
        f"- Conversion failures: {summary['conversion_failures']}",
        "",
        "## Extension Counts By Ref",
        "",
    ]
    for key, count in sorted(summary["extension_counts_by_ref"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Class Counts", ""])
    for key, count in sorted(summary["class_counts"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Edge Case Counts", ""])
    for key, count in sorted(summary["edge_case_counts"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Edge Case Samples", ""])
    for row in [row for row in rows if row["edge_case_flags"]][:30]:
        sample = row["sample_refs"][0] if row["sample_refs"] else {}
        lines.append(
            f"- `{row['asset_hash'][:12]}` {row['original_ext']} {row['asset_class']} "
            f"flags={','.join(row['edge_case_flags'])} sample_q={sample.get('question_id')} media={sample.get('media')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/ws1_batch_v4_20260703/questions_deduped.jsonl"))
    parser.add_argument("--batch-root", type=Path, default=Path("data/ws1_batch_v4_20260703"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/yher_ws2_assets_v1"))
    parser.add_argument("--no-convert", action="store_true")
    args = parser.parse_args()
    summary = build_asset_manifest(args.questions, args.batch_root, args.out_dir, convert_vectors=not args.no_convert)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

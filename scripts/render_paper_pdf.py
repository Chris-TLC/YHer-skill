#!/usr/bin/env python3
"""Render a bound paper manuscript to a verified A4 PDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


REFERENCE_SCHEMA = "yher.verified-references.v1"
RENDER_RECEIPT_SCHEMA = "yher.paper_pdf.render_receipt.v2"
SOURCE_EQUIVALENCE_SCHEMA = "yher.paper_pdf.source_equivalence.v1"
RASTER_DPI = 150
LOCAL_TOOL_TIMEOUT_SECONDS = 60.0


class RenderError(RuntimeError):
    """Raised when paper rendering cannot satisfy a release gate."""


@dataclass(frozen=True)
class RenderResult:
    """Verified output metadata returned after atomic PDF promotion."""

    output_path: Path
    pages: int
    receipt_path: Path | None = None


@dataclass(frozen=True)
class PdfInfo:
    """Poppler metadata needed by the PDF release gates."""

    pages: int
    width_points: float
    height_points: float


@dataclass(frozen=True)
class PageGeometry:
    """Per-page geometry used to compare independent PDF renders."""

    page: int
    width_points: float
    height_points: float
    rotation_degrees: int


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    data = resolved.read_bytes()
    return {
        "source_path": str(resolved),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _binding_from_bytes(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "source_path": str(path.expanduser().resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _portable_binding_from_bytes(
    path: Path,
    *,
    anchor: Path,
    data: bytes,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    anchor_resolved = anchor.expanduser().resolve()
    return {
        "relative_path": Path(os.path.relpath(resolved, anchor_resolved)).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _assert_distinct_paths(**paths: Path) -> dict[str, Path]:
    resolved = {
        name: Path(path).expanduser().resolve()
        for name, path in paths.items()
    }
    names = list(resolved)
    for index, left_name in enumerate(names):
        left = resolved[left_name]
        for right_name in names[index + 1 :]:
            right = resolved[right_name]
            aliased = left == right
            if not aliased and left.exists() and right.exists():
                try:
                    aliased = left.samefile(right)
                except OSError:
                    aliased = False
            if aliased:
                raise RenderError(
                    f"publication paths must be distinct; {left_name} aliases {right_name}"
                )
    return resolved


def _assert_outside_tree(path: Path, root: Path, *, label: str) -> None:
    resolved_path = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return
    raise RenderError(f"{label} must remain outside the manuscript source tree")


def _stable_read_file(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RenderError(f"{label} is not a regular file")
            data = handle.read()
            after = os.fstat(handle.fileno())
        path_after = resolved.lstat()
    except OSError as exc:
        raise RenderError(f"could not snapshot {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not stat.S_ISREG(path_after.st_mode):
        raise RenderError(f"{label} path is not a regular file")
    identity_before = (
        before.st_mode,
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_mode,
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity = (
        path_after.st_mode,
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if identity_before != identity_after or identity_after != path_identity:
        raise RenderError(f"{label} changed while it was being snapshotted")
    if len(data) != after.st_size:
        raise RenderError(f"{label} snapshot length differs from file size")
    state = {
        "mode": after.st_mode,
        "device": after.st_dev,
        "inode": after.st_ino,
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return data, state


def _assert_file_unchanged(
    path: Path,
    expected_state: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _, current_state = _stable_read_file(path, label=label)
    comparable_expected = {
        key: value for key, value in expected_state.items() if key != "ctime_ns"
    }
    comparable_current = {
        key: value for key, value in current_state.items() if key != "ctime_ns"
    }
    if comparable_current != comparable_expected:
        raise RenderError(f"{label} changed during source-bound verification")


def _regular_tree_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RenderError(f"source snapshot contains a symbolic link: {path}")
        if path.is_file():
            paths.append(path)
        elif not path.is_dir():
            raise RenderError(f"source snapshot contains a special entry: {path}")
    return sorted(paths, key=lambda value: value.relative_to(root).as_posix())


def _snapshot_source_tree(
    source_root: Path,
    snapshot_root: Path,
) -> dict[str, dict[str, Any]]:
    source_root = source_root.expanduser().resolve(strict=True)
    snapshot_root.mkdir(parents=True, exist_ok=False)
    states: dict[str, dict[str, Any]] = {}
    initial_files = _regular_tree_files(source_root)
    for source in initial_files:
        relative = source.relative_to(source_root)
        data, state = _stable_read_file(source, label=f"source file {relative.as_posix()}")
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        states[relative.as_posix()] = state
    if [
        path.relative_to(source_root).as_posix()
        for path in _regular_tree_files(source_root)
    ] != list(states):
        raise RenderError("source file roster changed while it was being snapshotted")
    return states


def _assert_source_tree_unchanged(
    source_root: Path,
    expected_states: Mapping[str, Mapping[str, Any]],
) -> None:
    source_root = source_root.expanduser().resolve(strict=True)
    current_files = [
        path.relative_to(source_root).as_posix()
        for path in _regular_tree_files(source_root)
    ]
    if current_files != list(expected_states):
        raise RenderError("source file roster changed during source-bound verification")
    for relative, expected_state in expected_states.items():
        _assert_file_unchanged(
            source_root / relative,
            expected_state,
            label=f"source file {relative}",
        )


def _portable_tree_manifest(
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        {
            "relative_path": relative,
            "bytes": state["bytes"],
            "sha256": state["sha256"],
        }
        for relative, state in states.items()
    ]
    return {
        "root": "manuscript-resource-root",
        "files": rows,
        "file_set_sha256": hashlib.sha256(_canonical_json(rows)).hexdigest(),
    }


def _portable_tree_manifest_from_root(root: Path) -> dict[str, Any]:
    states: dict[str, Mapping[str, Any]] = {}
    for path in _regular_tree_files(root):
        relative = path.relative_to(root).as_posix()
        _, state = _stable_read_file(path, label=f"source file {relative}")
        states[relative] = state
    return _portable_tree_manifest(states)


def _portable_file_binding(path: Path, *, anchor: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    anchor_resolved = anchor.expanduser().resolve()
    data = resolved.read_bytes()
    return {
        "relative_path": Path(os.path.relpath(resolved, anchor_resolved)).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_render_receipt(
    *,
    profile: str,
    input_path: Path,
    references_path: Path,
    output_path: Path,
    pages: int,
    pandoc: str,
    chrome: str,
    receipt_path: Path,
    pdfinfo: str = "pdfinfo",
    pdftotext: str = "pdftotext",
    pdftoppm: str = "pdftoppm",
) -> dict[str, Any]:
    if profile not in {"main", "yau"} or pages < 1:
        raise RenderError("render receipt profile or page count is invalid")
    publication_paths = _assert_distinct_paths(
        manuscript=input_path,
        references=references_path,
        pdf=output_path,
        receipt=receipt_path,
    )
    destination = publication_paths["receipt"]
    input_locator = Path(input_path).expanduser()
    reference_locator = Path(references_path).expanduser()
    pdf_locator = Path(output_path).expanduser()
    original_input = input_locator.resolve(strict=True)
    original_references = reference_locator.resolve(strict=True)
    original_pdf = pdf_locator.resolve(strict=True)
    source_root = original_input.parent
    _assert_outside_tree(destination, source_root, label="render receipt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path = Path(__file__).resolve(strict=True)
    tools_before = _render_tool_descriptors(
        pandoc=pandoc,
        chrome=chrome,
        pdfinfo=pdfinfo,
        pdftotext=pdftotext,
        pdftoppm=pdftoppm,
    )
    wrapper_bytes, wrapper_state = _stable_read_file(
        wrapper_path, label="render wrapper"
    )

    with tempfile.TemporaryDirectory(prefix=".paper-receipt-proof-") as temporary_directory:
        snapshot_root = Path(temporary_directory)
        source_snapshot_root = snapshot_root / "source"
        source_states = _snapshot_source_tree(source_root, source_snapshot_root)
        snapshot_input = source_snapshot_root / original_input.relative_to(source_root)
        reference_bytes, reference_state = _stable_read_file(
            original_references, label="reference registry"
        )
        snapshot_references = snapshot_root / "references.json"
        snapshot_references.write_bytes(reference_bytes)
        pdf_bytes, pdf_state = _stable_read_file(original_pdf, label="rendered PDF")
        snapshot_pdf = snapshot_root / "candidate.pdf"
        snapshot_pdf.write_bytes(pdf_bytes)
        tool_paths = {
            name: str(descriptor["resolved_path"])
            for name, descriptor in tools_before.items()
        }

        verified_pages = _page_count(snapshot_pdf, pdfinfo=tool_paths["pdfinfo"])
        rendered_text = _pdf_text(
            snapshot_pdf, pdftotext=tool_paths["pdftotext"]
        )
        validate_rendered_text(
            rendered_text,
            pages=verified_pages,
            expected_pages=4 if profile == "yau" else None,
        )
        if profile == "main" and not 8 <= verified_pages <= 12:
            raise RenderError(
                f"rendered page count is {verified_pages}; main profile requires 8-12 pages"
            )
        if pages != verified_pages:
            raise RenderError("render receipt page count differs from verified PDF")
        source_equivalence = verify_source_bound_pdf(
            profile=profile,
            input_path=snapshot_input,
            references_path=snapshot_references,
            output_path=snapshot_pdf,
            pages=pages,
            pandoc=tool_paths["pandoc"],
            chrome=tool_paths["chrome"],
            pdfinfo=tool_paths["pdfinfo"],
            pdftotext=tool_paths["pdftotext"],
            pdftoppm=tool_paths["pdftoppm"],
        )
        input_bytes = snapshot_input.read_bytes()
        try:
            source_text = input_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RenderError("snapshotted manuscript is not UTF-8") from exc

        _assert_source_tree_unchanged(source_root, source_states)
        _assert_file_unchanged(
            original_references,
            reference_state,
            label="reference registry",
        )
        _assert_file_unchanged(original_pdf, pdf_state, label="rendered PDF")
        _assert_file_unchanged(wrapper_path, wrapper_state, label="render wrapper")
        _assert_render_tools_unchanged(tools_before)
        if (
            input_locator.resolve(strict=True) != original_input
            or reference_locator.resolve(strict=True) != original_references
            or pdf_locator.resolve(strict=True) != original_pdf
        ):
            raise RenderError(
                "source locator changed during source-bound verification"
            )

    payload: dict[str, Any] = {
        "schema_version": RENDER_RECEIPT_SCHEMA,
        "profile": profile,
        "input": _portable_binding_from_bytes(
            original_input,
            anchor=destination.parent,
            data=input_bytes,
        ),
        "references": _portable_binding_from_bytes(
            original_references,
            anchor=destination.parent,
            data=reference_bytes,
        ),
        "prepared_markdown_sha256": hashlib.sha256(
            prepare_markdown(source_text, profile=profile).encode("utf-8")
        ).hexdigest(),
        "css_sha256": hashlib.sha256(css_for_profile(profile).encode("utf-8")).hexdigest(),
        "renderer": {
            "wrapper": _portable_wrapper_binding(wrapper_path, wrapper_bytes),
            "tools": _portable_tool_descriptors(tools_before),
        },
        "source_resources": _portable_tree_manifest(source_states),
        "source_equivalence": source_equivalence,
        "pdf": {
            **_portable_binding_from_bytes(
                original_pdf,
                anchor=destination.parent,
                data=pdf_bytes,
            ),
            "pages": pages,
        },
    }
    payload["render_receipt_sha256"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return payload


def to_csl_references(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert the verified project reference format to deterministic CSL JSON."""
    if payload.get("schema_version") != REFERENCE_SCHEMA:
        raise RenderError(f"unsupported reference schema: {payload.get('schema_version')!r}")

    references = payload.get("references")
    if not isinstance(references, list):
        raise RenderError("reference payload must contain a references list")

    type_map = {
        "journal-article": "article-journal",
        "conference-paper": "paper-conference",
        "software-repository": "software",
        "software-documentation": "webpage",
    }
    field_map = {
        "container_title": "container-title",
        "volume": "volume",
        "issue": "issue",
        "pages": "page",
        "doi": "DOI",
        "url": "URL",
    }

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for reference in references:
        if not isinstance(reference, Mapping):
            raise RenderError("each reference must be an object")

        reference_id = reference.get("id")
        source_type = reference.get("type")
        title = reference.get("title")
        if not isinstance(reference_id, str) or not reference_id:
            raise RenderError("each reference must have a non-empty id")
        if reference_id in seen_ids:
            raise RenderError(f"duplicate reference id: {reference_id}")
        if source_type not in type_map:
            raise RenderError(f"unsupported reference type for {reference_id}: {source_type!r}")
        if not isinstance(title, str) or not title:
            raise RenderError(f"reference {reference_id} must have a non-empty title")
        seen_ids.add(reference_id)

        record: dict[str, Any] = {
            "id": reference_id,
            "type": type_map[source_type],
        }
        authors = reference.get("authors")
        if authors is not None:
            if not isinstance(authors, list) or not all(
                isinstance(author, str) and author for author in authors
            ):
                raise RenderError(f"reference {reference_id} has invalid authors")
            record["author"] = [{"literal": author} for author in authors]
        record["title"] = title

        year = reference.get("year")
        if year is not None:
            if not isinstance(year, int):
                raise RenderError(f"reference {reference_id} has an invalid year")
            record["issued"] = {"date-parts": [[year]]}

        for source_field, csl_field in field_map.items():
            value = reference.get(source_field)
            if value not in (None, ""):
                record[csl_field] = value
        records.append(record)

    return sorted(records, key=lambda record: record["id"])


_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*$")
_BIBLIOGRAPHY_ENTRY = re.compile(r"^\s*(?:\d+[.)]|[-+*])\s+\S", re.MULTILINE)
_YAU_MACHINE_AUDIT_BEGIN = re.compile(
    r"^[ \t]*<!-- BEGIN YAU MACHINE AUDIT -->[ \t]*$", re.MULTILINE
)
_YAU_MACHINE_AUDIT_END = re.compile(
    r"^[ \t]*<!-- END YAU MACHINE AUDIT -->[ \t]*$", re.MULTILINE
)
_YAU_MACHINE_AUDIT = re.compile(
    rf"{_YAU_MACHINE_AUDIT_BEGIN.pattern}.*?{_YAU_MACHINE_AUDIT_END.pattern}",
    re.MULTILINE | re.DOTALL,
)


def _markdown_headings(source: str) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    fence_character: str | None = None
    fence_length = 0
    offset = 0
    for line_with_ending in source.splitlines(keepends=True):
        line = line_with_ending.rstrip("\r\n")
        if fence_character is not None:
            closing_fence = re.compile(
                rf"^ {{0,3}}{re.escape(fence_character)}{{{fence_length},}}[ \t]*$"
            )
            if closing_fence.match(line):
                fence_character = None
                fence_length = 0
            offset += len(line_with_ending)
            continue

        fence = _FENCE_OPEN.match(line)
        if fence is not None:
            marker = fence.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            offset += len(line_with_ending)
            continue

        heading = _ATX_HEADING.match(line)
        if heading is not None:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group(2)).strip()
            headings.append((offset, offset + len(line_with_ending), title))
        offset += len(line_with_ending)
    return headings


def prepare_markdown(source: str, *, profile: str | None = None) -> str:
    """Prepare one publication profile while retaining machine audit in source."""
    if profile == "yau":
        audit_starts = list(_YAU_MACHINE_AUDIT_BEGIN.finditer(source))
        audit_ends = list(_YAU_MACHINE_AUDIT_END.finditer(source))
        if audit_starts or audit_ends:
            if (
                len(audit_starts) != 1
                or len(audit_ends) != 1
                or audit_starts[0].start() >= audit_ends[0].start()
            ):
                raise RenderError(
                    "Yau machine audit markers must appear exactly once in "
                    "BEGIN/END order"
                )
            source = _YAU_MACHINE_AUDIT.sub("", source, count=1)
    headings = _markdown_headings(source)
    references = [heading for heading in headings if heading[2].casefold() == "references"]
    if len(references) == 1:
        start, end, _ = references[0]
        later_headings = [heading for heading in headings if heading[0] > start]
        bibliography = source[end:]
        if not later_headings and _BIBLIOGRAPHY_ENTRY.search(bibliography):
            source = source[:start]
    return source.rstrip() + "\n"


def css_for_profile(profile: str) -> str:
    """Return print CSS for the full paper or compact four-page submission."""
    if profile not in {"main", "yau"}:
        raise RenderError(f"unknown render profile: {profile}")

    if profile == "yau":
        font_size = "9.2pt"
        line_height = "1.18"
        page_margin = "11mm 12mm 12mm"
        heading_gap = "0.42em"
        figure_height = "58mm"
    else:
        font_size = "10pt"
        line_height = "1.25"
        page_margin = "16mm 18mm 17mm"
        heading_gap = "0.62em"
        figure_height = "86mm"

    return f"""@page {{
  size: A4;
  margin: {page_margin};
}}

* {{
  box-sizing: border-box;
}}

html {{
  font-family: "Times New Roman", "Noto Serif CJK SC", "Songti SC", serif;
  color: #111;
}}

body {{
  margin: 0;
  font-size: {font_size};
  line-height: {line_height};
  text-align: justify;
  hyphens: auto;
}}

h1, h2, h3 {{
  break-after: avoid-page;
  page-break-after: avoid;
  line-height: 1.12;
  text-align: left;
}}

h1 {{
  font-size: 1.62em;
  margin: 0 0 {heading_gap};
  text-align: center;
}}

h2 {{
  font-size: 1.18em;
  margin: {heading_gap} 0 0.24em;
}}

h3 {{
  font-size: 1.02em;
  margin: 0.45em 0 0.18em;
}}

p, ul, ol, table, figure {{
  margin-top: 0.28em;
  margin-bottom: 0.42em;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88em;
  break-inside: avoid-page;
}}

th, td {{
  border: 0.4pt solid #777;
  padding: 0.18em 0.3em;
  vertical-align: top;
}}

img {{
  display: block;
  max-width: 100%;
  max-height: {figure_height};
  margin: 0 auto;
  object-fit: contain;
}}

.figure-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 3mm;
  break-inside: avoid-page;
}}

.figure-grid figure {{
  margin: 0;
}}

.persona-v2-composite {{
  margin-left: 0;
  margin-right: 0;
  width: 100%;
  break-inside: avoid-page;
}}

.persona-v2-summary {{
  display: block;
  width: 100%;
  height: auto;
  max-height: 60mm;
}}

.math.display {{
  display: block;
  margin: 0.3em 0;
  overflow: visible;
  text-align: center;
}}

#refs {{
  column-count: 2;
  column-gap: 5mm;
  font-size: 0.76em;
  line-height: 1.08;
}}

#refs > div {{
  break-inside: avoid;
  margin-bottom: 0.15em;
}}

a {{
  color: inherit;
  text-decoration: none;
}}
"""


_PANDOC_CITATION = re.compile(
    r"\[(?:-?@|[^\]\n]*?(?:\s|;)-?@)[A-Za-z0-9][A-Za-z0-9_.:-]*[^\]\n]*\]"
)
_PANDOC_TEXTUAL_CITATION = re.compile(
    r"(?<![A-Za-z0-9._%+-])@[A-Za-z0-9][A-Za-z0-9_.:-]*"
)
_TEX_CONTROL = re.compile(r"\\(?:[A-Za-z]+|[\[\]()])")
_WINDOWS_DRIVE_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\+[^\\\s]+)+"
)
_WINDOWS_UNC_PATH = re.compile(r"(?<!\\)\\{2,}[^\\\s]+(?:\\+[^\\\s]+)+")
_UNRESOLVED_PATTERNS = (
    ("PENDING placeholder", re.compile(r"\bPENDING(?:_[A-Z0-9_]+)?\b")),
    ("citation", _PANDOC_CITATION),
    ("citation", _PANDOC_TEXTUAL_CITATION),
)


def _has_unresolved_tex_control(text: str) -> bool:
    """Detect any TeX word control, except compact Windows path spans.

    Drive-letter and UNC paths are the deliberate exception because their ordinary
    backslash-separated components are indistinguishable from TeX word controls in
    extracted plain text. Paths containing whitespace are not exempted.
    """
    path_spans = [
        match.span()
        for pattern in (_WINDOWS_DRIVE_PATH, _WINDOWS_UNC_PATH)
        for match in pattern.finditer(text)
    ]
    for control in _TEX_CONTROL.finditer(text):
        if not any(start <= control.start() < end for start, end in path_spans):
            return True
    return False


def validate_rendered_text(
    text: str,
    *,
    pages: int,
    expected_pages: int | None = None,
) -> None:
    """Fail closed on unresolved source tokens and an exact pagination gate."""
    if not text.replace("\f", "").strip():
        raise RenderError("rendered PDF text is empty")

    for label, pattern in _UNRESOLVED_PATTERNS:
        if pattern.search(text):
            raise RenderError(f"unresolved {label} remains in rendered PDF")
    if _has_unresolved_tex_control(text):
        raise RenderError("unresolved TeX control sequence remains in rendered PDF")

    if pages < 1:
        raise RenderError(f"invalid rendered page count: {pages}")
    if expected_pages is not None and pages != expected_pages:
        raise RenderError(
            f"rendered page count is {pages}; expected exactly {expected_pages}"
        )


def _run_command(
    command: list[str],
    *,
    label: str,
    input_text: str | None = None,
    timeout_seconds: float = LOCAL_TOOL_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
    except OSError as exc:
        raise RenderError(f"{label} could not start: {exc}") from exc

    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _stop_process_group(process)
        raise RenderError(
            f"{label} timed out after {timeout_seconds:g} seconds"
        ) from exc
    except BaseException:
        _stop_process_group(process)
        raise

    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise RenderError(f"{label} failed with exit code {result.returncode}: {detail}")
    return result


def _run_binary_command(
    command: list[str],
    *,
    label: str,
    timeout_seconds: float = LOCAL_TOOL_TIMEOUT_SECONDS,
) -> bytes:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise RenderError(f"{label} could not start: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _stop_process_group(process)
        raise RenderError(
            f"{label} timed out after {timeout_seconds:g} seconds"
        ) from exc
    except BaseException:
        _stop_process_group(process)
        raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode != 0:
        detail_bytes = result.stderr or result.stdout or b"no diagnostic output"
        detail = detail_bytes.decode("utf-8", errors="replace").strip()
        raise RenderError(f"{label} failed with exit code {result.returncode}: {detail}")
    if isinstance(result.stdout, str):
        return result.stdout.encode("utf-8")
    return result.stdout


def _tool_descriptor(command: str, *, version_flag: str) -> dict[str, Any]:
    candidate = shutil.which(command)
    if candidate is None:
        candidate = str(Path(command).expanduser())
    try:
        resolved = Path(candidate).resolve(strict=True)
    except OSError as exc:
        raise RenderError(f"render tool is unavailable: {command}: {exc}") from exc
    _, before_state = _stable_read_file(
        resolved, label=f"render tool {resolved.name}"
    )
    result = _run_command([str(resolved), version_flag], label=f"{resolved.name} version")
    _, after_state = _stable_read_file(
        resolved, label=f"render tool {resolved.name}"
    )
    if before_state != after_state:
        raise RenderError(f"render tool changed while its version was inspected: {resolved}")
    version_lines = [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    if not version_lines:
        raise RenderError(f"render tool did not report a version: {resolved}")
    return {
        "command": command,
        "resolved_path": str(resolved),
        "version": version_lines[0],
        **after_state,
    }


def _render_tool_descriptors(
    *,
    pandoc: str,
    chrome: str,
    pdfinfo: str,
    pdftotext: str,
    pdftoppm: str,
) -> dict[str, dict[str, Any]]:
    commands = {
        "pandoc": (pandoc, "--version"),
        "chrome": (chrome, "--version"),
        "pdfinfo": (pdfinfo, "-v"),
        "pdftotext": (pdftotext, "-v"),
        "pdftoppm": (pdftoppm, "-v"),
    }
    return {
        name: _tool_descriptor(command, version_flag=version_flag)
        for name, (command, version_flag) in commands.items()
    }


def _assert_render_tools_unchanged(
    before: Mapping[str, Mapping[str, Any]],
) -> None:
    version_flags = {
        "pandoc": "--version",
        "chrome": "--version",
        "pdfinfo": "-v",
        "pdftotext": "-v",
        "pdftoppm": "-v",
    }
    after = {
        name: _tool_descriptor(
            str(descriptor.get("command") or ""),
            version_flag=version_flags[name],
        )
        for name, descriptor in before.items()
    }
    comparable_before = {
        name: {key: value for key, value in descriptor.items() if key != "ctime_ns"}
        for name, descriptor in before.items()
    }
    comparable_after = {
        name: {key: value for key, value in descriptor.items() if key != "ctime_ns"}
        for name, descriptor in after.items()
    }
    if comparable_after != comparable_before:
        changed = {
            name: sorted(
                field
                for field in set(comparable_before[name]) | set(comparable_after[name])
                if comparable_before[name].get(field)
                != comparable_after[name].get(field)
            )
            for name in before
            if comparable_before[name] != comparable_after[name]
        }
        raise RenderError(
            "render tool identity changed during source-bound verification: "
            f"{changed}"
        )


def _portable_tool_descriptors(
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "executable": Path(str(descriptor["resolved_path"])).name,
            "version": descriptor["version"],
            "bytes": descriptor["bytes"],
            "sha256": descriptor["sha256"],
        }
        for name, descriptor in descriptors.items()
    }


def _portable_wrapper_binding(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "filename": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


_A4_WIDTH_POINTS = 210 / 25.4 * 72
_A4_HEIGHT_POINTS = 297 / 25.4 * 72
# One PostScript point is about 0.35 mm and covers Chrome's observed rounding.
_A4_TOLERANCE_POINTS = 1.0


def parse_pdfinfo(output: str) -> PdfInfo:
    """Parse Poppler output and require portrait A4 within one point per axis."""
    pages_match = re.search(r"^Pages:\s*(\d+)\s*$", output, re.MULTILINE)
    if pages_match is None:
        raise RenderError("pdfinfo did not report a page count")
    size_match = re.search(
        r"^Page size:\s*([0-9]+(?:\.[0-9]+)?)\s+x\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+pts(?:\s+.*)?$",
        output,
        re.MULTILINE,
    )
    if size_match is None:
        raise RenderError("pdfinfo did not report a page size")

    width_points = float(size_match.group(1))
    height_points = float(size_match.group(2))
    if (
        abs(width_points - _A4_WIDTH_POINTS) > _A4_TOLERANCE_POINTS
        or abs(height_points - _A4_HEIGHT_POINTS) > _A4_TOLERANCE_POINTS
    ):
        raise RenderError(
            "rendered page size is not portrait A4 within the 1-point tolerance: "
            f"{width_points:g} x {height_points:g} pts"
        )
    return PdfInfo(
        pages=int(pages_match.group(1)),
        width_points=width_points,
        height_points=height_points,
    )


def _page_count(pdf_path: Path, *, pdfinfo: str) -> int:
    result = _run_command([pdfinfo, str(pdf_path)], label="pdfinfo")
    return parse_pdfinfo(result.stdout).pages


def _pdf_text(pdf_path: Path, *, pdftotext: str) -> str:
    try:
        return _pdf_text_bytes(pdf_path, pdftotext=pdftotext).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RenderError("pdftotext output is not valid UTF-8") from exc


def _pdf_text_bytes(pdf_path: Path, *, pdftotext: str) -> bytes:
    return _run_binary_command(
        [pdftotext, "-layout", str(pdf_path), "-"],
        label="pdftotext",
    )


def _page_geometries(
    pdf_path: Path,
    *,
    pages: int,
    pdfinfo: str,
) -> list[PageGeometry]:
    result = _run_command(
        [pdfinfo, "-f", "1", "-l", str(pages), str(pdf_path)],
        label="pdfinfo per-page geometry",
    )
    sizes = {
        int(match.group(1)): (float(match.group(2)), float(match.group(3)))
        for match in re.finditer(
            r"^Page\s+(\d+)\s+size:\s*([0-9]+(?:\.[0-9]+)?)\s+x\s+"
            r"([0-9]+(?:\.[0-9]+)?)\s+pts(?:\s+.*)?$",
            result.stdout,
            re.MULTILINE,
        )
    }
    rotations = {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(
            r"^Page\s+(\d+)\s+rot:\s*(-?\d+)\s*$",
            result.stdout,
            re.MULTILINE,
        )
    }
    expected = set(range(1, pages + 1))
    if set(sizes) != expected or set(rotations) != expected:
        raise RenderError("pdfinfo did not report size and rotation for every page")
    geometries: list[PageGeometry] = []
    for page in range(1, pages + 1):
        width, height = sizes[page]
        if (
            abs(width - _A4_WIDTH_POINTS) > _A4_TOLERANCE_POINTS
            or abs(height - _A4_HEIGHT_POINTS) > _A4_TOLERANCE_POINTS
            or rotations[page] != 0
        ):
            raise RenderError(
                "rendered page geometry is not unrotated portrait A4: "
                f"page {page}, {width:g} x {height:g} pts, rotation {rotations[page]}"
            )
        geometries.append(
            PageGeometry(
                page=page,
                width_points=width,
                height_points=height,
                rotation_degrees=rotations[page],
            )
        )
    return geometries


def _geometry_snapshot(
    pdf_path: Path,
    *,
    pages: int,
    pdfinfo: str,
) -> dict[str, Any]:
    rows = [
        {
            "page": geometry.page,
            "width_points": geometry.width_points,
            "height_points": geometry.height_points,
            "rotation_degrees": geometry.rotation_degrees,
        }
        for geometry in _page_geometries(pdf_path, pages=pages, pdfinfo=pdfinfo)
    ]
    return {"pages": rows, "sha256": hashlib.sha256(_canonical_json(rows)).hexdigest()}


def _raster_snapshot(
    pdf_path: Path,
    *,
    pages: int,
    pdftoppm: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=".paper-raster-") as temporary_directory:
        prefix = Path(temporary_directory) / "page"
        _run_command(
            [
                pdftoppm,
                "-r",
                str(RASTER_DPI),
                "-f",
                "1",
                "-l",
                str(pages),
                str(pdf_path),
                str(prefix),
            ],
            label="pdftoppm source-equivalence raster",
        )
        indexed: dict[int, Path] = {}
        for raster_path in Path(temporary_directory).glob("page-*.ppm"):
            match = re.fullmatch(r"page-(\d+)\.ppm", raster_path.name)
            if match is not None:
                indexed[int(match.group(1))] = raster_path
        expected = set(range(1, pages + 1))
        if set(indexed) != expected:
            raise RenderError("pdftoppm did not produce one PPM raster for every page")
        rows = []
        for page in range(1, pages + 1):
            data = indexed[page].read_bytes()
            rows.append(
                {
                    "page": page,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return {
        "dpi": RASTER_DPI,
        "format": "ppm-rgb",
        "pages": rows,
        "sha256": hashlib.sha256(_canonical_json(rows)).hexdigest(),
    }


def _pdf_equivalence_snapshot(
    pdf_path: Path,
    *,
    pages: int,
    pdfinfo: str,
    pdftotext: str,
    pdftoppm: str,
) -> dict[str, Any]:
    text_bytes = _pdf_text_bytes(pdf_path, pdftotext=pdftotext)
    return {
        "page_geometry": _geometry_snapshot(
            pdf_path, pages=pages, pdfinfo=pdfinfo
        ),
        "layout_text": {
            "bytes": len(text_bytes),
            "sha256": hashlib.sha256(text_bytes).hexdigest(),
        },
        "raster": _raster_snapshot(
            pdf_path, pages=pages, pdftoppm=pdftoppm
        ),
    }


def _load_reference_payload(path: Path) -> Mapping[str, Any]:
    try:
        data, _ = _stable_read_file(path, label="reference registry")
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RenderError) as exc:
        raise RenderError(f"could not read references from {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RenderError("reference payload must be a JSON object")
    return payload


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def print_html_with_chrome(
    *,
    chrome: str,
    html_path: Path,
    pdf_path: Path,
    user_data_path: Path,
    timeout_seconds: float = 20.0,
) -> None:
    """Print HTML and contain Chrome versions that remain alive after writing."""
    command = [
        chrome,
        "--headless=new",
        "--disable-background-networking",
        "--disable-gpu",
        "--disable-extensions",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--user-data-dir={user_data_path}",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RenderError(f"Chrome PDF print could not start: {exc}") from exc

    try:
        deadline = time.monotonic() + timeout_seconds
        last_size = -1
        stable_since: float | None = None
        timed_out = False
        while process.poll() is None:
            now = time.monotonic()
            try:
                size = pdf_path.stat().st_size
            except FileNotFoundError:
                size = 0
            if size > 0 and size == last_size:
                stable_since = stable_since or now
                if now - stable_since >= 0.25:
                    break
            else:
                last_size = size
                stable_since = None
            if now >= deadline:
                timed_out = True
                break
            time.sleep(0.05)

        if process.poll() is not None:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                detail = (stderr or stdout or "no diagnostic output").strip()
                raise RenderError(
                    f"Chrome PDF print failed with exit code {process.returncode}: "
                    f"{detail}"
                )

        if timed_out and (not pdf_path.is_file() or pdf_path.stat().st_size == 0):
            raise RenderError(
                f"Chrome PDF print timed out after {timeout_seconds:g} seconds"
            )
    finally:
        _stop_process_group(process)


def render_paper(
    *,
    profile: str,
    input_path: Path,
    output_path: Path,
    references_path: Path,
    pandoc: str,
    chrome: str,
    pdfinfo: str = "pdfinfo",
    pdftotext: str = "pdftotext",
    pdftoppm: str = "pdftoppm",
    expected_pages: int | None = None,
    receipt_path: Path | None = None,
) -> RenderResult:
    """Build HTML, print it with isolated Chrome, verify, then promote the PDF."""
    named_paths = {
        "manuscript": Path(input_path),
        "references": Path(references_path),
        "pdf": Path(output_path),
    }
    if receipt_path is not None:
        named_paths["receipt"] = Path(receipt_path)
    publication_paths = _assert_distinct_paths(**named_paths)
    input_path = publication_paths["manuscript"]
    references_path = publication_paths["references"]
    output_path = publication_paths["pdf"]
    rendered_receipt = publication_paths.get("receipt")
    if rendered_receipt is not None:
        _assert_outside_tree(
            rendered_receipt,
            input_path.parent,
            label="render receipt",
        )

    css = css_for_profile(profile)
    try:
        source_bytes, _ = _stable_read_file(input_path, label="paper manuscript")
        source = source_bytes.decode("utf-8")
    except (OSError, UnicodeError, RenderError) as exc:
        raise RenderError(f"could not read manuscript from {input_path}: {exc}") from exc

    if expected_pages is not None and expected_pages < 1:
        raise RenderError("expected page count must be positive")
    if profile == "yau" and expected_pages not in (None, 4):
        raise RenderError("the yau profile requires an exact four-page gate")

    prepared = prepare_markdown(source, profile=profile)
    csl_references = to_csl_references(_load_reference_payload(references_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rendered_receipt is not None:
        rendered_receipt.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".paper-pdf-", dir=output_path.parent
    ) as temporary_directory:
        workdir = Path(temporary_directory)
        csl_path = workdir / "references.csl.json"
        css_path = workdir / "paper.css"
        html_path = workdir / "paper.html"
        candidate_path = workdir / "paper.pdf"
        chrome_data_path = workdir / "chrome-data"

        csl_path.write_text(
            json.dumps(
                csl_references,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        css_path.write_text(css, encoding="utf-8")

        _run_command(
            [
                pandoc,
                "--from",
                "markdown+tex_math_single_backslash",
                "--to",
                "html5",
                "--standalone",
                "--embed-resources",
                "--citeproc",
                "--mathml",
                "--bibliography",
                str(csl_path),
                "--css",
                str(css_path),
                "--resource-path",
                str(input_path.parent),
                "--metadata",
                "link-citations=true",
                "--output",
                str(html_path),
            ],
            label="pandoc HTML render",
            input_text=prepared,
        )
        if not html_path.is_file() or html_path.stat().st_size == 0:
            raise RenderError("pandoc completed without producing HTML")

        print_html_with_chrome(
            chrome=chrome,
            html_path=html_path,
            pdf_path=candidate_path,
            user_data_path=chrome_data_path,
        )
        if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
            raise RenderError("Chrome completed without producing a PDF")

        pages = _page_count(candidate_path, pdfinfo=pdfinfo)
        rendered_text = _pdf_text(candidate_path, pdftotext=pdftotext)
        profile_expected_pages = 4 if profile == "yau" else expected_pages
        validate_rendered_text(
            rendered_text,
            pages=pages,
            expected_pages=profile_expected_pages,
        )
        if profile == "main" and not 8 <= pages <= 12:
            raise RenderError(
                f"rendered page count is {pages}; main profile requires 8-12 pages"
            )

        if rendered_receipt is None:
            candidate_path.replace(output_path)
        else:
            rollback_path: Path | None = None
            rollback_ready = False
            new_pdf_published = False
            staged_receipt: Path | None = None
            try:
                if output_path.exists():
                    descriptor, rollback_name = tempfile.mkstemp(
                        prefix=f".{output_path.name}.rollback-",
                        dir=output_path.parent,
                    )
                    os.close(descriptor)
                    rollback_path = Path(rollback_name)
                    rollback_path.unlink()
                    output_path.replace(rollback_path)
                    rollback_ready = True
                candidate_path.replace(output_path)
                new_pdf_published = True
                descriptor, staged_name = tempfile.mkstemp(
                    prefix=f".{rendered_receipt.name}.staged-",
                    dir=rendered_receipt.parent,
                )
                os.close(descriptor)
                staged_receipt = Path(staged_name)
                staged_receipt.unlink()
                write_render_receipt(
                    profile=profile,
                    input_path=input_path,
                    references_path=references_path,
                    output_path=output_path,
                    pages=pages,
                    pandoc=pandoc,
                    chrome=chrome,
                    receipt_path=staged_receipt,
                    pdfinfo=pdfinfo,
                    pdftotext=pdftotext,
                    pdftoppm=pdftoppm,
                )
                staged_receipt.replace(rendered_receipt)
                staged_receipt = None
            except BaseException:
                if staged_receipt is not None:
                    staged_receipt.unlink(missing_ok=True)
                if new_pdf_published:
                    output_path.unlink(missing_ok=True)
                if rollback_ready and rollback_path is not None:
                    rollback_path.replace(output_path)
                    rollback_ready = False
                raise
            finally:
                if rollback_ready and rollback_path is not None:
                    rollback_path.unlink(missing_ok=True)
    return RenderResult(
        output_path=output_path,
        pages=pages,
        receipt_path=rendered_receipt,
    )


def verify_source_bound_pdf(
    *,
    profile: str,
    input_path: Path,
    references_path: Path,
    output_path: Path,
    pages: int,
    pandoc: str,
    chrome: str,
    pdfinfo: str = "pdfinfo",
    pdftotext: str = "pdftotext",
    pdftoppm: str = "pdftoppm",
) -> dict[str, Any]:
    """Prove a PDF matches an independent render of its bound source inputs."""
    target = Path(output_path).expanduser().resolve(strict=True)
    target_snapshot = _pdf_equivalence_snapshot(
        target,
        pages=pages,
        pdfinfo=pdfinfo,
        pdftotext=pdftotext,
        pdftoppm=pdftoppm,
    )
    with tempfile.TemporaryDirectory(prefix=".paper-source-proof-") as temporary_directory:
        independent_pdf = Path(temporary_directory) / "independent.pdf"
        independent_result = render_paper(
            profile=profile,
            input_path=input_path,
            output_path=independent_pdf,
            references_path=references_path,
            pandoc=pandoc,
            chrome=chrome,
            pdfinfo=pdfinfo,
            pdftotext=pdftotext,
            pdftoppm=pdftoppm,
            expected_pages=pages,
            receipt_path=None,
        )
        if independent_result.pages != pages:
            raise RenderError("independent source rerender page count differs")
        independent_snapshot = _pdf_equivalence_snapshot(
            independent_pdf,
            pages=pages,
            pdfinfo=pdfinfo,
            pdftotext=pdftotext,
            pdftoppm=pdftoppm,
        )

    if target_snapshot["page_geometry"] != independent_snapshot["page_geometry"]:
        raise RenderError("independent source rerender page geometry differs")
    if target_snapshot["layout_text"] != independent_snapshot["layout_text"]:
        raise RenderError("independent source rerender layout text differs")
    if target_snapshot["raster"] != independent_snapshot["raster"]:
        raise RenderError("independent source rerender raster differs")
    return {
        "schema_version": SOURCE_EQUIVALENCE_SCHEMA,
        "method": "independent-source-rerender",
        "comparison": {
            "page_geometry": "exact",
            "layout_text_bytes": "exact",
            "raster_bytes": "exact",
        },
        "snapshot": target_snapshot,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("main", "yau"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--pandoc", default="/opt/homebrew/bin/pandoc")
    parser.add_argument(
        "--chrome",
        default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    parser.add_argument("--pdfinfo", default="pdfinfo")
    parser.add_argument("--pdftotext", default="pdftotext")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render_paper(
            profile=args.profile,
            input_path=args.input,
            output_path=args.output,
            references_path=args.references,
            pandoc=args.pandoc,
            chrome=args.chrome,
            pdfinfo=args.pdfinfo,
            pdftotext=args.pdftotext,
            pdftoppm=args.pdftoppm,
            expected_pages=args.expected_pages,
            receipt_path=args.receipt,
        )
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"rendered {result.pages} pages to {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

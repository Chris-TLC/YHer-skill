#!/usr/bin/env python3
"""Render a bound paper manuscript to a verified A4 PDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


REFERENCE_SCHEMA = "yher.verified-references.v1"


class RenderError(RuntimeError):
    """Raised when paper rendering cannot satisfy a release gate."""


@dataclass(frozen=True)
class RenderResult:
    """Verified output metadata returned after atomic PDF promotion."""

    output_path: Path
    pages: int


@dataclass(frozen=True)
class PdfInfo:
    """Poppler metadata needed by the PDF release gates."""

    pages: int
    width_points: float
    height_points: float


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


def prepare_markdown(source: str) -> str:
    """Remove the terminal hand-written bibliography so citeproc owns it once."""
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
        font_size = "10.5pt"
        line_height = "1.38"
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

.math.display {{
  display: block;
  margin: 0.3em 0;
  overflow: visible;
  text-align: center;
}}

#refs {{
  column-count: 2;
  column-gap: 5mm;
  font-size: 0.82em;
  line-height: 1.16;
}}

#refs > div {{
  break-inside: avoid;
  margin-bottom: 0.3em;
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
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            input=input_text,
            text=True,
        )
    except OSError as exc:
        raise RenderError(f"{label} could not start: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise RenderError(f"{label} failed with exit code {result.returncode}: {detail}")
    return result


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
    result = _run_command(
        [pdftotext, "-layout", str(pdf_path), "-"],
        label="pdftotext",
    )
    return result.stdout


def _load_reference_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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
    expected_pages: int | None = None,
) -> RenderResult:
    """Build HTML, print it with isolated Chrome, verify, then promote the PDF."""
    css = css_for_profile(profile)
    try:
        source = input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RenderError(f"could not read manuscript from {input_path}: {exc}") from exc

    if expected_pages is not None and expected_pages < 1:
        raise RenderError("expected page count must be positive")
    if profile == "yau" and expected_pages not in (None, 4):
        raise RenderError("the yau profile requires an exact four-page gate")

    prepared = prepare_markdown(source)
    csl_references = to_csl_references(_load_reference_payload(references_path))
    output_path = output_path.resolve()
    input_path = input_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                "markdown",
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

        candidate_path.replace(output_path)

    return RenderResult(output_path=output_path, pages=pages)


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
    parser.add_argument("--expected-pages", type=int)
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
            expected_pages=args.expected_pages,
        )
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"rendered {result.pages} pages to {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

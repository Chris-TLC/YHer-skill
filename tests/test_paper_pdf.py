from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from scripts import render_paper_pdf as paper_pdf


SCRIPT = Path(__file__).parents[1] / "scripts/render_paper_pdf.py"
PANDOC = Path(shutil.which("pandoc") or "/opt/homebrew/bin/pandoc")
CHROME = Path(
    os.environ.get(
        "PAPER_CHROME",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
)
PDFINFO = Path(shutil.which("pdfinfo") or "/opt/homebrew/bin/pdfinfo")
PDFTOTEXT = Path(shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext")
BOUND_RENDER_TOOLS_AVAILABLE = all(
    path.is_file() for path in (PANDOC, CHROME, PDFINFO, PDFTOTEXT)
)


def test_source_tree_snapshot_rejects_special_entries(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manuscript.md").write_text("# Paper\n", encoding="utf-8")
    os.mkfifo(source / "unexpected-pipe")

    with pytest.raises(paper_pdf.RenderError, match="special|regular"):
        paper_pdf._snapshot_source_tree(source, tmp_path / "snapshot")


def test_stable_read_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "unexpected-pipe"
    os.mkfifo(fifo)
    script = (
        "from pathlib import Path\n"
        "from scripts import render_paper_pdf as paper_pdf\n"
        "try:\n"
        f"    paper_pdf._stable_read_file(Path({str(fifo)!r}), label='fifo')\n"
        "except paper_pdf.RenderError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=SCRIPT.parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            returncode = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            pytest.fail("stable read blocked while opening a FIFO")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    assert returncode == 0, process.stderr.read()


def test_file_stability_ignores_ctime_only_metadata_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.md"
    source.write_text("stable bytes\n", encoding="utf-8")
    data, expected = paper_pdf._stable_read_file(source, label="source")
    current = {**expected, "ctime_ns": expected["ctime_ns"] + 1}
    monkeypatch.setattr(
        paper_pdf,
        "_stable_read_file",
        lambda path, *, label: (data, current),
    )

    paper_pdf._assert_file_unchanged(source, expected, label="source")


def test_local_tool_timeout_stops_the_process_group(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(child_pid)!r}, 'w', encoding='utf-8').write(str(child.pid))\n"
        "time.sleep(30)\n"
    )

    with pytest.raises(paper_pdf.RenderError, match="timed out"):
        paper_pdf._run_command(
            [sys.executable, "-c", script],
            label="hung fixture",
            timeout_seconds=0.2,
        )

    assert child_pid.is_file()
    pid = int(child_pid.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pytest.fail("timed-out tool child process remained alive")


@pytest.mark.parametrize("alias", ("input", "references", "receipt"))
def test_render_paper_rejects_write_path_aliases_before_running_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("# Paper\n", encoding="utf-8")
    references = tmp_path / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    output = tmp_path / "paper.pdf"
    receipt = tmp_path / "paper.pdf.render.json"
    if alias == "input":
        output = manuscript
    elif alias == "references":
        output = references
    else:
        receipt = output
    original_manuscript = manuscript.read_bytes()
    original_references = references.read_bytes()

    monkeypatch.setattr(
        paper_pdf,
        "_run_command",
        lambda *args, **kwargs: pytest.fail("tool ran before alias rejection"),
    )
    with pytest.raises(paper_pdf.RenderError, match="distinct|alias"):
        paper_pdf.render_paper(
            profile="main",
            input_path=manuscript,
            output_path=output,
            references_path=references,
            pandoc="pandoc-bin",
            chrome="chrome-bin",
            receipt_path=receipt,
        )

    assert manuscript.read_bytes() == original_manuscript
    assert references.read_bytes() == original_references


def test_receipt_failure_rolls_back_pdf_and_preserves_prior_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    manuscript = source / "paper.md"
    manuscript.write_text("# Paper\n", encoding="utf-8")
    references = source / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    output = publication / "paper.pdf"
    output.write_bytes(b"OLD-PDF")
    receipt = publication / "paper.pdf.render.json"
    receipt.write_bytes(b"OLD-RECEIPT")

    def fake_run(
        command: list[str],
        *,
        label: str,
        input_text: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        html = Path(command[command.index("--output") + 1])
        html.write_text("<!doctype html><title>Paper</title>", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_chrome(**kwargs: object) -> None:
        Path(kwargs["pdf_path"]).write_bytes(b"NEW-PDF")

    monkeypatch.setattr(paper_pdf, "_run_command", fake_run)
    monkeypatch.setattr(paper_pdf, "print_html_with_chrome", fake_chrome)
    monkeypatch.setattr(paper_pdf, "_page_count", lambda *args, **kwargs: 8)
    monkeypatch.setattr(
        paper_pdf, "_pdf_text", lambda *args, **kwargs: "Paper\nBound result.\n"
    )
    monkeypatch.setattr(
        paper_pdf,
        "write_render_receipt",
        lambda **kwargs: (_ for _ in ()).throw(paper_pdf.RenderError("proof failed")),
    )

    with pytest.raises(paper_pdf.RenderError, match="proof failed"):
        paper_pdf.render_paper(
            profile="main",
            input_path=manuscript,
            output_path=output,
            references_path=references,
            pandoc="pandoc-bin",
            chrome="chrome-bin",
            receipt_path=receipt,
        )

    assert output.read_bytes() == b"OLD-PDF"
    assert receipt.read_bytes() == b"OLD-RECEIPT"


def test_backup_move_failure_preserves_existing_pdf_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    manuscript = source / "paper.md"
    manuscript.write_text("# Paper\n", encoding="utf-8")
    references = source / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    output = publication / "paper.pdf"
    output.write_bytes(b"OLD-PDF")
    receipt = publication / "paper.pdf.render.json"
    receipt.write_bytes(b"OLD-RECEIPT")

    def fake_run(
        command: list[str],
        *,
        label: str,
        input_text: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        del label, input_text, timeout_seconds
        html = Path(command[command.index("--output") + 1])
        html.write_text("<!doctype html><title>Paper</title>", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_chrome(**kwargs: object) -> None:
        Path(kwargs["pdf_path"]).write_bytes(b"NEW-PDF")

    real_replace = Path.replace

    def fail_backup_move(self: Path, target: Path) -> Path:
        target = Path(target)
        if self == output and target.name.startswith(f".{output.name}.rollback-"):
            raise OSError("backup move failed")
        return real_replace(self, target)

    monkeypatch.setattr(paper_pdf, "_run_command", fake_run)
    monkeypatch.setattr(paper_pdf, "print_html_with_chrome", fake_chrome)
    monkeypatch.setattr(paper_pdf, "_page_count", lambda *args, **kwargs: 8)
    monkeypatch.setattr(
        paper_pdf, "_pdf_text", lambda *args, **kwargs: "Paper\nBound result.\n"
    )
    monkeypatch.setattr(Path, "replace", fail_backup_move)

    with pytest.raises(OSError, match="backup move failed"):
        paper_pdf.render_paper(
            profile="main",
            input_path=manuscript,
            output_path=output,
            references_path=references,
            pandoc="pandoc-bin",
            chrome="chrome-bin",
            receipt_path=receipt,
        )

    assert output.read_bytes() == b"OLD-PDF"
    assert receipt.read_bytes() == b"OLD-RECEIPT"


def test_render_receipt_rejects_receipt_pdf_alias_before_tool_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("# Paper\n", encoding="utf-8")
    references = tmp_path / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"ORIGINAL-PDF")
    monkeypatch.setattr(
        paper_pdf,
        "_render_tool_descriptors",
        lambda **kwargs: pytest.fail("tool ran before alias rejection"),
    )

    with pytest.raises(paper_pdf.RenderError, match="distinct|alias"):
        paper_pdf.write_render_receipt(
            profile="main",
            input_path=manuscript,
            references_path=references,
            output_path=pdf,
            pages=8,
            pandoc="pandoc-bin",
            chrome="chrome-bin",
            receipt_path=pdf,
        )

    assert pdf.read_bytes() == b"ORIGINAL-PDF"


def test_render_receipt_rejects_destination_inside_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manuscript = source / "paper.md"
    manuscript.write_text("# Paper\n", encoding="utf-8")
    references = source / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"ORIGINAL-PDF")
    receipt = source / "nested" / "paper.pdf.render.json"
    monkeypatch.setattr(
        paper_pdf,
        "_render_tool_descriptors",
        lambda **kwargs: pytest.fail("tool ran before source-tree rejection"),
    )

    with pytest.raises(paper_pdf.RenderError, match="source.*tree|outside"):
        paper_pdf.write_render_receipt(
            profile="main",
            input_path=manuscript,
            references_path=references,
            output_path=pdf,
            pages=8,
            pandoc="pandoc-bin",
            chrome="chrome-bin",
            receipt_path=receipt,
        )

    assert not receipt.parent.exists()


def test_renderer_cli_advertises_reproducible_inputs_and_gates() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for option in (
        "--profile",
        "--input",
        "--output",
        "--references",
        "--pandoc",
        "--chrome",
        "--pdftoppm",
        "--expected-pages",
    ):
        assert option in result.stdout


def test_render_receipt_rejects_a_pseudo_pdf(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    manuscript = source / "manuscript.md"
    manuscript.write_text("# Paper\n\nRendered evidence.\n", encoding="utf-8")
    references = source / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    pseudo_pdf = publication / "paper.pdf"
    pseudo_pdf.write_bytes(b"%PDF-1.7\npseudo\n%%EOF\n")

    with pytest.raises(paper_pdf.RenderError):
        paper_pdf.write_render_receipt(
            profile="main",
            input_path=manuscript,
            references_path=references,
            output_path=pseudo_pdf,
            pages=9,
            pandoc="fixture-pandoc",
            chrome="fixture-chrome",
            receipt_path=publication / "paper.pdf.render.json",
        )


@pytest.mark.skipif(
    not BOUND_RENDER_TOOLS_AVAILABLE,
    reason="Pandoc, Chrome, and Poppler are required for source-binding render proof",
)
def test_render_receipt_rejects_a_valid_pdf_from_another_manuscript(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    references = source / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )

    def manuscript(label: str) -> str:
        pages = []
        for page in range(1, 9):
            pages.append(
                f"# {label} page {page}\n\n"
                f"This page belongs only to {label}.\n"
                + (
                    '\n<div style="break-after: page"></div>\n'
                    if page < 8
                    else ""
                )
            )
        return "\n".join(pages)

    expected = source / "expected.md"
    expected.write_text(manuscript("expected manuscript"), encoding="utf-8")
    foreign = source / "foreign.md"
    foreign.write_text(manuscript("foreign manuscript"), encoding="utf-8")
    foreign_pdf = publication / "foreign.pdf"
    rendered = paper_pdf.render_paper(
        profile="main",
        input_path=foreign,
        output_path=foreign_pdf,
        references_path=references,
        pandoc=str(PANDOC),
        chrome=str(CHROME),
        pdfinfo=str(PDFINFO),
        pdftotext=str(PDFTOTEXT),
    )

    with pytest.raises(paper_pdf.RenderError, match="independent|source|raster|text"):
        paper_pdf.write_render_receipt(
            profile="main",
            input_path=expected,
            references_path=references,
            output_path=foreign_pdf,
            pages=rendered.pages,
            pandoc=str(PANDOC),
            chrome=str(CHROME),
            receipt_path=publication / "foreign.pdf.render.json",
            pdfinfo=str(PDFINFO),
            pdftotext=str(PDFTOTEXT),
        )


@pytest.mark.skipif(
    not BOUND_RENDER_TOOLS_AVAILABLE,
    reason="Pandoc, Chrome, and Poppler are required for source snapshot proof",
)
def test_render_receipt_rejects_source_replacement_during_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    references = source / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    manuscript = source / "manuscript.md"
    pages = [
        f"# Stable source page {page}\n\nAAAA source payload.\n"
        + ('\n<div style="break-after: page"></div>\n' if page < 8 else "")
        for page in range(1, 9)
    ]
    manuscript.write_text("\n".join(pages), encoding="utf-8")
    pdf = publication / "paper.pdf"
    rendered = paper_pdf.render_paper(
        profile="main",
        input_path=manuscript,
        output_path=pdf,
        references_path=references,
        pandoc=str(PANDOC),
        chrome=str(CHROME),
        pdfinfo=str(PDFINFO),
        pdftotext=str(PDFTOTEXT),
    )

    def replace_source(**_: object) -> dict[str, object]:
        original = manuscript.read_text(encoding="utf-8")
        manuscript.write_text(original.replace("AAAA", "BBBB"), encoding="utf-8")
        return {
            "schema_version": "yher.paper_pdf.source_equivalence.v1",
            "method": "independent-source-rerender",
            "comparison": {
                "page_geometry": "exact",
                "layout_text_bytes": "exact",
                "raster_bytes": "exact",
            },
            "snapshot": {},
        }

    monkeypatch.setattr(paper_pdf, "verify_source_bound_pdf", replace_source)

    with pytest.raises(paper_pdf.RenderError, match="source.*changed|changed.*source"):
        paper_pdf.write_render_receipt(
            profile="main",
            input_path=manuscript,
            references_path=references,
            output_path=pdf,
            pages=rendered.pages,
            pandoc=str(PANDOC),
            chrome=str(CHROME),
            receipt_path=publication / "paper.pdf.render.json",
            pdfinfo=str(PDFINFO),
            pdftotext=str(PDFTOTEXT),
        )


@pytest.mark.skipif(
    not BOUND_RENDER_TOOLS_AVAILABLE,
    reason="Pandoc, Chrome, and Poppler are required for tool identity proof",
)
def test_render_receipt_rejects_tool_identity_change_during_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    publication = tmp_path / "publication"
    publication.mkdir()
    references = source / "references.json"
    references.write_text(
        json.dumps({"schema_version": "yher.verified-references.v1", "references": []}),
        encoding="utf-8",
    )
    manuscript = source / "manuscript.md"
    manuscript.write_text(
        "\n".join(
            f"# Tool identity page {page}\n\nStable.\n"
            + ('\n<div style="break-after: page"></div>\n' if page < 8 else "")
            for page in range(1, 9)
        ),
        encoding="utf-8",
    )
    pdf = publication / "paper.pdf"
    rendered = paper_pdf.render_paper(
        profile="main",
        input_path=manuscript,
        output_path=pdf,
        references_path=references,
        pandoc=str(PANDOC),
        chrome=str(CHROME),
        pdfinfo=str(PDFINFO),
        pdftotext=str(PDFTOTEXT),
    )
    proof = {
        "schema_version": "yher.paper_pdf.source_equivalence.v1",
        "method": "independent-source-rerender",
        "comparison": {
            "page_geometry": "exact",
            "layout_text_bytes": "exact",
            "raster_bytes": "exact",
        },
        "snapshot": {},
    }
    monkeypatch.setattr(
        paper_pdf,
        "verify_source_bound_pdf",
        lambda **_: proof,
    )
    calls: dict[str, int] = {}

    def changing_descriptor(command: str, *, version_flag: str) -> dict[str, object]:
        del version_flag
        calls[command] = calls.get(command, 0) + 1
        return {
            "command": command,
            "resolved_path": command,
            "version": "fixture 1",
            "device": 1,
            "inode": 1 if calls[command] == 1 else 2,
            "bytes": 100,
            "mtime_ns": 1,
            "sha256": "a" * 64,
        }

    monkeypatch.setattr(paper_pdf, "_tool_descriptor", changing_descriptor)

    with pytest.raises(paper_pdf.RenderError, match="tool.*changed|changed.*tool"):
        paper_pdf.write_render_receipt(
            profile="main",
            input_path=manuscript,
            references_path=references,
            output_path=pdf,
            pages=rendered.pages,
            pandoc=str(PANDOC),
            chrome=str(CHROME),
            receipt_path=publication / "paper.pdf.render.json",
            pdfinfo=str(PDFINFO),
            pdftotext=str(PDFTOTEXT),
        )


def test_reference_conversion_produces_citeproc_csl_records() -> None:
    payload = {
        "schema_version": "yher.verified-references.v1",
        "references": [
            {
                "id": "example2026",
                "type": "journal-article",
                "authors": ["A. Example", "B. Reviewer"],
                "title": "Verified Work",
                "year": 2026,
                "container_title": "Journal of Tests",
                "volume": "4",
                "issue": "2",
                "pages": "10-19",
                "doi": "10.1234/example",
                "url": "https://doi.org/10.1234/example",
            }
        ],
    }

    records = paper_pdf.to_csl_references(payload)

    assert records == [
        {
            "id": "example2026",
            "type": "article-journal",
            "author": [{"literal": "A. Example"}, {"literal": "B. Reviewer"}],
            "title": "Verified Work",
            "issued": {"date-parts": [[2026]]},
            "container-title": "Journal of Tests",
            "volume": "4",
            "issue": "2",
            "page": "10-19",
            "DOI": "10.1234/example",
            "URL": "https://doi.org/10.1234/example",
        }
    ]


def test_prepare_markdown_keeps_citations_but_removes_manual_bibliography() -> None:
    source = (
        "# Paper\n\nEvidence [@example2026].\n\n"
        "## References\n\n1. Manual duplicate.\n"
    )

    prepared = paper_pdf.prepare_markdown(source)

    assert "Evidence [@example2026]." in prepared
    assert "## References" not in prepared
    assert "Manual duplicate" not in prepared
    assert prepared.endswith("\n")


def test_main_profile_renders_the_persona_summary_at_publication_size() -> None:
    css = paper_pdf.css_for_profile("main")

    assert ".persona-v2-composite" in css
    assert "margin-left: 0;" in css
    assert "margin-right: 0;" in css
    assert ".persona-v2-summary" in css
    assert "width: 100%;" in css
    content_width_mm = 210 - 18 - 18
    minimum_font_points = 16 / 1000 * content_width_mm * 72 / 25.4
    assert minimum_font_points >= 7.0


def test_prepare_markdown_preserves_references_heading_inside_fenced_code() -> None:
    source = (
        "# Paper\n\n"
        "```markdown\n## References\n1. Fixture entry.\n```\n\n"
        "Closing argument.\n"
    )

    assert paper_pdf.prepare_markdown(source) == source


def test_prepare_markdown_preserves_references_section_before_appendix() -> None:
    source = (
        "# Paper\n\nEvidence [@example2026].\n\n"
        "## References\n\n1. Manual entry.\n\n"
        "## Appendix\n\nAudit details.\n"
    )

    assert paper_pdf.prepare_markdown(source) == source


def test_prepare_markdown_preserves_terminal_non_bibliography_heading() -> None:
    source = "# Paper\n\n## References\n\nReferences are discussed here.\n"

    assert paper_pdf.prepare_markdown(source) == source


def test_prepare_markdown_omits_yau_machine_audit_from_pdf_source() -> None:
    source = (
        "# Four-page paper\n\nVisible results.\n\n"
        "<!-- BEGIN YAU MACHINE AUDIT -->\n"
        "machine-only provenance\n"
        "<!-- END YAU MACHINE AUDIT -->\n\n"
        "Visible discussion.\n"
    )

    prepared = paper_pdf.prepare_markdown(source, profile="yau")

    assert "Visible results." in prepared
    assert "machine-only provenance" not in prepared
    assert "Visible discussion." in prepared


@pytest.mark.parametrize(
    "source",
    (
        "<!-- BEGIN YAU MACHINE AUDIT -->\nunclosed\n",
        "orphan\n<!-- END YAU MACHINE AUDIT -->\n",
        (
            "<!-- END YAU MACHINE AUDIT -->\nreversed\n"
            "<!-- BEGIN YAU MACHINE AUDIT -->\n"
        ),
        (
            "<!-- BEGIN YAU MACHINE AUDIT -->\none\n"
            "<!-- END YAU MACHINE AUDIT -->\n"
            "<!-- BEGIN YAU MACHINE AUDIT -->\ntwo\n"
            "<!-- END YAU MACHINE AUDIT -->\n"
        ),
    ),
)
def test_prepare_markdown_rejects_malformed_yau_machine_audit(source: str) -> None:
    with pytest.raises(paper_pdf.RenderError, match="Yau machine audit"):
        paper_pdf.prepare_markdown(source, profile="yau")


def test_yau_css_is_one_column_except_for_references_and_compacts_figures() -> None:
    css = paper_pdf.css_for_profile("yau")

    assert "@page" in css and "size: A4" in css
    assert "font-size: 9.2pt" in css
    assert "#refs" in css and "column-count: 2" in css
    assert "body {" in css
    assert "body {\n  column-count" not in css
    assert "figure-grid" in css


@pytest.mark.parametrize(
    "unresolved",
    (
        "PENDING",
        "raw [@example2026] citation",
        "raw [-@example2026] suppressed-author citation",
        "raw [see @example2026, p. 4] citation with prefix",
        "raw textual @example2026 citation",
        r"raw \gamma token",
        r"raw \mid token",
        r"raw \[ display math",
        r"raw \alpha token",
        r"raw \frac{1}{2} token",
        r"raw \text{label} token",
        r"raw \begin{aligned} token",
        r"raw \theta token",
        r"raw \rightarrow token",
        r"raw \overset{a}{b} token",
        r"raw \binom{n}{k} token",
        r"raw \langle token",
        r"raw \displaystyle token",
        r"raw \customcontrol{value} token",
    ),
)
def test_render_validation_rejects_unresolved_tokens(unresolved: str) -> None:
    with pytest.raises(paper_pdf.RenderError, match="unresolved"):
        paper_pdf.validate_rendered_text(
            f"Bound result text. {unresolved}",
            pages=4,
            expected_pages=4,
        )


@pytest.mark.parametrize(
    "ordinary_text",
    (
        r"Bound result text with C:\Users\student.",
        r"Bound result text with \\server\share\paper.pdf.",
        r"Bound result text with A \ B.",
        "Contact student@example.com for the bound result.",
    ),
)
def test_render_validation_allows_non_source_tokens(ordinary_text: str) -> None:
    paper_pdf.validate_rendered_text(
        ordinary_text,
        pages=4,
        expected_pages=4,
    )


@pytest.mark.parametrize("empty_text", ("", " \t\n\f\f"))
def test_render_validation_rejects_empty_extracted_text(empty_text: str) -> None:
    with pytest.raises(paper_pdf.RenderError, match="empty"):
        paper_pdf.validate_rendered_text(
            empty_text,
            pages=4,
            expected_pages=4,
        )


def test_render_validation_enforces_exact_page_gate() -> None:
    with pytest.raises(paper_pdf.RenderError, match="page count"):
        paper_pdf.validate_rendered_text(
            "Bound result text.",
            pages=5,
            expected_pages=4,
        )


def test_pdfinfo_parser_accepts_a4_within_point_tolerance() -> None:
    metadata = paper_pdf.parse_pdfinfo(
        "Pages:           4\n"
        "Page size:       594.96 x 841.92 pts (A4)\n"
    )

    assert metadata.pages == 4
    assert metadata.width_points == pytest.approx(594.96)
    assert metadata.height_points == pytest.approx(841.92)


@pytest.mark.parametrize(
    "pdfinfo_output",
    (
        "Pages: 4\nPage size: 612 x 792 pts (letter)\n",
        "Pages: 4\n",
    ),
)
def test_pdfinfo_parser_rejects_non_a4_or_missing_size(
    pdfinfo_output: str,
) -> None:
    with pytest.raises(paper_pdf.RenderError, match="A4|page size"):
        paper_pdf.parse_pdfinfo(pdfinfo_output)


def test_render_paper_runs_pandoc_chrome_and_poppler_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manuscript = tmp_path / "paper.md"
    manuscript.write_text(
        "# Paper\n\nEvidence [@example2026].\n\n"
        "## References\n\n1. Manual duplicate.\n",
        encoding="utf-8",
    )
    references = tmp_path / "references.json"
    references.write_text(
        json.dumps(
            {
                "schema_version": "yher.verified-references.v1",
                "references": [
                    {
                        "id": "example2026",
                        "type": "journal-article",
                        "authors": ["A. Example"],
                        "title": "Verified Work",
                        "year": 2026,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output" / "paper.pdf"
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        label: str,
        input_text: str | None = None,
        timeout_seconds: float = paper_pdf.LOCAL_TOOL_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        del label, timeout_seconds
        command = [str(part) for part in command]
        calls.append(command)
        executable = command[0]
        if executable == "pandoc-bin":
            assert input_text == "# Paper\n\nEvidence [@example2026].\n"
            assert "--standalone" in command
            assert command[command.index("--from") + 1] == (
                "markdown+tex_math_single_backslash"
            )
            assert "--citeproc" in command
            assert "--mathml" in command
            bibliography = Path(command[command.index("--bibliography") + 1])
            csl_records = json.loads(bibliography.read_text(encoding="utf-8"))
            assert csl_records[0]["type"] == "article-journal"
            html = Path(command[command.index("--output") + 1])
            html.write_text("<!doctype html><title>Paper</title>", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if executable == "pdfinfo-bin":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "Pages:           4\n"
                    "Page size:       594.96 x 841.92 pts (A4)\n"
                ),
                stderr="",
            )
        if executable == "pdftotext-bin":
            assert command[-1] == "-"
            return subprocess.CompletedProcess(
                command, 0, stdout="Paper\nBound result text.\n", stderr=""
            )
        raise AssertionError(f"unexpected executable: {executable}")

    def fake_chrome_print(
        *,
        chrome: str,
        html_path: Path,
        pdf_path: Path,
        user_data_path: Path,
        timeout_seconds: float = 20.0,
    ) -> None:
        assert html_path.is_file()
        assert user_data_path.name == "chrome-data"
        assert timeout_seconds == 20.0
        calls.append([chrome])
        pdf_path.write_bytes(b"candidate pdf")

    def fake_binary_run(
        command: list[str],
        *,
        label: str,
        timeout_seconds: float = paper_pdf.LOCAL_TOOL_TIMEOUT_SECONDS,
    ) -> bytes:
        result = fake_run(
            command,
            label=label,
            timeout_seconds=timeout_seconds,
        )
        return result.stdout.encode("utf-8")

    monkeypatch.setattr(paper_pdf, "_run_command", fake_run)
    monkeypatch.setattr(paper_pdf, "_run_binary_command", fake_binary_run)
    monkeypatch.setattr(paper_pdf, "print_html_with_chrome", fake_chrome_print)

    result = paper_pdf.render_paper(
        profile="yau",
        input_path=manuscript,
        output_path=output,
        references_path=references,
        pandoc="pandoc-bin",
        chrome="chrome-bin",
        pdfinfo="pdfinfo-bin",
        pdftotext="pdftotext-bin",
        expected_pages=4,
    )

    assert result.pages == 4
    assert result.output_path == output
    assert output.read_bytes() == b"candidate pdf"
    assert [call[0] for call in calls] == [
        "pandoc-bin",
        "chrome-bin",
        "pdfinfo-bin",
        "pdftotext-bin",
    ]


def test_chrome_print_stops_a_hung_process_after_pdf_is_complete(
    tmp_path: Path,
) -> None:
    html = tmp_path / "paper.html"
    html.write_text("<!doctype html><title>Paper</title>", encoding="utf-8")
    output = tmp_path / "paper.pdf"
    arguments = tmp_path / "arguments.txt"
    process_id = tmp_path / "pid.txt"
    fake_chrome = tmp_path / "fake-chrome"
    fake_chrome.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$$\" > \"{process_id}\"\n"
        f"printf '%s\\n' \"$@\" > \"{arguments}\"\n"
        "for argument in \"$@\"; do\n"
        "  case \"$argument\" in\n"
        "    --print-to-pdf=*) pdf=${argument#--print-to-pdf=} ;;\n"
        "  esac\n"
        "done\n"
        "printf 'candidate pdf' > \"$pdf\"\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    fake_chrome.chmod(0o755)

    paper_pdf.print_html_with_chrome(
        chrome=str(fake_chrome),
        html_path=html,
        pdf_path=output,
        user_data_path=tmp_path / "chrome-data",
        timeout_seconds=1.0,
    )

    assert output.read_bytes() == b"candidate pdf"
    recorded_arguments = arguments.read_text(encoding="utf-8").splitlines()
    assert "--headless=new" in recorded_arguments
    assert f"--print-to-pdf={output}" in recorded_arguments
    assert recorded_arguments[-1] == html.as_uri()
    with pytest.raises(ProcessLookupError):
        os.kill(int(process_id.read_text(encoding="utf-8")), 0)


def test_chrome_print_cleans_process_group_when_polling_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = tmp_path / "paper.html"
    html.write_text("<!doctype html><title>Paper</title>", encoding="utf-8")
    process_id = tmp_path / "pid.txt"
    fake_chrome = tmp_path / "fake-chrome"
    fake_chrome.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$$\" > \"{process_id}\"\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    fake_chrome.chmod(0o755)
    real_sleep = time.sleep

    def raise_after_process_starts(_: float) -> None:
        deadline = time.monotonic() + 2
        while not process_id.exists() and time.monotonic() < deadline:
            real_sleep(0.01)
        raise RuntimeError("polling failed")

    monkeypatch.setattr(paper_pdf.time, "sleep", raise_after_process_starts)
    pid: int | None = None
    try:
        with pytest.raises(RuntimeError, match="polling failed"):
            paper_pdf.print_html_with_chrome(
                chrome=str(fake_chrome),
                html_path=html,
                pdf_path=tmp_path / "paper.pdf",
                user_data_path=tmp_path / "chrome-data",
                timeout_seconds=1.0,
            )
        pid = int(process_id.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(
    not Path("/opt/homebrew/bin/pandoc").is_file(),
    reason="Pandoc is required for renderer integration",
)
def test_pandoc_fenced_div_matches_figure_grid_css() -> None:
    markdown = "::: figure-grid\n\nFirst figure\n\nSecond figure\n\n:::\n"
    result = subprocess.run(
        ["/opt/homebrew/bin/pandoc", "--from", "markdown", "--to", "html5"],
        input=markdown,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '<div class="figure-grid">' in result.stdout
    assert ".figure-grid {" in paper_pdf.css_for_profile("yau")


@pytest.mark.skipif(
    not BOUND_RENDER_TOOLS_AVAILABLE,
    reason="Pandoc, Chrome, and Poppler are required for bound-paper rendering",
)
def test_bound_main_manuscript_fits_release_page_gate(tmp_path: Path) -> None:
    repository = SCRIPT.parents[1]

    result = paper_pdf.render_paper(
        profile="main",
        input_path=repository / "docs/paper/main.md",
        output_path=tmp_path / "main.pdf",
        references_path=repository / "docs/paper/references.json",
        pandoc=str(PANDOC),
        chrome=str(CHROME),
        pdfinfo=str(PDFINFO),
        pdftotext=str(PDFTOTEXT),
    )

    assert 8 <= result.pages <= 12
    rendered_text = subprocess.run(
        [str(PDFTOTEXT), "-layout", str(result.output_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "𝛾" in rendered_text
    assert "∣" in rendered_text


@pytest.mark.skipif(
    not BOUND_RENDER_TOOLS_AVAILABLE,
    reason="Pandoc, Chrome, and Poppler are required for bound-paper rendering",
)
def test_bound_yau_manuscript_is_exactly_four_pages(tmp_path: Path) -> None:
    repository = SCRIPT.parents[1]

    result = paper_pdf.render_paper(
        profile="yau",
        input_path=repository / "docs/paper/yau_award_4page.md",
        output_path=tmp_path / "yau.pdf",
        references_path=repository / "docs/paper/references.json",
        pandoc=str(PANDOC),
        chrome=str(CHROME),
        pdfinfo=str(PDFINFO),
        pdftotext=str(PDFTOTEXT),
        expected_pages=4,
    )

    assert result.pages == 4
    rendered_text = subprocess.run(
        [str(PDFTOTEXT), "-layout", str(result.output_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for required in (
        "Machine integrity summary",
        "H1: A P convergence",
        "0 qualifying providers",
        "metric_registry.json",
    ):
        assert required in rendered_text
    assert "H1_P_A_CORRECT_CONVERGENCE_MATCHED" not in rendered_text


def test_makefile_offers_full_and_yau_pdf_targets() -> None:
    result = subprocess.run(
        ["make", "-n", "paper-pdf-main", "paper-pdf-yau"],
        cwd=SCRIPT.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "output/pdf/main.pdf" in result.stdout
    assert "output/pdf/yau_award_4page.pdf" in result.stdout
    assert "--expected-pages 4" in result.stdout

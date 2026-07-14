from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from scripts import render_paper_pdf as paper_pdf


SCRIPT = Path(__file__).parents[1] / "scripts/render_paper_pdf.py"


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
        "--expected-pages",
    ):
        assert option in result.stdout


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

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = [str(part) for part in command]
        calls.append(command)
        executable = command[0]
        if executable == "pandoc-bin":
            assert kwargs["input"] == "# Paper\n\nEvidence [@example2026].\n"
            assert kwargs["text"] is True
            assert "--standalone" in command
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

    monkeypatch.setattr(paper_pdf.subprocess, "run", fake_run)
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

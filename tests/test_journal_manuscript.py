from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
MANUSCRIPT = ROOT / "docs/paper/journal_main.md"
COVER_LETTER = ROOT / "docs/paper/journal_cover_letter.md"
REVIEW_MATRIX = ROOT / "docs/paper/reviewer_objections_matrix.md"
REFERENCES = ROOT / "docs/paper/references.json"
RENDER_SCRIPT = ROOT / "scripts/render_paper_pdf.py"
PANDOC = Path(shutil.which("pandoc") or "/opt/homebrew/bin/pandoc")
CHROME = Path(
    os.environ.get(
        "PAPER_CHROME",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
)
PDFINFO = Path(shutil.which("pdfinfo") or "/opt/homebrew/bin/pdfinfo")
PDFTOTEXT = Path(shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext")
RENDER_TOOLS_AVAILABLE = all(
    path.is_file() for path in (PANDOC, CHROME, PDFINFO, PDFTOTEXT)
)


def _text(path: Path) -> str:
    assert path.is_file(), f"missing journal deliverable: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def _slot(text: str, name: str) -> str:
    pattern = re.compile(
        rf"<!-- BEGIN RESULT SLOT: {re.escape(name)} -->"
        rf"(.*?)"
        rf"<!-- END RESULT SLOT: {re.escape(name)} -->",
        flags=re.DOTALL,
    )
    matches = pattern.findall(text)
    assert len(matches) == 1, f"expected exactly one {name} result slot"
    return matches[0]


def test_journal_manuscript_has_three_titles_structured_abstract_and_imrad() -> None:
    text = _text(MANUSCRIPT)

    candidates = re.findall(r"^TITLE_CANDIDATE_[123]: (.+)$", text, re.MULTILINE)
    assert len(candidates) == 3
    selected = re.search(r"^SELECTED_TITLE: (.+)$", text, re.MULTILINE)
    assert selected is not None
    assert selected.group(1) in candidates
    assert text.startswith("<!-- JOURNAL TITLE OPTIONS\n")
    assert f"# {selected.group(1)}" in text

    abstract = text.split("## Structured Abstract", 1)[1].split("## Keywords", 1)[0]
    for label in ("Background", "Objective", "Methods", "Results", "Conclusions"):
        assert f"**{label}:**" in abstract

    required_sections = (
        "## 1. Introduction",
        "## 2. Related Work",
        "## 3. Methods",
        "## 4. Results",
        "## 5. Discussion",
        "## 6. Limitations",
        "## 7. Conclusion",
        "## Declarations",
    )
    offsets = [text.index(section) for section in required_sections]
    assert offsets == sorted(offsets)
    assert len(re.findall(r"\b[\w'-]+\b", text)) >= 4_500


def test_journal_results_use_bound_h1_h4_evidence_and_same_support_pairs() -> None:
    text = _text(MANUSCRIPT)
    flat = " ".join(text.split())

    for statement in (
        "H1 was partially supported",
        "H2 was not supported",
        "H3 was supported",
        "H4 was supported",
        "1,131/1,350 (83.8%)",
        "168/1,350 (12.4%)",
        "1,043/1,150 (90.7%)",
        "148/1,150 (12.9%)",
        "selector-stopping mismatch",
        "post-hoc",
    ):
        assert statement in flat

    assert "83.8% vs 12.9%" not in text
    assert "1131/1350" not in text
    assert "1043/1150" not in text


def test_persona_v2_and_p2_are_explicit_empty_machine_slots() -> None:
    text = _text(MANUSCRIPT)
    flat = " ".join(text.split())

    persona_slot = _slot(text, "PERSONA_V2_DUAL")
    p2_slot = _slot(text, "P2_ILLUSTRATIVE")
    for slot in (persona_slot, p2_slot):
        assert "No outcome estimate is reported in this slot." in slot
        assert "%" not in slot
        assert re.search(r"\d", slot) is None

    for statement in (
        "independent response-channel stress test",
        "controlled manipulation",
        "blind response robustness",
        "50 persona clusters",
        "provider as a repeated measure",
        "text-only",
        "6 of 100",
        "13 nodes and 68 trusted segments",
        "illustrative",
    ):
        assert statement in flat


def test_submission_surface_excludes_blacklist_internal_state_and_stale_h5() -> None:
    text = _text(MANUSCRIPT)
    prohibited = (
        r"\b600 (?:learners|students)\b",
        r"\breal student distribution\b",
        r"\bteacher gold\b",
        r"\bhuman[- ]validated\b",
        r"\bexpert[- ]validated\b",
        r"\blearning trajector(?:y|ies)\b",
        r"\bfour-state persona\b",
        r"\bfirst[- ]ever\b",
        r"\bpractical non-identifiability\b",
        r"\bcausal state\b",
        r"\bcross-node(?: prerequisite)? evidence\b",
        r"\b(?:partially_supported|not_supported|excluded_pre_outcome)\b",
        r"\bmanuscript status\b",
        r"\b(?:submitted|under review|accepted|published)\b",
        r"/Users/",
        r"/tmp/",
        r"file://",
        r"\b23\.4\b",
        r"\bJCR Q1\b",
        r"fig-manipulation-checks",
        r"fig-provider-agreement",
    )
    for pattern in prohibited:
        assert re.search(pattern, text, re.IGNORECASE) is None, pattern


def test_claim_boundaries_limit_the_study_to_model_defined_simulation() -> None:
    text = _text(MANUSCRIPT)
    flat = " ".join(text.split())

    for statement in (
        "simulation-only",
        "model-defined diagnostic states",
        "does not estimate learning gains",
        "does not establish educational efficacy",
        "does not establish human behavioral validity",
        "budget-limited confident-convergence failure",
    ):
        assert statement in flat


def test_manuscript_has_exactly_ten_limitations_and_ai_disclosures() -> None:
    text = _text(MANUSCRIPT)
    limitations = text.split("## 6. Limitations", 1)[1].split("## 7. Conclusion", 1)[0]

    assert len(re.findall(r"^\d+\. \*\*", limitations, re.MULTILINE)) == 10
    assert "### 3.8 AI-assisted research workflow" in text
    declaration = "### Declaration of Generative AI and AI-assisted Technologies"
    assert declaration in text
    assert text.index(declaration) > text.index("## Declarations")
    assert text.rstrip().endswith("## References")
    for role in (
        "drafting and revising prose",
        "software implementation and testing",
        "simulation execution",
        "figure generation",
        "adversarial manuscript review",
        "AI systems were not treated as authors",
    ):
        assert role in text


def test_journal_citations_resolve_to_verified_registry() -> None:
    text = _text(MANUSCRIPT)
    payload = json.loads(REFERENCES.read_text(encoding="utf-8"))
    known = {row["id"] for row in payload["references"]}
    cited = set(re.findall(r"@([A-Za-z0-9_-]+)", text))

    assert cited
    assert cited <= known
    assert {
        "lord1971",
        "barrada2010",
        "corbett-anderson1994",
        "lu-wang2024",
        "scarlatos2026",
    } <= cited


def test_cover_letter_and_reviewer_matrix_are_honest_drafts() -> None:
    cover = _text(COVER_LETTER)
    matrix = _text(REVIEW_MATRIX)

    assert cover.startswith("# Cover Letter Draft")
    assert "[TARGET JOURNAL TO BE SELECTED]" in cover
    assert "does not represent an external submission" in cover
    assert "No human participants were enrolled" in cover
    assert "Persona-v2 and prescription outcomes are not claimed" in cover

    assert matrix.startswith("# Anticipated Reviewer Objections and Response Matrix")
    assert "| ID | Anticipated objection |" in matrix
    rows = re.findall(r"^\| R\d{2} \|", matrix, re.MULTILINE)
    assert len(rows) >= 12
    assert "Evidence required before insertion" in matrix


@pytest.mark.skipif(not RENDER_TOOLS_AVAILABLE, reason="journal render tools unavailable")
def test_journal_manuscript_renders_to_a4_submission_surface(tmp_path: Path) -> None:
    output = tmp_path / "journal-main.pdf"
    result = subprocess.run(
        [
            sys.executable,
            str(RENDER_SCRIPT),
            "--profile",
            "main",
            "--input",
            str(MANUSCRIPT),
            "--output",
            str(output),
            "--references",
            str(REFERENCES),
            "--pandoc",
            str(PANDOC),
            "--chrome",
            str(CHROME),
            "--pdfinfo",
            str(PDFINFO),
            "--pdftotext",
            str(PDFTOTEXT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert output.is_file() and output.stat().st_size > 0
    metadata = subprocess.run(
        [str(PDFINFO), str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)$", metadata, re.MULTILINE).group(1))
    assert 8 <= pages <= 12
    assert re.search(r"^Page size:\s+59[45](?:\.\d+)? x 84[12](?:\.\d+)? pts", metadata, re.MULTILINE)

    rendered = subprocess.run(
        [str(PDFTOTEXT), "-layout", str(output), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for visible in (
        "When Accurate Terminal Decisions Do Not Converge",
        "Structured Abstract",
        "Table 1. Same-support terminal and confident-convergence estimates.",
        "Declaration of Generative AI and AI-assisted Technologies",
        "References",
    ):
        assert visible in rendered
    for hidden in (
        "TITLE_CANDIDATE_1",
        "BEGIN RESULT SLOT",
        "END RESULT SLOT",
        "/Users/",
        "/tmp/",
        "PENDING",
    ):
        assert hidden not in rendered
    assert re.search(r"\[[^\]]*@[-A-Za-z0-9]", rendered) is None

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
REFERENCES_PATH = ROOT / "docs/paper/references.json"
AUDIT_PATH = ROOT / "experiments/config/paper_reference_audit_v1.json"
JOURNAL_PATH = ROOT / "docs/paper/journal_main.md"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _journal_citation_loci() -> list[tuple[str, int]]:
    text = JOURNAL_PATH.read_text(encoding="utf-8")
    loci: list[tuple[str, int]] = []
    for group in re.finditer(r"\[([^\[\]]*?@[^\[\]]*?)\]", text, flags=re.DOTALL):
        for key in re.finditer(r"@([A-Za-z0-9_-]+)", group.group(1)):
            offset = group.start(1) + key.start()
            loci.append((key.group(1), text.count("\n", 0, offset) + 1))
    return loci


def test_reference_registry_is_bound_to_complete_existence_audit() -> None:
    references = _json(REFERENCES_PATH)
    audit = _json(AUDIT_PATH)
    audit_bytes = AUDIT_PATH.read_bytes()

    assert references["audit_artifact"] == "experiments/config/paper_reference_audit_v1.json"
    assert references["audit_sha256"] == hashlib.sha256(audit_bytes).hexdigest()
    assert "independently verified" in str(references["verification_note"])
    assert references["verified_at"] == str(audit["timestamp"])[:10]
    assert audit["schema_version"] == "yher.citation-audit.v1"
    assert audit["critical_findings"] == []

    counts = audit["counts"]
    assert isinstance(counts, dict)
    assert counts["reference_records"] == 19
    assert counts["existence_verified"] == 19
    assert counts["doi_records"] == counts["doi_registered"] == 17
    assert counts["fabricated"] == counts["conflated"] == 0
    assert counts["uncited_bibliography_items"] == 0
    assert counts["unresolved_citation_keys"] == 0
    assert counts["duplicate_ids"] == 0

    reference_rows = references["references"]
    audit_rows = audit["records"]
    assert isinstance(reference_rows, list)
    assert isinstance(audit_rows, list)
    assert {row["id"] for row in reference_rows} == {row["id"] for row in audit_rows}
    references_by_id = {row["id"]: row for row in reference_rows}
    audit_by_id = {row["id"]: row for row in audit_rows}
    for reference_id, reference in references_by_id.items():
        identifier = audit_by_id[reference_id]["identifier"]
        assert reference.get("doi") == identifier.get("doi")
        if identifier.get("doi") is None:
            assert reference["url"] == identifier["url"]
    assert all(row["existence"] == "verified" for row in audit_rows)
    doi_rows = [row for row in audit_rows if row["identifier"].get("doi")]
    assert len(doi_rows) == 17
    assert all(row["identifier"]["registered"] is True for row in doi_rows)
    assert all(
        str(row["identifier"]["resolution"]).startswith("resolved")
        for row in doi_rows
    )

    manuscript_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/paper/main.md",
            "docs/paper/yau_award_4page.md",
            "docs/paper/journal_main.md",
        )
    )
    cited = set(re.findall(r"@([A-Za-z0-9_-]+)", manuscript_text))
    assert cited == {row["id"] for row in reference_rows}


def test_journal_citation_audit_covers_every_actual_key_locus() -> None:
    audit = _json(AUDIT_PATH)
    rows = audit["records"]
    assert isinstance(rows, list)

    actual_loci = Counter(_journal_citation_loci())
    journal_uses = [
        (row["id"], use)
        for row in rows
        for use in row["citation_use"]
        if use["file"] == "journal_main.md"
    ]
    audited_loci = Counter(
        (reference_id, use.get("line"))
        for reference_id, use in journal_uses
        if use["status"] == "supported"
    )

    assert actual_loci == audited_loci
    assert len(actual_loci) == len(journal_uses) == 20
    assert len({reference_id for reference_id, _ in actual_loci}) == 19

    statuses = Counter(use["status"] for _, use in journal_uses)
    assert statuses == Counter({"supported": 20})
    counts = audit["counts"]
    assert isinstance(counts, dict)
    assert counts["unique_citation_keys"] == len(
        {reference_id for reference_id, _ in actual_loci}
    )
    assert counts["citation_key_loci"] == len(journal_uses)
    assert counts["claim_supported_loci"] == statuses["supported"]
    assert counts["context_only_or_misplaced_loci"] == statuses[
        "context_only_misplaced"
    ]
    assert counts["contradicted_claim_loci"] == statuses["contradicted"]


def test_reference_registry_uses_audited_exact_metadata_corrections() -> None:
    rows = {
        row["id"]: row
        for row in _json(REFERENCES_PATH)["references"]
    }

    assert rows["barrada2010"]["authors"] == [
        "Juan Ramón Barrada",
        "Julio Olea",
        "Vicente Ponsoda",
        "Francisco José Abad",
    ]
    assert rows["pelanek2017"]["authors"] == ["Radek Pelánek"]
    assert "Bernhard Schölkopf" in rows["tabibian2019"]["authors"]
    assert rows["fsrs-repository"]["title"] == (
        "Free Spaced Repetition Scheduling Algorithm"
    )
    assert rows["anki-fsrs-manual"]["title"] == "Deck Options - Anki Manual"
    assert rows["anki-fsrs-manual"]["metadata_note"] == (
        "Official Anki documentation, section 'FSRS'; not treated as a "
        "peer-reviewed paper."
    )
    assert rows["lu-wang2024"]["container_title"] == (
        "Proceedings of the Eleventh ACM Conference on Learning @ Scale"
    )
    assert rows["jin2025"]["container_title"] == (
        "Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems"
    )
    assert rows["wu2025"]["container_title"].endswith(
        "(Volume 1: Long Papers)"
    )
    assert rows["scarlatos2026"]["container_title"].endswith(
        "(Volume 1: Long Papers)"
    )


def test_yau_prior_art_citations_do_not_source_internal_h5_design() -> None:
    text = (ROOT / "docs/paper/yau_award_4page.md").read_text(encoding="utf-8")
    prior_art = re.search(
        r"Prior\s+LLM-simulated-student work.*?\[@lu-wang2024;.*?@scarlatos2026\]",
        text,
        flags=re.DOTALL,
    )
    internal = re.search(
        r"In\s+this\s+project, H5 is secondary and manipulation-gated;.*?\.\n",
        text,
        flags=re.DOTALL,
    )

    assert prior_art is not None
    assert internal is not None
    assert "@" not in internal.group(0)

"""Atomic, hash-bound finalization for the journal manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from . import journal_binder


class FinalizationError(ValueError):
    """Raised when a manuscript cannot satisfy the publication binding."""


SLOT_FIELDS = {
    "PROGRAMMATIC_ABSTRACT_RESULTS": "programmatic_abstract_results_markdown",
    "BOUND_ABSTRACT_RESULTS": "bound_abstract_results_markdown",
    "EXECUTION_INTEGRITY": "execution_integrity_markdown",
    "HYPOTHESIS_DECISIONS": "hypothesis_decisions_markdown",
    "PRIMARY_H1_H4_RESULTS": "primary_h1_h4_results_markdown",
    "SAME_SUPPORT_CONVERGENCE": "same_support_convergence_markdown",
    "PERSONA_V2_DUAL": "persona_v2_markdown",
    "P2_ILLUSTRATIVE": "p2_markdown",
}
SLOT_PATTERN = re.compile(
    r"<!-- BEGIN RESULT SLOT: (?P<name>[A-Z0-9_]+) -->"
    r"(?P<body>.*?)"
    r"<!-- END RESULT SLOT: (?P=name) -->",
    flags=re.DOTALL,
)
MANUAL_RESULT_NUMBER = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*%|\b\d[\d,]*/\d[\d,]*\b|\b95\s*%?\s*CI\b|\bn\s*=\s*\d+)",
    flags=re.IGNORECASE,
)
PENDING_PATTERNS = (
    r"\bPENDING\b",
    r"No outcome estimate is reported in this slot",
    r"Insert only machine-bound",
    r"BEGIN RESULT SLOT",
    r"END RESULT SLOT",
    r"Formal Persona-v2 W3 artifacts are pending",
    r"P2 is not bound in this binder generation",
    r"results? (?:are|is) absent by design until",
    r"result slot remains empty until",
    r"until a bound analysis artifact is inserted",
    r"no Persona-v2 figure",
    r"no Persona-v2 outcome enters this version",
    r"Persona-v2 response-channel results and the P2 prescription illustration are absent",
    r"P2 (?:component )?will (?:illustrate|compare|select)",
    r"illustrative P2 result slot remains empty",
    r"(?:Persona-v2|P2).{0,100}\b(?:will|reserved|absent by design|slot remains empty)\b",
    r"\b(?:will|reserved|absent by design|slot remains empty)\b.{0,100}(?:Persona-v2|P2)",
)
BLACKLIST_PATTERNS = (
    r"\b600 (?:learners|students)\b",
    r"\breal student distribution\b",
    r"\bteacher gold\b",
    r"\bhuman gold\b",
    r"\bhuman[- ]validated\b",
    r"\bexpert[- ]validated\b",
    r"\blearning trajector(?:y|ies)\b",
    r"\bfour-state persona\b",
    r"\bfirst[- ]ever\b",
    r"\b23\.4\b",
    r"\bJCR Q1\b",
    r"/Users/",
    r"/tmp/",
    r"file://",
)
MAX_STRUCTURED_ABSTRACT_WORDS = 300
STRUCTURED_ABSTRACT_PATTERN = re.compile(
    r"^## Structured Abstract[ \t]*\n(?P<body>.*?)(?=^## [^\n]+$)",
    flags=re.MULTILINE | re.DOTALL,
)
ABSTRACT_WORD_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[.,/][A-Za-z0-9]+)*(?:[-'][A-Za-z0-9]+)*"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON object")
    return value, raw


def _artifact_index(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise FinalizationError(f"{label} has no artifact list")
    index: dict[str, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping) or set(value) != {"filename", "bytes", "sha256"}:
            raise FinalizationError(f"{label} artifact row is invalid")
        filename = value.get("filename")
        if not isinstance(filename, str) or not filename or filename in index:
            raise FinalizationError(f"{label} artifact filename is invalid")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise FinalizationError(f"{label} artifact path is unsafe")
        path = root / relative
        if not path.is_file():
            raise FinalizationError(f"{label} artifact is missing: {filename}")
        data = path.read_bytes()
        if value.get("bytes") != len(data) or value.get("sha256") != _sha(data):
            raise FinalizationError(f"{label} artifact bytes drifted: {filename}")
        index[filename] = dict(value)
    expected_set = _sha(_canonical(rows))
    if manifest.get("generation_set_sha256") not in (None, expected_set):
        raise FinalizationError(f"{label} artifact-set hash drifted")
    if manifest.get("artifact_set_sha256") not in (None, expected_set):
        raise FinalizationError(f"{label} artifact-set hash drifted")
    return index


def _load_binder_generation(
    path: Path | str,
    *,
    expected_generation_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], bytes, bytes, bytes]:
    try:
        generation = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise FinalizationError(f"binder generation cannot resolve: {path}") from exc
    if not generation.is_dir() or generation.name != expected_generation_id:
        raise FinalizationError("binder generation ID differs from the frozen expectation")
    manifest, manifest_bytes = _load_json(
        generation / "artifact_manifest.json", label="binder artifact manifest"
    )
    if (
        manifest.get("schema_version") != "yher.journal_support_binder.output.v2"
        or manifest.get("generation_id") != generation.name
    ):
        raise FinalizationError("binder generation envelope drifted")
    artifacts = _artifact_index(manifest, root=generation, label="binder generation")
    if set(artifacts) != {"journal_binder.json", "manuscript_slots.json"}:
        raise FinalizationError("binder generation artifact set is incomplete")
    if manifest.get("generation_set_sha256") != _sha(_canonical(manifest["artifacts"])):
        raise FinalizationError("binder generation artifact-set hash drifted")
    expected_generation_id = _sha(
        _canonical(
            {
                "journal_binder.json": artifacts["journal_binder.json"]["sha256"],
                "manuscript_slots.json": artifacts["manuscript_slots.json"]["sha256"],
            }
        )
    )[:24]
    if generation.name != expected_generation_id:
        raise FinalizationError("binder generation ID does not match its content")
    binder, binder_bytes = _load_json(
        generation / "journal_binder.json", label="journal binder"
    )
    slots, slots_bytes = _load_json(
        generation / "manuscript_slots.json", label="manuscript slots"
    )
    if binder.get("schema_version") != "yher.journal_support_binder.v1":
        raise FinalizationError("journal binder schema drifted")
    if (
        binder.get("status") != "bound"
        or not isinstance(binder.get("persona_v2"), Mapping)
        or binder["persona_v2"].get("status") != "bound_formal_w3"
        or not isinstance(binder.get("p2"), Mapping)
        or binder["p2"].get("status") != "bound"
    ):
        raise FinalizationError("journal finalization requires a fully bound Persona-v2 and P2 binder")
    recomputed_slots = journal_binder.render_manuscript_slots(binder)
    if slots != recomputed_slots:
        raise FinalizationError("manuscript slots are stale relative to the journal binder")
    return (
        generation,
        manifest,
        binder,
        slots,
        manifest_bytes,
        binder_bytes,
        slots_bytes,
    )


def _iter_source_bindings(value: Any) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if {"source_path", "sha256", "bytes"}.issubset(value):
            output.append(value)
        for child in value.values():
            output.extend(_iter_source_bindings(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_iter_source_bindings(child))
    return output


def _verify_binder_sources(binder: Mapping[str, Any]) -> None:
    seen: set[tuple[str, str]] = set()
    bindings = _iter_source_bindings(binder)
    if not bindings:
        raise FinalizationError("binder has no recheckable source bindings")
    for value in bindings:
        source = value.get("source_path")
        digest = value.get("sha256")
        size = value.get("bytes")
        if not isinstance(source, str) or not isinstance(digest, str):
            raise FinalizationError("binder source binding is malformed")
        key = (source, digest)
        if key in seen:
            continue
        seen.add(key)
        path = Path(source)
        if not path.is_file():
            raise FinalizationError(f"stale binder source is missing: {source}")
        data = path.read_bytes()
        if size != len(data) or digest != _sha(data):
            raise FinalizationError(f"stale binder source bytes drifted: {source}")


def _replace_slots(template: str, slots: Mapping[str, Any]) -> str:
    matches = list(SLOT_PATTERN.finditer(template))
    names = [match.group("name") for match in matches]
    if sorted(names) != sorted(SLOT_FIELDS) or len(names) != len(set(names)):
        raise FinalizationError(
            "journal template does not contain the exact bound result-slot roster"
        )
    replacements: dict[str, str] = {}
    for match in matches:
        name = match.group("name")
        body = match.group("body")
        if MANUAL_RESULT_NUMBER.search(body):
            raise FinalizationError(f"manual result number found inside template slot: {name}")
        field = SLOT_FIELDS[name]
        replacement = slots.get(field)
        if not isinstance(replacement, str) or not replacement.strip():
            raise FinalizationError(f"bound manuscript slot is empty: {name}")
        replacements[name] = replacement.strip()

    def replace(match: re.Match[str]) -> str:
        return replacements[match.group("name")]

    final, count = SLOT_PATTERN.subn(replace, template)
    if count != len(SLOT_FIELDS):
        raise FinalizationError("journal slot replacement count drifted")
    return final.rstrip() + "\n"


def _load_reference_ids(path: Path) -> tuple[set[str], str]:
    payload, raw = _load_json(path, label="verified reference registry")
    if payload.get("schema_version") != "yher.verified-references.v1":
        raise FinalizationError("verified reference registry schema drifted")
    rows = payload.get("references")
    if not isinstance(rows, list):
        raise FinalizationError("verified reference registry has no reference list")
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not row["id"]:
            raise FinalizationError("verified reference registry row is invalid")
        identifiers.append(str(row["id"]))
    if len(identifiers) != len(set(identifiers)):
        raise FinalizationError("verified reference registry repeats an ID")
    return set(identifiers), _sha(raw)


def audit_manuscript(text: str, *, references_path: Path | str) -> None:
    for pattern in PENDING_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise FinalizationError(f"pending manuscript marker remains: {pattern}")
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise FinalizationError(f"manuscript wording blacklist hit: {pattern}")
    known, _ = _load_reference_ids(Path(references_path))
    cited = set(re.findall(r"@([A-Za-z0-9_-]+)", text))
    unknown = cited - known
    if unknown:
        raise FinalizationError(
            "journal manuscript cites unknown reference IDs: " + ", ".join(sorted(unknown))
        )


def structured_abstract_word_count(text: str) -> int:
    matches = list(STRUCTURED_ABSTRACT_PATTERN.finditer(text))
    if len(matches) != 1:
        raise FinalizationError(
            "final journal manuscript must contain exactly one Structured Abstract"
        )
    return len(ABSTRACT_WORD_PATTERN.findall(matches[0].group("body")))


def _enforce_structured_abstract_word_limit(text: str) -> int:
    count = structured_abstract_word_count(text)
    if count > MAX_STRUCTURED_ABSTRACT_WORDS:
        raise FinalizationError(
            "Structured Abstract word count "
            f"{count} exceeds the {MAX_STRUCTURED_ABSTRACT_WORDS}-word maximum"
        )
    return count


def _copy_plan(binder: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    programmatic_assets = binder.get("programmatic_publication_assets")
    if not isinstance(programmatic_assets, Mapping) or set(
        programmatic_assets
    ) != set(journal_binder.PROGRAMMATIC_PUBLICATION_ASSETS):
        raise FinalizationError("programmatic publication figure roster is incomplete")
    persona_assets = binder["persona_v2"]["publication_assets"]
    p2_assets = binder["p2"]["publication_assets"]
    main = persona_assets["main_persona_composite_sources"]
    supplement = persona_assets["supplement_figures"]
    plan: dict[str, Mapping[str, Any]] = {
        str(relative): descriptor
        for relative, descriptor in programmatic_assets.items()
    }
    plan.update({
        f"assets/persona_v2/{name}.png": descriptor
        for name, descriptor in main.items()
    })
    for relative, descriptor in persona_assets["tables"].items():
        plan[f"assets/supplement/persona_v2/{Path(relative).name}"] = descriptor
    for relative, descriptor in supplement.items():
        plan[f"assets/supplement/persona_v2/{Path(relative).name}"] = descriptor
    plan["assets/supplement/p2/p2_supply_bound_illustration.png"] = p2_assets[
        "supplement_figure_png"
    ]
    plan["assets/supplement/p2/p2_supply_bound_illustration.svg"] = p2_assets[
        "supplement_figure_svg"
    ]
    return plan


def _local_asset_references(text: str) -> set[str]:
    references = set(
        re.findall(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)", text)
    )
    references.update(re.findall(r'<img\s+[^>]*src="([^"]+)"', text))
    return {
        value
        for value in references
        if not re.match(r"^(?:https?:|data:|#)", value, flags=re.IGNORECASE)
    }


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finalize_manuscript(
    *,
    template_path: Path | str,
    binder_generation: Path | str,
    references_path: Path | str,
    output_dir: Path | str,
    expected_template_sha256: str,
    expected_binder_generation_id: str,
) -> dict[str, Any]:
    template = Path(template_path)
    references = Path(references_path)
    if not template.is_file():
        raise FinalizationError(f"journal template is missing: {template}")
    template_bytes = template.read_bytes()
    if _sha(template_bytes) != expected_template_sha256:
        raise FinalizationError("journal template bytes differ from the frozen SHA-256")
    try:
        template_text = template_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinalizationError("journal template is not UTF-8") from exc
    (
        binder_path,
        binder_manifest,
        binder,
        slots,
        binder_manifest_bytes,
        binder_bytes,
        slots_bytes,
    ) = _load_binder_generation(
        binder_generation, expected_generation_id=expected_binder_generation_id
    )
    _verify_binder_sources(binder)
    final_text = _replace_slots(template_text, slots)
    audit_manuscript(final_text, references_path=references)
    abstract_word_count = _enforce_structured_abstract_word_limit(final_text)
    manuscript_bytes = final_text.encode("utf-8")
    _, references_sha = _load_reference_ids(references)

    asset_plan = _copy_plan(binder)
    asset_bytes: dict[str, bytes] = {}
    for relative, descriptor in asset_plan.items():
        source = Path(str(descriptor["source_path"]))
        data = source.read_bytes()
        if descriptor.get("sha256") != _sha(data) or descriptor.get("bytes") != len(data):
            raise FinalizationError(f"stale binder source bytes drifted: {source}")
        asset_bytes[relative] = data
    missing_assets = _local_asset_references(final_text) - set(asset_bytes)
    if missing_assets:
        raise FinalizationError(
            "final manuscript references an unbound local asset: "
            + ", ".join(sorted(missing_assets))
        )
    artifacts = [
        {
            "filename": "journal_main.md",
            "bytes": len(manuscript_bytes),
            "sha256": _sha(manuscript_bytes),
        },
        *[
            {
                "filename": relative,
                "bytes": len(data),
                "sha256": _sha(data),
            }
            for relative, data in sorted(asset_bytes.items())
        ],
    ]
    binding = {
        "template_sha256": expected_template_sha256,
        "binder_generation_id": binder_path.name,
        "binder_manifest_sha256": _sha(binder_manifest_bytes),
        "journal_binder_sha256": _sha(binder_bytes),
        "manuscript_slots_sha256": _sha(slots_bytes),
        "slot_content_sha256": slots["content_sha256"],
        "references_sha256": references_sha,
        "final_manuscript_sha256": _sha(manuscript_bytes),
        "structured_abstract_word_count": abstract_word_count,
        "artifact_set_sha256": _sha(_canonical(artifacts)),
    }
    generation_id = _sha(_canonical(binding))[:24]
    manifest = {
        "schema_version": "yher.journal_manuscript.finalization.v1",
        "generation_id": generation_id,
        "template": {
            "source_path": str(template.resolve()),
            "sha256": expected_template_sha256,
        },
        "binder": {
            "generation_id": binder_path.name,
            "source_path": str(binder_path),
            "artifact_manifest_sha256": _sha(binder_manifest_bytes),
            "journal_binder_sha256": _sha(binder_bytes),
        },
        "slots": {
            "manuscript_slots_sha256": _sha(slots_bytes),
            "content_sha256": slots["content_sha256"],
        },
        "references": {
            "source_path": str(references.resolve()),
            "sha256": references_sha,
        },
        "final_manuscript": {
            "filename": "journal_main.md",
            "bytes": len(manuscript_bytes),
            "sha256": _sha(manuscript_bytes),
        },
        "structured_abstract": {
            "word_count": abstract_word_count,
            "maximum_words": MAX_STRUCTURED_ABSTRACT_WORDS,
        },
        "artifact_set_sha256": _sha(_canonical(artifacts)),
        "artifacts": artifacts,
    }
    root = Path(output_dir)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    final_dir = generations / generation_id
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    link_temp = root / f".current.{generation_id}.{os.getpid()}.tmp"
    try:
        _write_file(staging / "journal_main.md", manuscript_bytes)
        for relative, data in asset_bytes.items():
            _write_file(staging / relative, data)
        _write_file(staging / "finalization_manifest.json", _pretty(manifest))
        _fsync_directory(staging)
        if final_dir.exists():
            existing = final_dir / "finalization_manifest.json"
            if not existing.is_file() or existing.read_bytes() != _pretty(manifest):
                raise FinalizationError("existing journal finalization generation drifted")
            shutil.rmtree(staging)
        else:
            os.replace(staging, final_dir)
            _fsync_directory(generations)
        if link_temp.exists() or link_temp.is_symlink():
            link_temp.unlink()
        os.symlink(Path("generations") / generation_id, link_temp)
        os.replace(link_temp, root / "current")
        _fsync_directory(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if link_temp.exists() or link_temp.is_symlink():
            link_temp.unlink()
        raise
    return manifest


def verify_finalized_generation(
    generation_path: Path | str,
    *,
    references_path: Path | str,
    expected_template_sha256: str | None = None,
    expected_binder_generation_id: str | None = None,
) -> dict[str, Any]:
    try:
        generation = Path(generation_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise FinalizationError("finalized journal generation cannot resolve") from exc
    manifest, _ = _load_json(
        generation / "finalization_manifest.json", label="finalization manifest"
    )
    if (
        manifest.get("schema_version") != "yher.journal_manuscript.finalization.v1"
        or manifest.get("generation_id") != generation.name
    ):
        raise FinalizationError("finalization manifest envelope drifted")
    artifacts = _artifact_index(manifest, root=generation, label="final manuscript")
    artifact_set_sha256 = _sha(_canonical(manifest.get("artifacts")))
    if manifest.get("artifact_set_sha256") != artifact_set_sha256:
        raise FinalizationError("final manuscript artifact-set hash drifted")
    final = manifest.get("final_manuscript")
    if not isinstance(final, Mapping) or artifacts.get("journal_main.md") != final:
        raise FinalizationError("final manuscript binding is incomplete")
    manuscript_path = generation / "journal_main.md"
    manuscript_bytes = manuscript_path.read_bytes()
    if final.get("sha256") != _sha(manuscript_bytes) or final.get("bytes") != len(manuscript_bytes):
        raise FinalizationError("final manuscript bytes drifted")
    manuscript_text = manuscript_bytes.decode("utf-8")
    audit_manuscript(manuscript_text, references_path=references_path)
    abstract_word_count = _enforce_structured_abstract_word_limit(manuscript_text)
    abstract_binding = manifest.get("structured_abstract")
    if abstract_binding != {
        "word_count": abstract_word_count,
        "maximum_words": MAX_STRUCTURED_ABSTRACT_WORDS,
    }:
        raise FinalizationError("Structured Abstract word-count binding drifted")
    reference_binding = manifest.get("references")
    _, current_reference_sha = _load_reference_ids(Path(references_path))
    if not isinstance(reference_binding, Mapping) or reference_binding.get("sha256") != current_reference_sha:
        raise FinalizationError("verified reference registry drifted after finalization")
    template_binding = manifest.get("template")
    binder_binding = manifest.get("binder")
    slot_binding = manifest.get("slots")
    if not all(
        isinstance(value, Mapping)
        for value in (template_binding, binder_binding, slot_binding)
    ):
        raise FinalizationError("final manuscript generation binding is incomplete")
    if (
        expected_template_sha256 is not None
        and template_binding.get("sha256") != expected_template_sha256
    ):
        raise FinalizationError("final manuscript template differs from expected SHA-256")
    if (
        expected_binder_generation_id is not None
        and binder_binding.get("generation_id") != expected_binder_generation_id
    ):
        raise FinalizationError("final manuscript binder generation differs from expectation")
    binding = {
        "template_sha256": template_binding.get("sha256"),
        "binder_generation_id": binder_binding.get("generation_id"),
        "binder_manifest_sha256": binder_binding.get("artifact_manifest_sha256"),
        "journal_binder_sha256": binder_binding.get("journal_binder_sha256"),
        "manuscript_slots_sha256": slot_binding.get("manuscript_slots_sha256"),
        "slot_content_sha256": slot_binding.get("content_sha256"),
        "references_sha256": reference_binding.get("sha256"),
        "final_manuscript_sha256": final.get("sha256"),
        "structured_abstract_word_count": abstract_word_count,
        "artifact_set_sha256": artifact_set_sha256,
    }
    if generation.name != _sha(_canonical(binding))[:24]:
        raise FinalizationError("final manuscript generation ID does not match its content")
    return manifest


def _validate_render_receipt(
    receipt_path: Path,
    *,
    generation: Path,
    references: Path,
    pdf: Path,
) -> tuple[dict[str, Any], bytes]:
    receipt, raw = _load_json(receipt_path, label="journal PDF render receipt")
    advertised = receipt.get("render_receipt_sha256")
    payload = dict(receipt)
    payload.pop("render_receipt_sha256", None)
    if (
        receipt.get("schema_version") != "yher.paper_pdf.render_receipt.v1"
        or advertised != _sha(_canonical(payload))
    ):
        raise FinalizationError("journal PDF render receipt self-hash drifted")
    if receipt.get("profile") != "main":
        raise FinalizationError("journal PDF render receipt profile is not main")
    input_binding = receipt.get("input")
    reference_binding = receipt.get("references")
    pdf_binding = receipt.get("pdf")
    renderer = receipt.get("renderer")
    if not all(
        isinstance(value, Mapping)
        for value in (input_binding, reference_binding, pdf_binding, renderer)
    ):
        raise FinalizationError("journal PDF render receipt bindings are incomplete")
    manuscript = generation / "journal_main.md"
    manuscript_bytes = manuscript.read_bytes()
    if (
        input_binding.get("source_path") != str(manuscript)
        or input_binding.get("bytes") != len(manuscript_bytes)
        or input_binding.get("sha256") != _sha(manuscript_bytes)
    ):
        raise FinalizationError("render receipt manuscript binding drifted")
    reference_bytes = references.read_bytes()
    if (
        reference_binding.get("source_path") != str(references)
        or reference_binding.get("bytes") != len(reference_bytes)
        or reference_binding.get("sha256") != _sha(reference_bytes)
    ):
        raise FinalizationError("render receipt reference binding drifted")
    pdf_bytes = pdf.read_bytes()
    if (
        pdf_binding.get("source_path") != str(pdf)
        or pdf_binding.get("bytes") != len(pdf_bytes)
        or pdf_binding.get("sha256") != _sha(pdf_bytes)
        or not isinstance(pdf_binding.get("pages"), int)
        or not 8 <= int(pdf_binding["pages"]) <= 12
    ):
        raise FinalizationError("render receipt PDF binding drifted")
    from scripts import render_paper_pdf

    source_text = manuscript_bytes.decode("utf-8")
    expected_prepared = _sha(
        render_paper_pdf.prepare_markdown(source_text, profile="main").encode("utf-8")
    )
    expected_css = _sha(render_paper_pdf.css_for_profile("main").encode("utf-8"))
    if (
        receipt.get("prepared_markdown_sha256") != expected_prepared
        or receipt.get("css_sha256") != expected_css
        or not isinstance(renderer.get("pandoc"), str)
        or not renderer.get("pandoc")
        or not isinstance(renderer.get("chrome"), str)
        or not renderer.get("chrome")
    ):
        raise FinalizationError("journal PDF render receipt transformation drifted")
    return receipt, raw


def write_pdf_metadata(
    *,
    pdf_path: Path | str,
    finalized_generation: Path | str,
    references_path: Path | str,
    render_receipt_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    generation = Path(finalized_generation).expanduser().resolve(strict=True)
    finalization = verify_finalized_generation(
        generation, references_path=references_path
    )
    pdf = Path(pdf_path).expanduser().resolve(strict=True)
    pdf_bytes = pdf.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-1024:]:
        raise FinalizationError("journal PDF is not a complete PDF byte stream")
    finalization_path = generation / "finalization_manifest.json"
    manuscript_path = generation / "journal_main.md"
    references = Path(references_path).expanduser().resolve(strict=True)
    _, references_sha = _load_reference_ids(references)
    receipt_path = Path(render_receipt_path).expanduser().resolve(strict=True)
    receipt, receipt_bytes = _validate_render_receipt(
        receipt_path,
        generation=generation,
        references=references,
        pdf=pdf,
    )
    metadata: dict[str, Any] = {
        "schema_version": "yher.journal_pdf.metadata.v1",
        "pdf": {
            "source_path": str(pdf),
            "bytes": len(pdf_bytes),
            "sha256": _sha(pdf_bytes),
        },
        "finalized_generation": {
            "source_path": str(generation),
            "generation_id": generation.name,
            "finalization_manifest_sha256": _sha(finalization_path.read_bytes()),
        },
        "finalized_manuscript": {
            "source_path": str(manuscript_path),
            "bytes": manuscript_path.stat().st_size,
            "sha256": _sha(manuscript_path.read_bytes()),
        },
        "binder_generation_id": finalization["binder"]["generation_id"],
        "references": {"source_path": str(references), "sha256": references_sha},
        "render_receipt": {
            "source_path": str(receipt_path),
            "bytes": len(receipt_bytes),
            "sha256": _sha(receipt_bytes),
            "render_receipt_sha256": receipt["render_receipt_sha256"],
        },
    }
    metadata["metadata_sha256"] = _sha(_canonical(metadata))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_pretty(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return metadata


def verify_pdf_metadata(
    metadata_path: Path | str,
    *,
    references_path: Path | str,
) -> dict[str, Any]:
    metadata, _ = _load_json(Path(metadata_path), label="journal PDF metadata")
    advertised = metadata.get("metadata_sha256")
    payload = dict(metadata)
    payload.pop("metadata_sha256", None)
    if (
        metadata.get("schema_version") != "yher.journal_pdf.metadata.v1"
        or advertised != _sha(_canonical(payload))
    ):
        raise FinalizationError("journal PDF metadata self-hash drifted")
    pdf = metadata.get("pdf")
    finalized = metadata.get("finalized_generation")
    manuscript = metadata.get("finalized_manuscript")
    if not all(isinstance(value, Mapping) for value in (pdf, finalized, manuscript)):
        raise FinalizationError("journal PDF metadata bindings are incomplete")
    pdf_path = Path(str(pdf.get("source_path") or ""))
    if not pdf_path.is_file():
        raise FinalizationError("journal PDF bound by metadata is missing")
    pdf_bytes = pdf_path.read_bytes()
    if pdf.get("sha256") != _sha(pdf_bytes) or pdf.get("bytes") != len(pdf_bytes):
        raise FinalizationError("journal PDF bytes drifted")
    generation = Path(str(finalized.get("source_path") or ""))
    finalization = verify_finalized_generation(
        generation, references_path=references_path
    )
    finalization_path = generation / "finalization_manifest.json"
    if (
        finalized.get("generation_id") != generation.name
        or finalized.get("finalization_manifest_sha256")
        != _sha(finalization_path.read_bytes())
        or metadata.get("binder_generation_id")
        != finalization["binder"]["generation_id"]
    ):
        raise FinalizationError("journal PDF finalization binding drifted")
    manuscript_path = generation / "journal_main.md"
    manuscript_bytes = manuscript_path.read_bytes()
    if (
        manuscript.get("source_path") != str(manuscript_path)
        or manuscript.get("sha256") != _sha(manuscript_bytes)
        or manuscript.get("bytes") != len(manuscript_bytes)
    ):
        raise FinalizationError("journal PDF manuscript binding drifted")
    references = metadata.get("references")
    _, reference_sha = _load_reference_ids(Path(references_path))
    if not isinstance(references, Mapping) or references.get("sha256") != reference_sha:
        raise FinalizationError("journal PDF reference binding drifted")
    render_binding = metadata.get("render_receipt")
    if not isinstance(render_binding, Mapping):
        raise FinalizationError("journal PDF render receipt binding is missing")
    receipt_path = Path(str(render_binding.get("source_path") or ""))
    if not receipt_path.is_file():
        raise FinalizationError("journal PDF render receipt is missing")
    receipt_bytes = receipt_path.read_bytes()
    receipt, _ = _validate_render_receipt(
        receipt_path,
        generation=generation.resolve(strict=True),
        references=Path(references_path).expanduser().resolve(strict=True),
        pdf=pdf_path.resolve(strict=True),
    )
    if (
        render_binding.get("bytes") != len(receipt_bytes)
        or render_binding.get("sha256") != _sha(receipt_bytes)
        or render_binding.get("render_receipt_sha256")
        != receipt.get("render_receipt_sha256")
    ):
        raise FinalizationError("journal PDF render receipt bytes drifted")
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--template", type=Path, required=True)
    finalize.add_argument("--binder-generation", type=Path, required=True)
    finalize.add_argument("--references", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--expected-template-sha256", required=True)
    finalize.add_argument("--expected-binder-generation-id", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--generation", type=Path, required=True)
    verify.add_argument("--references", type=Path, required=True)
    verify.add_argument("--expected-template-sha256")
    verify.add_argument("--expected-binder-generation-id")
    pdf_metadata = subparsers.add_parser("pdf-metadata")
    pdf_metadata.add_argument("--pdf", type=Path, required=True)
    pdf_metadata.add_argument("--generation", type=Path, required=True)
    pdf_metadata.add_argument("--references", type=Path, required=True)
    pdf_metadata.add_argument("--render-receipt", type=Path, required=True)
    pdf_metadata.add_argument("--output", type=Path, required=True)
    verify_pdf = subparsers.add_parser("verify-pdf-metadata")
    verify_pdf.add_argument("--metadata", type=Path, required=True)
    verify_pdf.add_argument("--references", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        manifest = finalize_manuscript(
            template_path=args.template,
            binder_generation=args.binder_generation,
            references_path=args.references,
            output_dir=args.output,
            expected_template_sha256=args.expected_template_sha256,
            expected_binder_generation_id=args.expected_binder_generation_id,
        )
    elif args.command == "verify":
        manifest = verify_finalized_generation(
            args.generation,
            references_path=args.references,
            expected_template_sha256=args.expected_template_sha256,
            expected_binder_generation_id=args.expected_binder_generation_id,
        )
    elif args.command == "pdf-metadata":
        manifest = write_pdf_metadata(
            pdf_path=args.pdf,
            finalized_generation=args.generation,
            references_path=args.references,
            render_receipt_path=args.render_receipt,
            output_path=args.output,
        )
    else:
        manifest = verify_pdf_metadata(
            args.metadata, references_path=args.references
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _portable_file_binding(path: Path, *, anchor: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    anchor_resolved = anchor.expanduser().resolve()
    data = resolved.read_bytes()
    return {
        "relative_path": Path(os.path.relpath(resolved, anchor_resolved)).as_posix(),
        "bytes": len(data),
        "sha256": _sha(data),
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
        "sha256": _sha(data),
    }


def _portable_directory_binding(path: Path, *, anchor: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise FinalizationError(f"portable directory binding is not a directory: {resolved}")
    anchor_resolved = anchor.expanduser().resolve()
    return {
        "relative_path": Path(os.path.relpath(resolved, anchor_resolved)).as_posix()
    }


def _resolve_portable_path(
    binding: Mapping[str, Any],
    *,
    anchor: Path,
    label: str,
) -> Path:
    relative_value = binding.get("relative_path")
    if not isinstance(relative_value, str) or not relative_value:
        raise FinalizationError(f"{label} relative path is missing")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise FinalizationError(f"{label} relative path is absolute")
    try:
        return (anchor / relative).resolve(strict=True)
    except OSError as exc:
        raise FinalizationError(f"{label} portable path cannot resolve") from exc


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON object")
    return value, raw


def _assert_exact_manifest(
    manifest: Mapping[str, Any],
    raw: bytes,
    *,
    expected_keys: set[str],
    label: str,
) -> None:
    if set(manifest) != expected_keys:
        raise FinalizationError(f"{label} manifest schema drifted")
    if raw != _pretty(manifest):
        raise FinalizationError(f"{label} manifest is not canonical")


def _regular_file_roster(root: Path, *, label: str) -> set[str]:
    roster: set[str] = set()
    try:
        entries = sorted(root.rglob("*"), key=lambda path: path.as_posix())
        for path in entries:
            if path.is_symlink():
                raise FinalizationError(f"{label} contains a symbolic link: {path}")
            if path.is_file():
                roster.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise FinalizationError(f"{label} contains a special entry: {path}")
    except OSError as exc:
        raise FinalizationError(f"cannot inspect {label} file roster") from exc
    return roster


def _artifact_index(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    label: str,
    manifest_filename: str,
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
    if manifest_filename in index:
        raise FinalizationError(f"{label} manifest cannot list itself as an artifact")
    expected_roster = set(index) | {manifest_filename}
    if _regular_file_roster(root, label=label) != expected_roster:
        raise FinalizationError(f"{label} file roster differs from its manifest")
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
    _assert_exact_manifest(
        manifest,
        manifest_bytes,
        expected_keys={
            "schema_version",
            "generation_id",
            "generation_set_sha256",
            "artifacts",
        },
        label="binder",
    )
    if (
        manifest.get("schema_version") != "yher.journal_support_binder.output.v2"
        or manifest.get("generation_id") != generation.name
    ):
        raise FinalizationError("binder generation envelope drifted")
    artifacts = _artifact_index(
        manifest,
        root=generation,
        label="binder generation",
        manifest_filename="artifact_manifest.json",
    )
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


def _verify_binder_sources(
    binder: Mapping[str, Any],
    captured_sources: Mapping[str, bytes] | None = None,
) -> None:
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
        if captured_sources is None:
            path = Path(source)
            if not path.is_file():
                raise FinalizationError(f"stale binder source is missing: {source}")
            data = path.read_bytes()
        else:
            data = captured_sources.get(source)
            if data is None:
                raise FinalizationError(
                    f"captured binder source is missing: {source}"
                )
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
    for relative, descriptor in persona_assets["figure_data"].items():
        plan[
            f"assets/supplement/persona_v2/figure_data/{Path(relative).name}"
        ] = descriptor
    plan["assets/supplement/p2/figure_data.json"] = p2_assets["figure_data"]
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


def _prepare_generation_archive(root: Path) -> Path:
    generations = root / "generations"
    if generations.is_symlink():
        raise FinalizationError("journal generations archive must not be a symlink")
    generations.mkdir(parents=True, exist_ok=True)
    if generations.is_symlink() or not generations.is_dir():
        raise FinalizationError("journal generations archive is unsafe")
    if generations.resolve(strict=True).parent != root.resolve(strict=True):
        raise FinalizationError("journal generations archive failed containment")
    return generations


def _promote_finalized_generation(
    *,
    built_generation: Path,
    manifest: Mapping[str, Any],
    output_dir: Path | str,
    references_path: Path,
    expected_template_sha256: str,
    expected_binder_generation_id: str,
) -> None:
    root = Path(output_dir)
    generations = _prepare_generation_archive(root)
    generation_id = str(manifest["generation_id"])
    final_dir = generations / generation_id
    if final_dir.is_symlink():
        raise FinalizationError("journal finalization generation must not be a symlink")
    promotion_root = Path(
        tempfile.mkdtemp(prefix=f".{generation_id}.promote.", dir=generations)
    )
    staged_generation = promotion_root / generation_id
    link_temp = root / f".current.{generation_id}.{os.getpid()}.tmp"
    try:
        if final_dir.exists():
            verify_finalized_generation(
                final_dir,
                references_path=references_path,
                expected_template_sha256=expected_template_sha256,
                expected_binder_generation_id=expected_binder_generation_id,
            )
            if (final_dir / "finalization_manifest.json").read_bytes() != _pretty(
                manifest
            ):
                raise FinalizationError(
                    "existing journal finalization generation drifted"
                )
        else:
            staged_generation.mkdir()
            for source in sorted(
                built_generation.rglob("*"),
                key=lambda value: value.relative_to(built_generation).as_posix(),
            ):
                relative = source.relative_to(built_generation)
                if source.is_symlink() or (not source.is_file() and not source.is_dir()):
                    raise FinalizationError(
                        f"private finalized generation has an unsafe entry: {relative}"
                    )
                if source.is_file():
                    _write_file(staged_generation / relative, source.read_bytes())
            verify_finalized_generation(
                staged_generation,
                references_path=references_path,
                expected_template_sha256=expected_template_sha256,
                expected_binder_generation_id=expected_binder_generation_id,
            )
            os.replace(staged_generation, final_dir)
            _fsync_directory(generations)
        if link_temp.exists() or link_temp.is_symlink():
            link_temp.unlink()
        os.symlink(Path("generations") / generation_id, link_temp)
        os.replace(link_temp, root / "current")
        _fsync_directory(root)
    finally:
        shutil.rmtree(promotion_root, ignore_errors=True)
        if link_temp.exists() or link_temp.is_symlink():
            link_temp.unlink()


def finalize_manuscript(
    *,
    template_path: Path | str,
    binder_generation: Path | str,
    references_path: Path | str,
    output_dir: Path | str,
    expected_template_sha256: str,
    expected_binder_generation_id: str,
    _snapshot_mode: bool = False,
    _captured_sources: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    if not _snapshot_mode:
        from scripts import render_paper_pdf

        original_template = Path(template_path).expanduser().resolve(strict=True)
        original_references = Path(references_path).expanduser().resolve(strict=True)
        binder_locator = Path(binder_generation).expanduser()
        original_binder = binder_locator.resolve(strict=True)
        with tempfile.TemporaryDirectory(
            prefix=".journal-finalization-snapshot-"
        ) as temporary_directory:
            snapshot_root = Path(temporary_directory)
            template_bytes, template_state = render_paper_pdf._stable_read_file(
                original_template, label="journal template"
            )
            snapshot_template = snapshot_root / "template.md"
            snapshot_template.write_bytes(template_bytes)
            reference_bytes, reference_state = render_paper_pdf._stable_read_file(
                original_references, label="journal references"
            )
            snapshot_references = snapshot_root / "references.json"
            snapshot_references.write_bytes(reference_bytes)
            snapshot_binder = snapshot_root / original_binder.name
            binder_states = render_paper_pdf._snapshot_source_tree(
                original_binder, snapshot_binder
            )
            loaded = _load_binder_generation(
                snapshot_binder,
                expected_generation_id=expected_binder_generation_id,
            )
            snapshot_binder_payload = loaded[2]
            captured_sources: dict[str, bytes] = {}
            source_states: dict[str, Mapping[str, Any]] = {}
            for binding in _iter_source_bindings(snapshot_binder_payload):
                source_value = binding.get("source_path")
                if not isinstance(source_value, str) or source_value in captured_sources:
                    continue
                source_path = Path(source_value).expanduser().resolve(strict=True)
                data, state = render_paper_pdf._stable_read_file(
                    source_path, label=f"binder source {source_value}"
                )
                captured_sources[source_value] = data
                source_states[source_value] = state

            private_output = snapshot_root / "built"
            manifest = finalize_manuscript(
                template_path=snapshot_template,
                binder_generation=snapshot_binder,
                references_path=snapshot_references,
                output_dir=private_output,
                expected_template_sha256=expected_template_sha256,
                expected_binder_generation_id=expected_binder_generation_id,
                _snapshot_mode=True,
                _captured_sources=captured_sources,
            )
            try:
                render_paper_pdf._assert_file_unchanged(
                    original_template, template_state, label="journal template"
                )
                render_paper_pdf._assert_file_unchanged(
                    original_references, reference_state, label="journal references"
                )
                render_paper_pdf._assert_source_tree_unchanged(
                    original_binder, binder_states
                )
                if binder_locator.resolve(strict=True) != original_binder:
                    raise FinalizationError(
                        "journal binder locator changed during finalization"
                    )
                for source_value, state in source_states.items():
                    render_paper_pdf._assert_file_unchanged(
                        Path(source_value),
                        state,
                        label=f"binder source {source_value}",
                    )
            except render_paper_pdf.RenderError as exc:
                raise FinalizationError(
                    f"journal finalization input changed during build: {exc}"
                ) from exc
            _promote_finalized_generation(
                built_generation=(private_output / "current").resolve(strict=True),
                manifest=manifest,
                output_dir=output_dir,
                references_path=original_references,
                expected_template_sha256=expected_template_sha256,
                expected_binder_generation_id=expected_binder_generation_id,
            )
            return manifest

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
    _verify_binder_sources(binder, _captured_sources)
    final_text = _replace_slots(template_text, slots)
    audit_manuscript(final_text, references_path=references)
    abstract_word_count = _enforce_structured_abstract_word_limit(final_text)
    manuscript_bytes = final_text.encode("utf-8")
    _, references_sha = _load_reference_ids(references)
    references_bytes = references.read_bytes()

    asset_plan = _copy_plan(binder)
    asset_bytes: dict[str, bytes] = {}
    for relative, descriptor in asset_plan.items():
        source_value = str(descriptor["source_path"])
        source = Path(source_value)
        if _captured_sources is None:
            data = source.read_bytes()
        else:
            data = _captured_sources.get(source_value)
            if data is None:
                raise FinalizationError(
                    f"captured binder asset is missing: {source_value}"
                )
        if descriptor.get("sha256") != _sha(data) or descriptor.get("bytes") != len(data):
            raise FinalizationError(f"stale binder source bytes drifted: {source}")
        asset_bytes[relative] = data
    missing_assets = _local_asset_references(final_text) - set(asset_bytes)
    if missing_assets:
        raise FinalizationError(
            "final manuscript references an unbound local asset: "
            + ", ".join(sorted(missing_assets))
        )
    provenance_bytes = {
        "provenance/template.md": template_bytes,
        "provenance/references.json": references_bytes,
        "provenance/binder/artifact_manifest.json": binder_manifest_bytes,
        "provenance/binder/journal_binder.json": binder_bytes,
        "provenance/binder/manuscript_slots.json": slots_bytes,
    }
    packaged_bytes = {**asset_bytes, **provenance_bytes}
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
            for relative, data in sorted(packaged_bytes.items())
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
            "filename": "provenance/template.md",
            "bytes": len(template_bytes),
            "sha256": expected_template_sha256,
        },
        "binder": {
            "generation_id": binder_path.name,
            "filenames": {
                "artifact_manifest": "provenance/binder/artifact_manifest.json",
                "journal_binder": "provenance/binder/journal_binder.json",
                "manuscript_slots": "provenance/binder/manuscript_slots.json",
            },
            "artifact_manifest_sha256": _sha(binder_manifest_bytes),
            "journal_binder_sha256": _sha(binder_bytes),
        },
        "slots": {
            "manuscript_slots_sha256": _sha(slots_bytes),
            "content_sha256": slots["content_sha256"],
        },
        "references": {
            "filename": "provenance/references.json",
            "bytes": len(references_bytes),
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
    generations = _prepare_generation_archive(root)
    final_dir = generations / generation_id
    if final_dir.is_symlink():
        raise FinalizationError("journal finalization generation must not be a symlink")
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    link_temp = root / f".current.{generation_id}.{os.getpid()}.tmp"
    try:
        _write_file(staging / "journal_main.md", manuscript_bytes)
        for relative, data in packaged_bytes.items():
            _write_file(staging / relative, data)
        _write_file(staging / "finalization_manifest.json", _pretty(manifest))
        _fsync_directory(staging)
        if final_dir.exists():
            verify_finalized_generation(
                final_dir,
                references_path=references,
                expected_template_sha256=expected_template_sha256,
                expected_binder_generation_id=expected_binder_generation_id,
            )
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
    _snapshot_mode: bool = False,
) -> dict[str, Any]:
    if not _snapshot_mode:
        from scripts import render_paper_pdf

        original_generation = Path(generation_path).expanduser().resolve(strict=True)
        original_references = Path(references_path).expanduser().resolve(strict=True)
        with tempfile.TemporaryDirectory(
            prefix=".journal-generation-verification-"
        ) as temporary_directory:
            snapshot_root = Path(temporary_directory)
            snapshot_generation = snapshot_root / original_generation.name
            generation_states = render_paper_pdf._snapshot_source_tree(
                original_generation, snapshot_generation
            )
            reference_bytes, reference_state = render_paper_pdf._stable_read_file(
                original_references, label="finalized generation references"
            )
            snapshot_references = snapshot_root / "references.json"
            snapshot_references.write_bytes(reference_bytes)
            result = verify_finalized_generation(
                snapshot_generation,
                references_path=snapshot_references,
                expected_template_sha256=expected_template_sha256,
                expected_binder_generation_id=expected_binder_generation_id,
                _snapshot_mode=True,
            )
            try:
                render_paper_pdf._assert_source_tree_unchanged(
                    original_generation, generation_states
                )
                render_paper_pdf._assert_file_unchanged(
                    original_references,
                    reference_state,
                    label="finalized generation references",
                )
            except render_paper_pdf.RenderError as exc:
                raise FinalizationError(
                    f"finalized generation input changed during verification: {exc}"
                ) from exc
            return result

    try:
        generation = Path(generation_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise FinalizationError("finalized journal generation cannot resolve") from exc
    manifest, manifest_bytes = _load_json(
        generation / "finalization_manifest.json", label="finalization manifest"
    )
    _assert_exact_manifest(
        manifest,
        manifest_bytes,
        expected_keys={
            "schema_version",
            "generation_id",
            "template",
            "binder",
            "slots",
            "references",
            "final_manuscript",
            "structured_abstract",
            "artifact_set_sha256",
            "artifacts",
        },
        label="finalization",
    )
    if (
        manifest.get("schema_version") != "yher.journal_manuscript.finalization.v1"
        or manifest.get("generation_id") != generation.name
    ):
        raise FinalizationError("finalization manifest envelope drifted")
    artifacts = _artifact_index(
        manifest,
        root=generation,
        label="final manuscript",
        manifest_filename="finalization_manifest.json",
    )
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
        set(template_binding) != {"filename", "bytes", "sha256"}
        or set(binder_binding)
        != {
            "generation_id",
            "filenames",
            "artifact_manifest_sha256",
            "journal_binder_sha256",
        }
        or set(slot_binding) != {"manuscript_slots_sha256", "content_sha256"}
        or not isinstance(reference_binding, Mapping)
        or set(reference_binding) != {"filename", "bytes", "sha256"}
        or set(final) != {"filename", "bytes", "sha256"}
        or not isinstance(abstract_binding, Mapping)
        or set(abstract_binding) != {"word_count", "maximum_words"}
    ):
        raise FinalizationError("finalization manifest schema drifted")
    if any(
        "source_path" in value
        for value in (template_binding, binder_binding, reference_binding)
    ):
        raise FinalizationError("final manuscript provenance is not portable")
    template_filename = template_binding.get("filename")
    reference_filename = reference_binding.get("filename")
    binder_filenames = binder_binding.get("filenames")
    if (
        not isinstance(template_filename, str)
        or artifacts.get(template_filename) != dict(template_binding)
        or not isinstance(reference_filename, str)
        or artifacts.get(reference_filename) != dict(reference_binding)
        or not isinstance(binder_filenames, Mapping)
        or set(binder_filenames)
        != {"artifact_manifest", "journal_binder", "manuscript_slots"}
    ):
        raise FinalizationError("final manuscript packaged provenance is incomplete")
    binder_artifact = artifacts.get(str(binder_filenames["artifact_manifest"]))
    binder_json = artifacts.get(str(binder_filenames["journal_binder"]))
    binder_slots = artifacts.get(str(binder_filenames["manuscript_slots"]))
    if (
        not isinstance(binder_artifact, Mapping)
        or binder_artifact.get("sha256")
        != binder_binding.get("artifact_manifest_sha256")
        or not isinstance(binder_json, Mapping)
        or binder_json.get("sha256") != binder_binding.get("journal_binder_sha256")
        or not isinstance(binder_slots, Mapping)
        or binder_slots.get("sha256") != slot_binding.get("manuscript_slots_sha256")
    ):
        raise FinalizationError("final manuscript packaged binder provenance drifted")
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
    pandoc: str,
    chrome: str,
    pdfinfo: str,
    pdftotext: str,
    pdftoppm: str,
    _snapshot_mode: bool = False,
    _binding_receipt_path: Path | None = None,
    _binding_generation: Path | None = None,
    _binding_references: Path | None = None,
    _binding_pdf: Path | None = None,
    _expected_wrapper: Mapping[str, Any] | None = None,
    _expected_tools: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], bytes]:
    from scripts import render_paper_pdf

    if not _snapshot_mode:
        original_receipt = receipt_path.expanduser().resolve(strict=True)
        original_generation = generation.expanduser().resolve(strict=True)
        original_references = references.expanduser().resolve(strict=True)
        original_pdf = pdf.expanduser().resolve(strict=True)
        tools_before = render_paper_pdf._render_tool_descriptors(
            pandoc=pandoc,
            chrome=chrome,
            pdfinfo=pdfinfo,
            pdftotext=pdftotext,
            pdftoppm=pdftoppm,
        )
        wrapper_path = Path(render_paper_pdf.__file__).resolve(strict=True)
        wrapper_bytes, wrapper_state = render_paper_pdf._stable_read_file(
            wrapper_path, label="journal render wrapper"
        )
        with tempfile.TemporaryDirectory(
            prefix=".journal-pdf-validation-"
        ) as temporary_directory:
            snapshot_root = Path(temporary_directory)
            snapshot_generation = snapshot_root / original_generation.name
            generation_states = render_paper_pdf._snapshot_source_tree(
                original_generation, snapshot_generation
            )
            reference_bytes, reference_state = render_paper_pdf._stable_read_file(
                original_references, label="journal reference registry"
            )
            snapshot_references = snapshot_root / "references.json"
            snapshot_references.write_bytes(reference_bytes)
            pdf_bytes, pdf_state = render_paper_pdf._stable_read_file(
                original_pdf, label="journal PDF"
            )
            snapshot_pdf = snapshot_root / "journal.pdf"
            snapshot_pdf.write_bytes(pdf_bytes)
            receipt_bytes, receipt_state = render_paper_pdf._stable_read_file(
                original_receipt, label="journal render receipt"
            )
            snapshot_receipt = snapshot_root / "journal.pdf.render.json"
            snapshot_receipt.write_bytes(receipt_bytes)
            verify_finalized_generation(
                snapshot_generation,
                references_path=snapshot_references,
            )
            tool_paths = {
                name: str(descriptor["resolved_path"])
                for name, descriptor in tools_before.items()
            }
            result = _validate_render_receipt(
                snapshot_receipt,
                generation=snapshot_generation,
                references=snapshot_references,
                pdf=snapshot_pdf,
                pandoc=tool_paths["pandoc"],
                chrome=tool_paths["chrome"],
                pdfinfo=tool_paths["pdfinfo"],
                pdftotext=tool_paths["pdftotext"],
                pdftoppm=tool_paths["pdftoppm"],
                _snapshot_mode=True,
                _binding_receipt_path=original_receipt,
                _binding_generation=original_generation,
                _binding_references=original_references,
                _binding_pdf=original_pdf,
                _expected_wrapper=render_paper_pdf._portable_wrapper_binding(
                    wrapper_path, wrapper_bytes
                ),
                _expected_tools=render_paper_pdf._portable_tool_descriptors(
                    tools_before
                ),
            )
            render_paper_pdf._assert_source_tree_unchanged(
                original_generation, generation_states
            )
            render_paper_pdf._assert_file_unchanged(
                original_references,
                reference_state,
                label="journal reference registry",
            )
            render_paper_pdf._assert_file_unchanged(
                original_pdf, pdf_state, label="journal PDF"
            )
            render_paper_pdf._assert_file_unchanged(
                original_receipt,
                receipt_state,
                label="journal render receipt",
            )
            render_paper_pdf._assert_file_unchanged(
                wrapper_path,
                wrapper_state,
                label="journal render wrapper",
            )
            render_paper_pdf._assert_render_tools_unchanged(tools_before)
            return result

    if not all(
        isinstance(path, Path)
        for path in (
            _binding_receipt_path,
            _binding_generation,
            _binding_references,
            _binding_pdf,
        )
    ) or _expected_wrapper is None or _expected_tools is None:
        raise FinalizationError("journal PDF private snapshot binding is incomplete")
    binding_receipt_path = _binding_receipt_path
    binding_generation = _binding_generation
    binding_references = _binding_references
    binding_pdf = _binding_pdf
    receipt, raw = _load_json(receipt_path, label="journal PDF render receipt")
    advertised = receipt.get("render_receipt_sha256")
    payload = dict(receipt)
    payload.pop("render_receipt_sha256", None)
    if (
        receipt.get("schema_version") != "yher.paper_pdf.render_receipt.v2"
        or advertised != _sha(_canonical(payload))
    ):
        raise FinalizationError("journal PDF render receipt self-hash drifted")
    if receipt.get("profile") != "main":
        raise FinalizationError("journal PDF render receipt profile is not main")
    input_binding = receipt.get("input")
    reference_binding = receipt.get("references")
    pdf_binding = receipt.get("pdf")
    renderer = receipt.get("renderer")
    source_resources = receipt.get("source_resources")
    source_equivalence = receipt.get("source_equivalence")
    if not all(
        isinstance(value, Mapping)
        for value in (
            input_binding,
            reference_binding,
            pdf_binding,
            renderer,
            source_resources,
            source_equivalence,
        )
    ):
        raise FinalizationError("journal PDF render receipt bindings are incomplete")
    manuscript = generation / "journal_main.md"
    manuscript_bytes = manuscript.read_bytes()
    expected_input_binding = render_paper_pdf._portable_binding_from_bytes(
        binding_generation / "journal_main.md",
        anchor=binding_receipt_path.parent,
        data=manuscript_bytes,
    )
    if input_binding != expected_input_binding:
        raise FinalizationError("render receipt manuscript binding drifted")
    expected_source_resources = render_paper_pdf._portable_tree_manifest_from_root(
        generation
    )
    if source_resources != expected_source_resources:
        raise FinalizationError("render receipt source-resource tree drifted")
    reference_bytes = references.read_bytes()
    expected_reference_binding = render_paper_pdf._portable_binding_from_bytes(
        binding_references,
        anchor=binding_receipt_path.parent,
        data=reference_bytes,
    )
    if reference_binding != expected_reference_binding:
        raise FinalizationError("render receipt reference binding drifted")
    pdf_bytes = pdf.read_bytes()
    expected_pdf_binding = {
        **render_paper_pdf._portable_binding_from_bytes(
            binding_pdf,
            anchor=binding_receipt_path.parent,
            data=pdf_bytes,
        ),
        "pages": pdf_binding.get("pages"),
    }
    if (
        pdf_binding != expected_pdf_binding
        or not isinstance(pdf_binding.get("pages"), int)
    ):
        raise FinalizationError("render receipt PDF binding drifted")
    try:
        verified_pages = render_paper_pdf._page_count(
            pdf, pdfinfo=pdfinfo
        )
        rendered_text = render_paper_pdf._pdf_text(
            pdf, pdftotext=pdftotext
        )
        render_paper_pdf.validate_rendered_text(
            rendered_text,
            pages=verified_pages,
            expected_pages=None,
        )
    except render_paper_pdf.RenderError as exc:
        raise FinalizationError("journal PDF independent validation failed") from exc
    if (
        not 8 <= verified_pages <= 12
        or pdf_binding.get("pages") != verified_pages
    ):
        raise FinalizationError("render receipt PDF page validation drifted")

    source_text = manuscript_bytes.decode("utf-8")
    expected_prepared = _sha(
        render_paper_pdf.prepare_markdown(source_text, profile="main").encode("utf-8")
    )
    expected_css = _sha(render_paper_pdf.css_for_profile("main").encode("utf-8"))
    wrapper = renderer.get("wrapper")
    tools = renderer.get("tools")
    if not isinstance(wrapper, Mapping) or wrapper != _expected_wrapper:
        raise FinalizationError("journal PDF render wrapper binding drifted")
    if not isinstance(tools, Mapping) or tools != _expected_tools:
        raise FinalizationError("journal PDF render tool roster drifted")
    if (
        receipt.get("prepared_markdown_sha256") != expected_prepared
        or receipt.get("css_sha256") != expected_css
    ):
        raise FinalizationError("journal PDF render receipt transformation drifted")
    try:
        independent_equivalence = render_paper_pdf.verify_source_bound_pdf(
            profile="main",
            input_path=manuscript,
            references_path=references,
            output_path=pdf,
            pages=verified_pages,
            pandoc=pandoc,
            chrome=chrome,
            pdfinfo=pdfinfo,
            pdftotext=pdftotext,
            pdftoppm=pdftoppm,
        )
    except render_paper_pdf.RenderError as exc:
        raise FinalizationError(
            f"journal PDF independent source rerender validation failed: {exc}"
        ) from exc
    if source_equivalence != independent_equivalence:
        raise FinalizationError(
            "journal PDF render receipt independent source-equivalence evidence drifted"
        )
    return receipt, raw


def write_pdf_metadata(
    *,
    pdf_path: Path | str,
    finalized_generation: Path | str,
    references_path: Path | str,
    render_receipt_path: Path | str,
    output_path: Path | str,
    pandoc: str = "/opt/homebrew/bin/pandoc",
    chrome: str = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    pdfinfo: str = "pdfinfo",
    pdftotext: str = "pdftotext",
    pdftoppm: str = "pdftoppm",
) -> dict[str, Any]:
    generation = Path(finalized_generation).expanduser().resolve(strict=True)
    pdf = Path(pdf_path).expanduser().resolve(strict=True)
    references = Path(references_path).expanduser().resolve(strict=True)
    receipt_path = Path(render_receipt_path).expanduser().resolve(strict=True)
    from scripts import render_paper_pdf
    try:
        publication_paths = render_paper_pdf._assert_distinct_paths(
            pdf=pdf,
            references=references,
            receipt=receipt_path,
            metadata=Path(output_path),
        )
    except render_paper_pdf.RenderError as exc:
        raise FinalizationError(str(exc)) from exc
    output = publication_paths["metadata"]
    if output == generation or generation in output.parents:
        raise FinalizationError(
            "journal PDF metadata output must be distinct from the finalized generation"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    anchor = output.parent
    with tempfile.TemporaryDirectory(
        prefix=".journal-metadata-snapshot-"
    ) as temporary_directory:
        snapshot_root = Path(temporary_directory)
        snapshot_generation = snapshot_root / generation.name
        generation_states = render_paper_pdf._snapshot_source_tree(
            generation, snapshot_generation
        )
        reference_bytes, reference_state = render_paper_pdf._stable_read_file(
            references, label="journal metadata references"
        )
        snapshot_references = snapshot_root / "references.json"
        snapshot_references.write_bytes(reference_bytes)
        pdf_bytes, pdf_state = render_paper_pdf._stable_read_file(
            pdf, label="journal metadata PDF"
        )
        if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-1024:]:
            raise FinalizationError("journal PDF is not a complete PDF byte stream")
        receipt_bytes, receipt_state = render_paper_pdf._stable_read_file(
            receipt_path, label="journal metadata render receipt"
        )
        finalization = verify_finalized_generation(
            snapshot_generation, references_path=snapshot_references
        )
        try:
            receipt, validated_receipt_bytes = _validate_render_receipt(
                receipt_path,
                generation=generation,
                references=references,
                pdf=pdf,
                pandoc=pandoc,
                chrome=chrome,
                pdfinfo=pdfinfo,
                pdftotext=pdftotext,
                pdftoppm=pdftoppm,
            )
        except render_paper_pdf.RenderError as exc:
            raise FinalizationError(
                f"journal PDF snapshot validation failed: {exc}"
            ) from exc
        if validated_receipt_bytes != receipt_bytes:
            raise FinalizationError("journal PDF render receipt changed during validation")

        finalization_bytes = (
            snapshot_generation / "finalization_manifest.json"
        ).read_bytes()
        manuscript_bytes = (snapshot_generation / "journal_main.md").read_bytes()
        _, references_sha = _load_reference_ids(snapshot_references)
        metadata: dict[str, Any] = {
            "schema_version": "yher.journal_pdf.metadata.v2",
            "pdf": _portable_binding_from_bytes(pdf, anchor=anchor, data=pdf_bytes),
            "finalized_generation": {
                **_portable_directory_binding(generation, anchor=anchor),
                "generation_id": generation.name,
                "finalization_manifest_sha256": _sha(finalization_bytes),
            },
            "finalized_manuscript": _portable_binding_from_bytes(
                generation / "journal_main.md",
                anchor=anchor,
                data=manuscript_bytes,
            ),
            "binder_generation_id": finalization["binder"]["generation_id"],
            "references": _portable_binding_from_bytes(
                references, anchor=anchor, data=reference_bytes
            ),
            "render_receipt": {
                **_portable_binding_from_bytes(
                    receipt_path, anchor=anchor, data=receipt_bytes
                ),
                "render_receipt_sha256": receipt["render_receipt_sha256"],
            },
        }
        if metadata["references"]["sha256"] != references_sha:
            raise FinalizationError("journal PDF metadata reference binding drifted")
        metadata["metadata_sha256"] = _sha(_canonical(metadata))
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", dir=output.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_pretty(metadata))
                handle.flush()
                os.fsync(handle.fileno())
            render_paper_pdf._assert_source_tree_unchanged(
                generation, generation_states
            )
            render_paper_pdf._assert_file_unchanged(
                references,
                reference_state,
                label="journal metadata references",
            )
            render_paper_pdf._assert_file_unchanged(
                pdf, pdf_state, label="journal metadata PDF"
            )
            render_paper_pdf._assert_file_unchanged(
                receipt_path,
                receipt_state,
                label="journal metadata render receipt",
            )
            os.replace(temporary, output)
            _fsync_directory(output.parent)
        except render_paper_pdf.RenderError as exc:
            temporary.unlink(missing_ok=True)
            raise FinalizationError(
                f"journal PDF metadata input changed before publication: {exc}"
            ) from exc
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return metadata


def verify_pdf_metadata(
    metadata_path: Path | str,
    *,
    references_path: Path | str,
    pandoc: str = "/opt/homebrew/bin/pandoc",
    chrome: str = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    pdfinfo: str = "pdfinfo",
    pdftotext: str = "pdftotext",
    pdftoppm: str = "pdftoppm",
) -> dict[str, Any]:
    metadata_file = Path(metadata_path).expanduser().resolve(strict=True)
    metadata, _ = _load_json(metadata_file, label="journal PDF metadata")
    anchor = metadata_file.parent
    advertised = metadata.get("metadata_sha256")
    payload = dict(metadata)
    payload.pop("metadata_sha256", None)
    if (
        metadata.get("schema_version") != "yher.journal_pdf.metadata.v2"
        or advertised != _sha(_canonical(payload))
    ):
        raise FinalizationError("journal PDF metadata self-hash drifted")
    pdf = metadata.get("pdf")
    finalized = metadata.get("finalized_generation")
    manuscript = metadata.get("finalized_manuscript")
    references = metadata.get("references")
    render_binding = metadata.get("render_receipt")
    if not all(
        isinstance(value, Mapping)
        for value in (pdf, finalized, manuscript, references, render_binding)
    ):
        raise FinalizationError("journal PDF metadata bindings are incomplete")
    pdf_path = _resolve_portable_path(pdf, anchor=anchor, label="journal PDF")
    pdf_bytes = pdf_path.read_bytes()
    if pdf != _portable_file_binding(pdf_path, anchor=anchor):
        raise FinalizationError("journal PDF bytes drifted")
    generation = _resolve_portable_path(
        finalized, anchor=anchor, label="finalized journal generation"
    )
    if not generation.is_dir():
        raise FinalizationError("finalized journal generation is not a directory")
    finalization = verify_finalized_generation(
        generation, references_path=references_path
    )
    finalization_path = generation / "finalization_manifest.json"
    expected_finalized = {
        **_portable_directory_binding(generation, anchor=anchor),
        "generation_id": generation.name,
        "finalization_manifest_sha256": _sha(finalization_path.read_bytes()),
    }
    if (
        finalized != expected_finalized
        or metadata.get("binder_generation_id")
        != finalization["binder"]["generation_id"]
    ):
        raise FinalizationError("journal PDF finalization binding drifted")
    manuscript_path = generation / "journal_main.md"
    manuscript_bytes = manuscript_path.read_bytes()
    if manuscript != _portable_file_binding(manuscript_path, anchor=anchor):
        raise FinalizationError("journal PDF manuscript binding drifted")
    reference_path = Path(references_path).expanduser().resolve(strict=True)
    bound_reference_path = _resolve_portable_path(
        references, anchor=anchor, label="journal references"
    )
    _, reference_sha = _load_reference_ids(reference_path)
    if (
        bound_reference_path != reference_path
        or references != _portable_file_binding(reference_path, anchor=anchor)
        or references.get("sha256") != reference_sha
    ):
        raise FinalizationError("journal PDF reference binding drifted")
    receipt_path = _resolve_portable_path(
        render_binding, anchor=anchor, label="journal PDF render receipt"
    )
    receipt_bytes = receipt_path.read_bytes()
    from scripts import render_paper_pdf

    try:
        receipt, _ = _validate_render_receipt(
            receipt_path,
            generation=generation.resolve(strict=True),
            references=reference_path,
            pdf=pdf_path.resolve(strict=True),
            pandoc=pandoc,
            chrome=chrome,
            pdfinfo=pdfinfo,
            pdftotext=pdftotext,
            pdftoppm=pdftoppm,
        )
    except render_paper_pdf.RenderError as exc:
        raise FinalizationError(f"journal PDF snapshot validation failed: {exc}") from exc
    if (
        {
            key: value
            for key, value in render_binding.items()
            if key != "render_receipt_sha256"
        }
        != _portable_file_binding(receipt_path, anchor=anchor)
        or render_binding.get("bytes") != len(receipt_bytes)
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
    pdf_metadata.add_argument("--pandoc", default="/opt/homebrew/bin/pandoc")
    pdf_metadata.add_argument(
        "--chrome",
        default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    pdf_metadata.add_argument("--pdfinfo", default="pdfinfo")
    pdf_metadata.add_argument("--pdftotext", default="pdftotext")
    pdf_metadata.add_argument("--pdftoppm", default="pdftoppm")
    verify_pdf = subparsers.add_parser("verify-pdf-metadata")
    verify_pdf.add_argument("--metadata", type=Path, required=True)
    verify_pdf.add_argument("--references", type=Path, required=True)
    verify_pdf.add_argument("--pandoc", default="/opt/homebrew/bin/pandoc")
    verify_pdf.add_argument(
        "--chrome",
        default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    verify_pdf.add_argument("--pdfinfo", default="pdfinfo")
    verify_pdf.add_argument("--pdftotext", default="pdftotext")
    verify_pdf.add_argument("--pdftoppm", default="pdftoppm")
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
            pandoc=args.pandoc,
            chrome=args.chrome,
            pdfinfo=args.pdfinfo,
            pdftotext=args.pdftotext,
            pdftoppm=args.pdftoppm,
        )
    else:
        manifest = verify_pdf_metadata(
            args.metadata,
            references_path=args.references,
            pandoc=args.pandoc,
            chrome=args.chrome,
            pdfinfo=args.pdfinfo,
            pdftotext=args.pdftotext,
            pdftoppm=args.pdftoppm,
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

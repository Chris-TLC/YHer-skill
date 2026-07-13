from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.dataset import DatasetContractError, load_manifest_dataset


def _json_line(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _write_dataset(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    shard = root / "shard-a.jsonl"
    journey = {
        "simulated": True,
        "persona_id": "confirmatory:T:M:matched:0:A",
        "provider": "programmatic",
        "model_id": "production-engine:test",
        "record_type": "confirmatory_journey",
        "target_node": "T",
        "truth": "M",
        "condition": "matched",
        "replicate": 0,
        "arm": "A",
        "events": [],
        "views": [],
    }
    records_payload = _json_line(journey)
    shard_manifest = {
        "simulated": True,
        "persona_id": "confirmatory-shard:target=T|truth=M|condition=matched",
        "provider": "programmatic",
        "model_id": "production-engine:test",
        "record_type": "confirmatory_shard_manifest",
        "record_count": 1,
        "records_sha256": hashlib.sha256(records_payload).hexdigest(),
        "shard_id": "target=T|truth=M|condition=matched",
    }
    shard.write_bytes(_json_line(shard_manifest) + records_payload)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "simulated": True,
                "persona_id": "confirmatory-run:test",
                "provider": "programmatic",
                "model_id": "production-engine:test",
                "record_type": "confirmatory_run_manifest",
                "expected_journey_count": 1,
                "bootstrap_seed": 2026071301,
                "shards": [
                    {
                        "filename": shard.name,
                        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                        "shard_id": "target=T|truth=M|condition=matched",
                    }
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest, shard


def test_loader_reads_only_manifest_shards_and_returns_stable_raw_hash(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_dataset(tmp_path / "run")
    (manifest.parent / "unlisted.jsonl").write_text("not json\n", encoding="utf-8")

    first = load_manifest_dataset(manifest)
    second = load_manifest_dataset(manifest)

    assert len(first.journeys) == 1
    assert first.journeys[0]["target_node"] == "T"
    assert first.raw_hash == second.raw_hash
    assert len(first.raw_hash) == 64
    assert first.shard_paths == (manifest.parent / "shard-a.jsonl",)
    assert first.intended_journey_count == 1
    assert first.invalid_record_count == 0
    assert first.invalid_reasons == {}


def test_loader_rejects_a_shard_whose_bytes_do_not_match_manifest_sha(
    tmp_path: Path,
) -> None:
    manifest, shard = _write_dataset(tmp_path / "run")
    shard.write_bytes(shard.read_bytes() + b"\n")

    with pytest.raises(DatasetContractError, match="SHA-256"):
        load_manifest_dataset(manifest)


def test_loader_reuses_pre_read_manifest_bytes_without_reopening_manifest(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_dataset(tmp_path / "run")
    manifest_bytes = manifest.read_bytes()
    manifest.unlink()

    dataset = load_manifest_dataset(manifest, manifest_bytes=manifest_bytes)

    assert len(dataset.journeys) == 1
    assert dataset.manifest["record_type"] == "confirmatory_run_manifest"


def test_loader_rejects_records_hash_or_simulated_envelope_drift(tmp_path: Path) -> None:
    manifest, shard = _write_dataset(tmp_path / "run")
    lines = shard.read_bytes().splitlines(keepends=True)
    shard_manifest = json.loads(lines[0])
    shard_manifest["records_sha256"] = "0" * 64
    shard.write_bytes(_json_line(shard_manifest) + b"".join(lines[1:]))
    run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    run_manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(run_manifest), encoding="utf-8")

    with pytest.raises(DatasetContractError, match="records_sha256"):
        load_manifest_dataset(manifest)

    manifest, shard = _write_dataset(tmp_path / "second-run")
    lines = shard.read_bytes().splitlines(keepends=True)
    journey = json.loads(lines[1])
    journey["simulated"] = False
    records_payload = _json_line(journey)
    shard_manifest = json.loads(lines[0])
    shard_manifest["records_sha256"] = hashlib.sha256(records_payload).hexdigest()
    shard.write_bytes(_json_line(shard_manifest) + records_payload)
    run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    run_manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(run_manifest), encoding="utf-8")

    quarantined = load_manifest_dataset(manifest)
    assert quarantined.journeys == ()
    assert quarantined.intended_journey_count == 1
    assert quarantined.invalid_record_count == 1
    assert quarantined.invalid_reasons == {"simulated_data_envelope": 1}


def test_loader_quarantines_projector_schema_errors_and_preserves_intention_count(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_dataset(tmp_path / "run")

    def invalid_projector(_record: dict[str, object]) -> dict[str, object]:
        raise DatasetContractError("posterior schema invalid")

    dataset = load_manifest_dataset(manifest, projector=invalid_projector)

    assert dataset.journeys == ()
    assert dataset.intended_journey_count == 1
    assert dataset.invalid_record_count == 1
    assert dataset.invalid_reasons == {"posterior_schema_invalid": 1}


def test_loader_quarantines_malformed_json_after_hash_verification(tmp_path: Path) -> None:
    manifest, shard = _write_dataset(tmp_path / "run")
    lines = shard.read_bytes().splitlines(keepends=True)
    records_payload = b"not-json\n"
    shard_manifest = json.loads(lines[0])
    shard_manifest["records_sha256"] = hashlib.sha256(records_payload).hexdigest()
    shard.write_bytes(_json_line(shard_manifest) + records_payload)
    run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    run_manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(run_manifest), encoding="utf-8")

    dataset = load_manifest_dataset(manifest)

    assert dataset.journeys == ()
    assert dataset.invalid_record_count == 1
    assert dataset.invalid_reasons == {"invalid_json": 1}
    assert dataset.invalid_primary_keys == (("T", "M", "matched", 0, "A"),)

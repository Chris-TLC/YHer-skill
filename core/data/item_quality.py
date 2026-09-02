"""Central item quality gate for diagnosis, practice, teaching, and profile evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

SKILL_DIR = Path(__file__).parent.parent.parent
DEFAULT_QUALITY_MANIFEST = SKILL_DIR / "data" / "quality" / "item_quality_manifest.jsonl"

Purpose = Literal["diagnosis", "practice", "teaching", "profile_evidence", "debug_all"]
PURPOSES = {"diagnosis", "practice", "teaching", "profile_evidence", "debug_all"}


@dataclass(frozen=True)
class QualityDecision:
    allowed: bool
    blockers: list[str]
    status: dict[str, Any] | None = None


class ItemQualityGate:
    """Load item quality manifest and decide whether an item may serve a purpose."""

    def __init__(self, manifest_path: Path | str | None = None):
        self.manifest_path = Path(manifest_path) if manifest_path else DEFAULT_QUALITY_MANIFEST
        self._loaded = False
        self._status_by_id: dict[str, dict[str, Any]] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_id = row.get("item_id")
                if item_id:
                    self._status_by_id[item_id] = row

    @property
    def has_manifest(self) -> bool:
        return self.manifest_path.exists()

    def status_for(self, item_id: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        return self._status_by_id.get(item_id)

    def _legacy_student_readable(self, status: dict[str, Any], purpose: Purpose) -> bool:
        if "student_readable" in status:
            return bool(status.get("student_readable"))
        if status.get("needs_image"):
            usable_key = f"usable_for_{purpose}"
            return bool(
                status.get(usable_key, False)
                and status.get("visual_asset_status") in {"strong", "weak"}
                and status.get("answer_status") == "verified"
                and status.get("rubric_status") in {"complete", "partial"}
            )
        return bool(
            status.get("answer_status") == "verified"
            and status.get("rubric_status") in {"complete", "partial"}
        )

    def _legacy_strong(self, status: dict[str, Any], purpose: Purpose) -> bool:
        if "strong" in status:
            return bool(status.get("strong"))
        if status.get("needs_image"):
            return bool(
                status.get(f"usable_for_{purpose}", False)
                and status.get("visual_asset_status") == "strong"
                and status.get("readability_status") == "pass"
                and status.get("llm_understanding_status") == "strong"
                and status.get("answer_status") == "verified"
                and status.get("rubric_status") in {"complete", "partial"}
                and not status.get("blocker_reasons")
            )
        return bool(
            status.get(f"usable_for_{purpose}", False)
            and status.get("answer_status") == "verified"
            and status.get("rubric_status") in {"complete", "partial"}
            and not status.get("blocker_reasons")
        )

    def _student_image_contract_blockers(self, status: dict[str, Any]) -> list[str]:
        if not status.get("needs_image"):
            return []
        has_image = bool(
            status.get("display_image_path")
            or status.get("crop_path")
            or status.get("page_image_path")
        )
        has_image_hash = bool(
            status.get("display_image_hash")
            or status.get("crop_hash")
            or status.get("page_image_hash")
        )
        if has_image and has_image_hash:
            return []
        return ["missing_student_image_evidence"]

    def _strong_contract_blockers(self, status: dict[str, Any]) -> list[str]:
        if not status.get("needs_image"):
            return []
        blockers: list[str] = []
        if not status.get("source_file"):
            blockers.append("missing_source_file")
        if status.get("page") is None:
            blockers.append("missing_source_page")
        if not status.get("page_image_hash"):
            blockers.append("missing_page_image_hash")
        has_crop_evidence = bool(status.get("crop_path") and status.get("crop_hash"))
        has_structured_transcript_evidence = status.get("visual_evidence_mode") == "structured_transcript"
        if not (has_crop_evidence or has_structured_transcript_evidence):
            blockers.append("missing_crop_evidence_for_strong")
        if not status.get("vl_model"):
            blockers.append("missing_vl_result")
        return blockers

    def evaluate(self, item: dict[str, Any], purpose: Purpose = "diagnosis") -> QualityDecision:
        if purpose not in PURPOSES:
            return QualityDecision(False, [f"unknown_purpose:{purpose}"], None)
        if purpose == "debug_all":
            return QualityDecision(True, [], None)

        item_id = item.get("item_id", "")
        status = self.status_for(item_id)
        if not status:
            return QualityDecision(False, ["missing_quality_manifest"], None)

        blockers = list(status.get("blocker_reasons") or [])
        needs_image = bool(status.get("needs_image"))
        visual_status = status.get("visual_asset_status")
        llm_status = status.get("llm_understanding_status")
        readability = status.get("readability_status")
        has_new_readability_contract = "student_readable" in status or "strong" in status

        if purpose in {"diagnosis", "profile_evidence"}:
            if not self._legacy_strong(status, purpose):
                blockers.append("not_strong")
            blockers.extend(status.get("strong_blocker_reasons") or [])
            blockers.extend(self._strong_contract_blockers(status))
            blockers.extend(self._student_image_contract_blockers(status))
        elif purpose in {"practice", "teaching"}:
            if not self._legacy_student_readable(status, purpose):
                blockers.append("not_student_readable")
            blockers.extend(self._student_image_contract_blockers(status))

        if not has_new_readability_contract:
            usable_key = f"usable_for_{purpose}"
            if not status.get(usable_key, False):
                blockers.append(f"not_usable_for_{purpose}")

        if purpose in {"diagnosis", "profile_evidence"} and needs_image:
            if visual_status != "strong":
                blockers.append("visual_asset_not_strong")
            if llm_status != "strong":
                blockers.append("llm_understanding_not_strong")
            if readability != "pass":
                blockers.append("readability_not_pass")

        if status.get("answer_status") == "missing":
            blockers.append("answer_missing")
        if purpose in {"diagnosis", "profile_evidence"} and status.get("answer_status") == "suspect":
            blockers.append("answer_suspect")
        if purpose in {"diagnosis", "profile_evidence"} and status.get("rubric_status") == "missing":
            blockers.append("rubric_missing")

        blockers = sorted(set(blockers))
        return QualityDecision(not blockers, blockers, status)


def load_quality_gate(manifest_path: Path | str | None = None) -> ItemQualityGate:
    return ItemQualityGate(manifest_path)

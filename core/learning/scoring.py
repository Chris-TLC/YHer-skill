"""Server-side scoring with conservative deterministic and LLM paths."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from engine.mastery import sanitize_llm_likelihood

from .item_catalog import CatalogItem


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SCALAR_RE = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*(?P<unit>%|[A-Za-z°℃一-鿿][A-Za-z0-9一-鿿°℃·⋅*/^().\-]*)?\s*$"
)
_MCQ_PREFIX_RE = re.compile(r"^(?:答案\s*[:：]?|选择\s*[:：]?|选\s*[:：]?)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedScalar:
    value: float
    unit: str | None


@dataclass(frozen=True)
class ScoreResult:
    status: str
    correct: bool | None
    likelihood: tuple[float, float, float, float]
    update_allowed: bool
    error_code: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "correct": self.correct,
            "likelihood": list(self.likelihood),
            "update_allowed": self.update_allowed,
            "error_code": self.error_code,
            "confidence": self.confidence,
        }


LLMGrader = Callable[[CatalogItem, str], Mapping[str, Any]]


def parse_scalar_answer(value: Any) -> ParsedScalar | None:
    """Parse exactly one finite scalar with an optional attached unit."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    normalized = unicodedata.normalize("NFKC", str(value))
    match = _SCALAR_RE.fullmatch(normalized)
    if not match:
        return None
    number = float(match.group("value"))
    if not math.isfinite(number):
        return None
    unit = _normalize_unit(match.group("unit"))
    return ParsedScalar(number, unit)


def normalize_mcq(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = _MCQ_PREFIX_RE.sub("", normalized).strip()
    normalized = re.sub(r"[\s,;/|\u3001]+", "", normalized).upper()
    normalized = normalized.strip(".:：()[]")
    if not re.fullmatch(r"[A-F]+", normalized):
        return None
    if len(set(normalized)) != len(normalized):
        return None
    return "".join(sorted(normalized))


def score_item(
    item: CatalogItem,
    submission: str,
    llm_grader: LLMGrader | None = None,
) -> ScoreResult:
    """Score one server-held item. The caller never supplies an item or key."""
    if item.scoring_mode == "mcq":
        expected = normalize_mcq(item.answer_values[0]) if len(item.answer_values) == 1 else None
        if expected is None:
            return _invalid_result("invalid_mcq_key")
        observed = normalize_mcq(submission)
        if observed is None:
            return _invalid_result("invalid_mcq_answer")
        correct = observed == expected
        return _deterministic_result(item, correct)
    if item.scoring_mode == "numeric":
        expected = parse_scalar_answer(item.answer_values[0]) if len(item.answer_values) == 1 else None
        observed = parse_scalar_answer(submission)
        correct = _numeric_equal(expected, observed)
        return _deterministic_result(item, correct)
    if item.scoring_mode != "free_llm":
        raise ValueError(f"unsupported scoring mode: {item.scoring_mode}")
    if llm_grader is None:
        return _deferred_result()
    try:
        raw = dict(llm_grader(item, str(submission)))
        confidence = float(raw.get("confidence", 0.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return _deferred_result()
        clean, illegal = sanitize_llm_likelihood(raw.get("likelihood"), confidence)
    except Exception:
        # The LLM boundary includes network, SDK and provider failures. A missing
        # slow path must never block deterministic study or mutate the profile.
        return _deferred_result()
    if illegal or not isinstance(raw.get("correct"), bool):
        return _deferred_result()
    return ScoreResult(
        status="scored",
        correct=raw["correct"],
        likelihood=tuple(float(value) for value in clean),
        update_allowed=True,
        error_code=str(raw.get("error_code") or "") or None,
        confidence=max(0.0, min(1.0, confidence)),
    )


def _deterministic_result(item: CatalogItem, correct: bool) -> ScoreResult:
    from engine import mastery

    engine_type = "numeric" if item.scoring_mode == "numeric" else "mcq"
    probabilities = mastery.local_correct_probs(item.difficulty, engine_type)
    likelihood = probabilities if correct else mastery.likelihood_wrong_binary(probabilities)
    return ScoreResult(
        status="scored",
        correct=bool(correct),
        likelihood=tuple(float(value) for value in likelihood),
        update_allowed=True,
        error_code=None if correct else "incorrect",
    )


def _numeric_equal(expected: ParsedScalar | None, observed: ParsedScalar | None) -> bool:
    if expected is None or observed is None:
        return False
    if expected.unit != observed.unit:
        return False
    absolute_tolerance = max(1e-6, abs(expected.value) * 1e-3)
    return math.isclose(expected.value, observed.value, rel_tol=1e-3, abs_tol=absolute_tolerance)


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return unicodedata.normalize("NFKC", unit).replace(" ", "").replace("⋅", "·").lower()


def _deferred_result() -> ScoreResult:
    return ScoreResult(
        status="deferred",
        correct=None,
        likelihood=(0.25, 0.25, 0.25, 0.25),
        update_allowed=False,
        error_code="llm_unavailable_or_invalid",
        confidence=0.0,
    )


def _invalid_result(error_code: str) -> ScoreResult:
    return ScoreResult(
        status="invalid",
        correct=None,
        likelihood=(0.25, 0.25, 0.25, 0.25),
        update_allowed=False,
        error_code=error_code,
        confidence=0.0,
    )

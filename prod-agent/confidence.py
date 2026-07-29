from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from config import Config
from models import Candidate, ConfidenceDecision, Evidence


@dataclass(frozen=True)
class PracticeCalibration:
    correct: int
    total: int

    def probability(
        self, prior_probability: float = 0.62, prior_weight: float = 4.0
    ) -> float:
        if self.correct < 0 or self.total < 0 or self.correct > self.total:
            raise ValueError("invalid practice calibration counts")
        return (
            self.correct + prior_probability * prior_weight
        ) / (self.total + prior_weight)


def _logit(probability: float) -> float:
    probability = min(0.999, max(0.001, probability))
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def urgent_threshold(config: Config, urgency: float | bool = 1.0) -> float:
    """Interpolate from strong to urgent without escaping configured bounds."""
    urgency = min(1.0, max(0.0, float(urgency)))
    upper = min(1.0, max(0.0, config.strong_confidence_threshold))
    lower = min(
        upper, min(1.0, max(0.0, config.urgent_confidence_floor))
    )
    return min(upper, max(lower, upper - urgency * (upper - lower)))


def _calibrated_probability(
    evidence: Evidence,
    practice: Mapping[
        str, PracticeCalibration | tuple[int, int] | float
    ]
    | PracticeCalibration
    | tuple[int, int]
    | float
    | None,
) -> tuple[float, str]:
    value = practice
    if isinstance(practice, Mapping):
        if "correct" in practice and "total" in practice:
            value = PracticeCalibration(
                int(practice["correct"]), int(practice["total"])
            )
        else:
            value = practice.get(evidence.method)
    if value is None:
        return 0.62, "conservative uncalibrated prior"
    if isinstance(value, PracticeCalibration):
        probability = value.probability()
        return probability, (
            f"practice calibration {value.correct}/{value.total}"
        )
    if isinstance(value, tuple):
        calibration = PracticeCalibration(*value)
        return calibration.probability(), (
            f"practice calibration {calibration.correct}/{calibration.total}"
        )
    if isinstance(value, Mapping):
        calibration = PracticeCalibration(
            int(value.get("correct", 0)), int(value.get("total", 0))
        )
        return calibration.probability(), (
            f"practice calibration {calibration.correct}/{calibration.total}"
        )
    probability = min(0.995, max(0.01, float(value)))
    return probability, f"practice calibration p={probability:.3f}"


class ConfidenceEngine:
    """Scores observable evidence; model self-confidence is intentionally absent."""

    def __init__(
        self,
        config: Config,
        practice: Mapping[
            str, PracticeCalibration | tuple[int, int] | float
        ]
        | PracticeCalibration
        | tuple[int, int]
        | float
        | None = None,
    ):
        self.config = config
        self.practice = practice

    def assess(
        self,
        candidate_or_evidence: Candidate | Evidence,
        *,
        urgent: bool = False,
        urgency: float | None = None,
    ) -> ConfidenceDecision:
        evidence = (
            candidate_or_evidence.evidence
            if isinstance(candidate_or_evidence, Candidate)
            else candidate_or_evidence
        )
        probability, calibration_reason = _calibrated_probability(
            evidence, self.practice
        )
        score = _logit(probability)
        reasons = [calibration_reason]

        deterministic_count = len(evidence.deterministic_checks)
        independent_count = len(evidence.independent_checks)
        score += min(deterministic_count, 3) * 0.72
        score += min(independent_count, 2) * 0.90
        if deterministic_count:
            reasons.append(
                f"{deterministic_count} deterministic check(s)"
            )
        if independent_count:
            reasons.append(f"{independent_count} independent check(s)")

        if evidence.direct_provenance:
            score += 0.45
            reasons.append("answer has direct provenance")
        else:
            score -= 0.80
            reasons.append("answer lacks direct provenance")
        if evidence.input_complete:
            score += 0.35
            reasons.append("inputs complete")
        else:
            score -= 1.20
            reasons.append("inputs incomplete")

        if evidence.assumptions:
            score -= min(len(evidence.assumptions), 4) * 0.60
            reasons.append(f"{len(evidence.assumptions)} assumption(s)")
        if evidence.tool_errors:
            score -= min(len(evidence.tool_errors), 3) * 1.10
            reasons.append(f"{len(evidence.tool_errors)} tool error(s)")

        probability = _sigmoid(score)
        if not deterministic_count and not independent_count:
            probability = min(probability, 0.82)
            reasons.append("capped: no verification checks")
        if not evidence.direct_provenance:
            probability = min(probability, 0.88)
        if not evidence.input_complete:
            probability = min(probability, 0.74)
        if evidence.tool_errors:
            probability = min(probability, 0.60)
        if evidence.assumptions:
            probability = min(probability, 0.92)

        if urgency is None:
            urgency = 1.0 if urgent else 0.0
        threshold = urgent_threshold(self.config, urgency)
        return ConfidenceDecision(
            probability=probability,
            threshold=threshold,
            should_submit=probability >= threshold,
            reasons=tuple(reasons),
        )


def assess_confidence(
    candidate_or_evidence: Candidate | Evidence,
    config: Config,
    *,
    practice: Mapping[
        str, PracticeCalibration | tuple[int, int] | float
    ]
    | PracticeCalibration
    | tuple[int, int]
    | float
    | None = None,
    urgent: bool = False,
    urgency: float | None = None,
) -> ConfidenceDecision:
    return ConfidenceEngine(config, practice).assess(
        candidate_or_evidence, urgent=urgent, urgency=urgency
    )

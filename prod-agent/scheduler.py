from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping

from models import AttemptState, BoardSnapshot, Phase, Priority, Tile


WRONG_PENALTY = 0.25
INITIAL_500_BOOST = 1.15


@dataclass(frozen=True)
class PerformanceEstimate:
    probability: float
    solve_seconds: float


_CATEGORY_DEFAULTS: dict[str, PerformanceEstimate] = {
    "ancient scrolls": PerformanceEstimate(0.72, 82.0),
    "cryptic": PerformanceEstimate(0.76, 68.0),
    "heavy compute": PerformanceEstimate(0.67, 96.0),
    "needle in the haystack": PerformanceEstimate(0.74, 78.0),
    "ship it": PerformanceEstimate(0.73, 76.0),
    "the dark web": PerformanceEstimate(0.69, 92.0),
}
_DEFAULT_CATEGORY = PerformanceEstimate(0.70, 85.0)
_TIER_ADJUSTMENTS: dict[int, tuple[float, float]] = {
    100: (0.15, 0.55),
    200: (0.09, 0.72),
    300: (0.03, 0.90),
    400: (-0.04, 1.10),
    500: (-0.10, 1.30),
}
_RACE_WINDOWS: dict[int, float] = {
    100: 240.0,
    200: 205.0,
    300: 175.0,
    400: 145.0,
    500: 120.0,
}


def _bounded_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _tier(points: int) -> int:
    return min(_TIER_ADJUSTMENTS, key=lambda value: abs(value - points))


def calibrated_default(category: str, points: int) -> PerformanceEstimate:
    """Return the conservative category/tier prior used before practice data."""
    base = _CATEGORY_DEFAULTS.get(category.strip().lower(), _DEFAULT_CATEGORY)
    probability_delta, time_multiplier = _TIER_ADJUSTMENTS[_tier(points)]
    return PerformanceEstimate(
        probability=_bounded_probability(base.probability + probability_delta),
        solve_seconds=max(1.0, base.solve_seconds * time_multiplier),
    )


DEFAULT_CALIBRATION = {
    (category, points): calibrated_default(category, points)
    for category in _CATEGORY_DEFAULTS
    for points in _TIER_ADJUSTMENTS
}


def expected_net_points(
    points: int,
    probability: float,
    wrong_penalty: float = WRONG_PENALTY,
) -> float:
    probability = _bounded_probability(probability)
    return points * (
        probability - wrong_penalty * (1.0 - probability)
    )


def expected_points_per_second(
    points: int,
    probability: float,
    solve_seconds: float,
    *,
    race_survival: float = 1.0,
    wrong_penalty: float = WRONG_PENALTY,
    boost: float = 1.0,
) -> float:
    if solve_seconds <= 0:
        raise ValueError("solve_seconds must be positive")
    return (
        expected_net_points(points, probability, wrong_penalty)
        * min(1.0, max(0.0, race_survival))
        * boost
        / solve_seconds
    )


class Scheduler:
    """Ranks currently actionable tiles by expected net points per second."""

    def __init__(
        self,
        calibration: Mapping[
            tuple[str, int], PerformanceEstimate | tuple[float, float]
        ]
        | None = None,
        *,
        wrong_penalty: float = WRONG_PENALTY,
        initial_500_boost: float = INITIAL_500_BOOST,
        mode: str = "scored",
        clock=time.monotonic,
    ):
        if not 0 <= wrong_penalty <= 1:
            raise ValueError("wrong_penalty must be between zero and one")
        if initial_500_boost < 1:
            raise ValueError("initial_500_boost must be at least one")
        if mode not in {"scored", "practice_eval"}:
            raise ValueError("mode must be scored or practice_eval")
        self._calibration = dict(calibration or {})
        self.wrong_penalty = wrong_penalty
        self.initial_500_boost = initial_500_boost
        self.mode = mode
        self._clock = clock

    def estimate(self, tile: Tile) -> PerformanceEstimate:
        value = self._calibration.get((tile.category, tile.points))
        if value is None:
            value = self._calibration.get((tile.category.lower(), tile.points))
        if value is None:
            return calibrated_default(tile.category, tile.points)
        if isinstance(value, PerformanceEstimate):
            estimate = value
        elif isinstance(value, Mapping):
            estimate = PerformanceEstimate(
                float(
                    value.get(
                        "p_correct", value.get("probability", 0.0)
                    )
                ),
                float(
                    value.get(
                        "solve_seconds", value.get("expected_seconds", 1.0)
                    )
                ),
            )
        else:
            estimate = PerformanceEstimate(*value)
        return PerformanceEstimate(
            _bounded_probability(estimate.probability),
            max(1.0, float(estimate.solve_seconds)),
        )

    def update_calibration(
        self,
        category: str,
        points: int,
        probability: float,
        solve_seconds: float,
    ) -> None:
        """Replace a category/tier estimate as practice evidence arrives."""
        self._calibration[(category, points)] = PerformanceEstimate(
            _bounded_probability(probability),
            max(1.0, float(solve_seconds)),
        )

    @staticmethod
    def _lookup(
        values: Mapping[str | tuple[str, int], float] | None,
        tile: Tile,
        default: float,
    ) -> float:
        if not values:
            return default
        if tile.id in values:
            return float(values[tile.id])
        if (tile.category, tile.points) in values:
            return float(values[(tile.category, tile.points)])
        return default

    def priority(
        self,
        tile: Tile,
        attempt: AttemptState | None = None,
        *,
        probability: float | None = None,
        solve_seconds: float | None = None,
        race_survival: float | None = None,
        now: float | None = None,
        initial_wave: bool = True,
    ) -> Priority:
        now = self._clock() if now is None else now
        attempt = attempt or AttemptState()
        estimate = self.estimate(tile)
        probability = _bounded_probability(
            estimate.probability if probability is None else probability
        )
        solve_seconds = max(
            1.0,
            estimate.solve_seconds if solve_seconds is None else solve_seconds,
        )

        reasons = [
            f"calibrated p_correct={probability:.3f}",
            f"solve={solve_seconds:.1f}s",
        ]
        if attempt.incorrect_attempts:
            probability = _bounded_probability(
                probability * (0.72 ** attempt.incorrect_attempts)
            )
            solve_seconds *= 1.0 + 0.15 * attempt.incorrect_attempts
            reasons.append(
                f"{attempt.incorrect_attempts} prior incorrect attempt(s)"
            )

        cooldown_remaining = max(0.0, attempt.cooldown_until - now)
        if cooldown_remaining:
            reasons.append(f"cooldown={cooldown_remaining:.1f}s")

        if race_survival is None:
            tier = _tier(tile.points)
            depletion = math.sqrt(
                max(1.0, tile.total) / max(1.0, tile.remaining)
            )
            race_survival = math.exp(
                -(solve_seconds + cooldown_remaining)
                * depletion
                / _RACE_WINDOWS[tier]
            )
        race_survival = min(1.0, max(0.0, float(race_survival)))

        net_value = expected_net_points(
            tile.points, probability, self.wrong_penalty
        )
        expected_seconds = solve_seconds + cooldown_remaining
        boost = 1.0
        if (
            initial_wave
            and tile.points == 500
            and attempt.attempts == 0
            and attempt.incorrect_attempts == 0
        ):
            boost = self.initial_500_boost
            reasons.append(f"initial 500 boost x{boost:.2f}")
        score = expected_points_per_second(
            tile.points,
            probability,
            expected_seconds,
            race_survival=race_survival,
            wrong_penalty=self.wrong_penalty,
            boost=boost,
        )
        if cooldown_remaining:
            score = 0.0
        reasons.extend(
            (
                f"net={net_value:.1f}",
                f"race_survival={race_survival:.3f}",
            )
        )
        return Priority(
            task_id=tile.id,
            score=score,
            probability=probability,
            expected_seconds=expected_seconds,
            race_survival=race_survival,
            net_value=net_value,
            reasons=tuple(reasons),
        )

    def rank(
        self,
        tiles: list[Tile] | tuple[Tile, ...],
        attempts: Mapping[str, AttemptState] | None = None,
        *,
        probabilities: Mapping[str | tuple[str, int], float] | None = None,
        solve_seconds: Mapping[str | tuple[str, int], float] | None = None,
        race_survival: Mapping[str | tuple[str, int], float] | None = None,
        now: float | None = None,
        initial_wave: bool = True,
        include_cooldown: bool = False,
    ) -> list[Priority]:
        now = self._clock() if now is None else now
        attempts = attempts or {}
        priorities: list[Priority] = []
        for tile in tiles:
            attempt = attempts.get(tile.id, AttemptState())
            if attempt.cooldown_until > now and not include_cooldown:
                continue
            estimate = self.estimate(tile)
            priorities.append(
                self.priority(
                    tile,
                    attempt,
                    probability=self._lookup(
                        probabilities, tile, estimate.probability
                    ),
                    solve_seconds=self._lookup(
                        solve_seconds, tile, estimate.solve_seconds
                    ),
                    race_survival=(
                        None
                        if race_survival is None
                        else self._lookup(race_survival, tile, 1.0)
                    ),
                    now=now,
                    initial_wave=initial_wave,
                )
            )
        return sorted(
            priorities,
            key=lambda item: (-item.score, item.expected_seconds, item.task_id),
        )

    def rank_board(
        self,
        board: BoardSnapshot,
        attempts: Mapping[str, AttemptState] | None = None,
        **kwargs,
    ) -> list[Priority]:
        """Rank a live board while keeping practice explicitly opt-in."""
        if self.mode == "practice_eval":
            if board.phase is not Phase.PRACTICE:
                raise ValueError(
                    "practice_eval mode only accepts the practice board"
                )
        elif board.phase not in {Phase.ROUND1, Phase.GAME}:
            raise ValueError(
                "scored mode requires a currently playable scored board"
            )
        return self.rank(board.tiles, attempts, **kwargs)


def rank_tiles(
    tiles: list[Tile] | tuple[Tile, ...],
    attempts: Mapping[str, AttemptState] | None = None,
    **kwargs,
) -> list[Priority]:
    return Scheduler().rank(tiles, attempts, **kwargs)

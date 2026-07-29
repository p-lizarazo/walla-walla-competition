from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    SETUP = "setup"
    PRACTICE = "practice"
    ROUND1 = "round1"
    GAME = "game"
    ENDED = "ended"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | None) -> "Phase":
        try:
            return cls(value or "")
        except ValueError:
            return cls.UNKNOWN


class SubmissionResult(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ALREADY_CLAIMED = "already_claimed"
    LOCKED_OUT = "locked_out"
    RATE_LIMITED = "rate_limited"
    LOCKED = "locked"
    WRONG_PHASE = "wrong_phase"
    FORBIDDEN = "forbidden"
    VOIDED = "voided"
    UNKNOWN_TASK = "unknown_task"


@dataclass(frozen=True)
class Tile:
    id: str
    category: str
    points: int
    answer_format: str = "exact"
    remaining: int = 1
    total: int = 1
    locked: bool = False


@dataclass(frozen=True)
class TaskDetail:
    id: str
    category: str
    title: str
    points: int
    prompt: str
    files: tuple[str, ...]
    answer_format: str
    claimed: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TaskDetail":
        return cls(
            id=str(payload["id"]),
            category=str(payload.get("category") or ""),
            title=str(payload.get("title") or ""),
            points=int(payload.get("points") or 0),
            prompt=str(payload.get("prompt") or ""),
            files=tuple(payload.get("files") or ()),
            answer_format=str(payload.get("answer_format") or "exact"),
            claimed=bool(payload.get("claimed")),
        )


@dataclass(frozen=True)
class BoardSnapshot:
    phase: Phase
    tiles: tuple[Tile, ...]
    solved_ids: frozenset[str]
    server_time: str | None
    fetched_monotonic: float
    raw_you: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DashboardSnapshot:
    payload: dict[str, Any]
    fetched_monotonic: float


@dataclass(frozen=True)
class Evidence:
    method: str
    deterministic_checks: tuple[str, ...] = ()
    independent_checks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    tool_errors: tuple[str, ...] = ()
    input_complete: bool = True
    direct_provenance: bool = True


@dataclass(frozen=True)
class Candidate:
    task_id: str
    answer: str
    answer_sha256: str
    evidence: Evidence
    model_temperature: float
    elapsed_seconds: float
    tool_turns: int


@dataclass(frozen=True)
class ConfidenceDecision:
    probability: float
    threshold: float
    should_submit: bool
    reasons: tuple[str, ...]


@dataclass
class AttemptState:
    attempts: int = 0
    incorrect_attempts: int = 0
    submitted_hashes: set[str] = field(default_factory=set)
    cooldown_until: float = 0.0
    last_method: str | None = None


@dataclass(frozen=True)
class Priority:
    task_id: str
    score: float
    probability: float
    expected_seconds: float
    race_survival: float
    net_value: float
    reasons: tuple[str, ...]

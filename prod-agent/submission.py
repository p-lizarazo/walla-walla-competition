from __future__ import annotations

import hashlib
import hmac
import inspect
import threading
import time
from typing import Any, Callable

from config import Config
from game_client import GameClient
from models import (
    AttemptState,
    BoardSnapshot,
    Candidate,
    ConfidenceDecision,
    Phase,
)


class SubmissionRejected(RuntimeError):
    pass


class ConfidenceRejected(SubmissionRejected):
    pass


class DuplicateAnswer(SubmissionRejected):
    pass


class CooldownActive(SubmissionRejected):
    pass


class TileNotOpen(SubmissionRejected):
    pass


class SubmissionLane:
    """The only component allowed to turn a verified Candidate into a POST."""

    def __init__(
        self,
        client: GameClient,
        config: Config,
        board_check: Callable[..., bool | BoardSnapshot | dict[str, Any]],
        *,
        clock=time.monotonic,
        sleep=time.sleep,
        max_rate_limit_retries: int = 3,
        board_max_age_seconds: float | None = None,
    ):
        if max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries cannot be negative")
        self.client = client
        self.config = config
        self._board_check = board_check
        self._clock = clock
        self._sleep = sleep
        self._max_rate_limit_retries = max_rate_limit_retries
        self._board_max_age = (
            max(config.board_poll_seconds * 2, config.submission_interval_seconds)
            if board_max_age_seconds is None
            else board_max_age_seconds
        )
        self._lock = threading.RLock()
        self._next_submission_at = 0.0
        self._attempts: dict[str, AttemptState] = {}

    def state(self, task_id: str) -> AttemptState:
        with self._lock:
            state = self._attempts.get(task_id, AttemptState())
            return AttemptState(
                attempts=state.attempts,
                incorrect_attempts=state.incorrect_attempts,
                submitted_hashes=set(state.submitted_hashes),
                cooldown_until=state.cooldown_until,
                last_method=state.last_method,
            )

    def states(self) -> dict[str, AttemptState]:
        with self._lock:
            return {task_id: self.state(task_id) for task_id in self._attempts}

    def _call_board_check(self, task_id: str) -> Any:
        try:
            parameters = inspect.signature(self._board_check).parameters
        except (TypeError, ValueError):
            parameters = {"task_id": None}
        if parameters:
            return self._board_check(task_id)
        return self._board_check()

    def _is_open(self, task_id: str) -> bool:
        status = self._call_board_check(task_id)
        if isinstance(status, bool):
            return status
        if isinstance(status, BoardSnapshot):
            return self._is_snapshot_open(task_id, status)
        if isinstance(status, dict):
            if "snapshot" in status:
                return self._snapshot_dict_open(task_id, status["snapshot"])
            return bool(status.get("fresh")) and bool(status.get("open"))
        if isinstance(status, tuple) and len(status) == 2:
            return bool(status[0]) and bool(status[1])
        return False

    def _snapshot_dict_open(self, task_id: str, snapshot: Any) -> bool:
        if isinstance(snapshot, BoardSnapshot):
            return self._is_snapshot_open(task_id, snapshot)
        return False

    def _is_snapshot_open(
        self, task_id: str, snapshot: BoardSnapshot
    ) -> bool:
        mode = getattr(self.config, "mode", "scored")
        if mode == "practice_eval":
            playable = snapshot.phase is Phase.PRACTICE
        else:
            playable = snapshot.phase in {Phase.ROUND1, Phase.GAME}
        already_solved = (
            task_id in snapshot.solved_ids and mode != "practice_eval"
        )
        return (
            self._clock() - snapshot.fetched_monotonic <= self._board_max_age
            and playable
            and not already_solved
            and any(tile.id == task_id for tile in snapshot.tiles)
        )

    def _wait_for_lane(self, extra_delay: float = 0.0) -> None:
        target = max(
            self._next_submission_at,
            self._clock() + max(0.0, extra_delay),
        )
        remaining = target - self._clock()
        if remaining > 0:
            self._sleep(remaining)

    @staticmethod
    def _verified_hash(candidate: Candidate) -> str:
        if (
            not candidate.answer.strip()
            or "\n" in candidate.answer
            or len(candidate.answer) > 1000
        ):
            raise SubmissionRejected(
                "candidate answer must be one nonempty line of at most "
                "1000 characters"
            )
        digest = hashlib.sha256(candidate.answer.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(digest, candidate.answer_sha256):
            raise SubmissionRejected("candidate answer hash does not match")
        return digest

    def _preflight(
        self,
        candidate: Candidate,
        decision: ConfidenceDecision | None,
    ) -> tuple[AttemptState, str]:
        if not isinstance(candidate, Candidate):
            raise TypeError("submission requires a verified Candidate")
        if decision is not None and not decision.should_submit:
            raise ConfidenceRejected(
                f"confidence {decision.probability:.3f} is below "
                f"{decision.threshold:.3f}"
            )
        digest = self._verified_hash(candidate)
        state = self._attempts.setdefault(candidate.task_id, AttemptState())
        now = self._clock()
        if digest in state.submitted_hashes:
            raise DuplicateAnswer(
                f"answer hash already submitted for {candidate.task_id}"
            )
        if state.cooldown_until > now:
            raise CooldownActive(
                f"{candidate.task_id} cooldown has "
                f"{state.cooldown_until - now:.1f}s remaining"
            )
        return state, digest

    def _record_result(
        self,
        state: AttemptState,
        candidate: Candidate,
        digest: str,
        payload: dict[str, Any],
    ) -> None:
        result = str(payload.get("result") or "")
        if result == "rate_limited":
            return
        state.attempts += 1
        state.submitted_hashes.add(digest)
        state.last_method = candidate.evidence.method
        if result == "incorrect":
            state.incorrect_attempts += 1
            cooldown = min(
                480.0, 30.0 * (2 ** (state.incorrect_attempts - 1))
            )
            cooldown = max(
                cooldown,
                min(480.0, float(payload.get("retry_in") or 0.0)),
            )
            state.cooldown_until = self._clock() + cooldown
        elif result == "locked_out":
            retry_in = max(0.0, float(payload.get("retry_in") or 0.0))
            state.cooldown_until = max(
                state.cooldown_until, self._clock() + retry_in
            )
        elif result == "correct":
            state.cooldown_until = 0.0

    def submit(
        self,
        candidate: Candidate,
        decision: ConfidenceDecision | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state, digest = self._preflight(candidate, decision)
            retry_delay = 0.0
            for retry in range(self._max_rate_limit_retries + 1):
                self._wait_for_lane(retry_delay)
                if not self._is_open(candidate.task_id):
                    raise TileNotOpen(
                        f"{candidate.task_id} is stale, solved, or not open"
                    )
                try:
                    payload = self.client.submit(
                        candidate.task_id, candidate.answer
                    )
                finally:
                    self._next_submission_at = (
                        self._clock()
                        + self.config.submission_interval_seconds
                    )
                if not isinstance(payload, dict) or "result" not in payload:
                    raise SubmissionRejected(
                        "submission response has no result"
                    )
                if payload.get("result") != "rate_limited":
                    self._record_result(state, candidate, digest, payload)
                    return payload
                if retry == self._max_rate_limit_retries:
                    return payload
                retry_delay = max(
                    0.0, float(payload.get("retry_in") or 0.0)
                )
            raise AssertionError("unreachable")

    def try_submit(
        self,
        candidate: Candidate,
        decision: ConfidenceDecision | None = None,
    ) -> dict[str, Any]:
        try:
            return self.submit(candidate, decision)
        except SubmissionRejected as error:
            return {
                "result": "rejected",
                "submitted": False,
                "reason": str(error),
                "rejection": type(error).__name__,
            }

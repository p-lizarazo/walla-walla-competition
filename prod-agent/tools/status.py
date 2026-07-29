from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
import threading
from typing import Any, Callable, Generic, Mapping, TypeVar


class StatusProviderError(ValueError):
    pass


T = TypeVar("T")


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise StatusProviderError(f"unsupported status value: {type(value).__name__}")


_SECRET_WORDS = (
    "key",
    "token",
    "secret",
    "credential",
    "password",
    "authorization",
    "cookie",
)

_PLAYABLE_BOARDS = {
    "practice": "practice",
    "round1": "qual",
    "game": "main",
}


def playable_board_for_phase(
    phase: str | Enum | None,
    *,
    evaluation_mode: bool = False,
    available_boards: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the server board for a phase without introducing round strategy."""
    if not isinstance(evaluation_mode, bool):
        raise StatusProviderError("evaluation_mode must be a boolean")
    if isinstance(phase, Enum):
        phase = str(phase.value)
    phase_name = str(phase or "").lower()
    if phase_name == "practice" and not evaluation_mode:
        return None
    board = _PLAYABLE_BOARDS.get(phase_name)
    if board is None:
        return None
    if available_boards is not None and board not in available_boards:
        return None
    return board


def _safe_dashboard(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_dashboard(item)
            for key, item in value.items()
            if not any(word in str(key).lower() for word in _SECRET_WORDS)
        }
    if isinstance(value, list):
        return [_safe_dashboard(item) for item in value]
    return value


class _ReadOnlyProvider(Generic[T]):
    def __init__(
        self,
        *,
        callback: Callable[[], T] | None,
        snapshot: T | None,
    ) -> None:
        if (callback is None) == (snapshot is None):
            raise StatusProviderError("provide exactly one callback or snapshot")
        if callback is not None and not callable(callback):
            raise StatusProviderError("callback must be callable")
        self._callback = callback
        self._snapshot = deepcopy(snapshot)
        self._lock = threading.Lock()

    def _read_source(self) -> T:
        if self._callback is not None:
            with self._lock:
                return self._callback()
        return deepcopy(self._snapshot)


class ProblemStatusProvider(_ReadOnlyProvider[Any]):
    """Expose task details from either a live read callback or a fixed snapshot."""

    def __init__(
        self,
        callback: Callable[[], Any] | None = None,
        snapshot: Any | None = None,
    ) -> None:
        super().__init__(callback=callback, snapshot=snapshot)

    def read(self) -> dict[str, Any]:
        value = _plain(self._read_source())
        if not isinstance(value, dict):
            raise StatusProviderError("problem status must be an object")
        return deepcopy(value)

    get = read
    snapshot = read
    __call__ = read


class GameStatusProvider(_ReadOnlyProvider[Any]):
    """Expose current phase and playable board using one scored-round policy."""

    def __init__(
        self,
        callback: Callable[[], Any] | None = None,
        snapshot: Any | None = None,
        *,
        evaluation_mode: bool = False,
    ) -> None:
        if not isinstance(evaluation_mode, bool):
            raise StatusProviderError("evaluation_mode must be a boolean")
        super().__init__(callback=callback, snapshot=snapshot)
        self.evaluation_mode = evaluation_mode

    def read(self) -> dict[str, Any]:
        value = _plain(self._read_source())
        if not isinstance(value, dict):
            raise StatusProviderError("game status must be an object")
        phase = str(value.get("phase") or "unknown").lower()
        boards = value.get("boards")
        available = boards if isinstance(boards, Mapping) else None
        board = playable_board_for_phase(
            phase,
            evaluation_mode=self.evaluation_mode,
            available_boards=available,
        )
        status: dict[str, Any] = {
            "phase": phase,
            "playable_board": board,
            "evaluation_mode": self.evaluation_mode,
            "competition_mode": (
                "practice_eval"
                if phase == "practice" and self.evaluation_mode
                else "scored"
                if phase in {"round1", "game"}
                else "inactive"
            ),
        }
        if "server_time" in value:
            status["server_time"] = value["server_time"]
        if "fetched_monotonic" in value:
            status["fetched_monotonic"] = value["fetched_monotonic"]
        tiles = value.get("tiles")
        if isinstance(tiles, list):
            status["open_tile_count"] = len(tiles)
        solved = value.get("solved_ids")
        if isinstance(solved, list):
            status["solved_count"] = len(solved)
        return status

    get = read
    snapshot = read
    __call__ = read


BoardStatusProvider = GameStatusProvider


class DashboardStatusProvider(_ReadOnlyProvider[Any]):
    """Expose a credential-scrubbed dashboard snapshot without mutation methods."""

    def __init__(
        self,
        callback: Callable[[], Any] | None = None,
        snapshot: Any | None = None,
    ) -> None:
        super().__init__(callback=callback, snapshot=snapshot)

    def read(self) -> dict[str, Any]:
        value = _plain(self._read_source())
        if not isinstance(value, dict):
            raise StatusProviderError("dashboard status must be an object")
        return deepcopy(_safe_dashboard(value))

    get = read
    snapshot = read
    __call__ = read

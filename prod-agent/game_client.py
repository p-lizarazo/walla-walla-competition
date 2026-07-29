from __future__ import annotations

import pathlib
import tempfile
import threading
import time
from typing import Any

import requests

from config import Config
from models import BoardSnapshot, DashboardSnapshot, Phase, TaskDetail, Tile


class GameError(RuntimeError):
    pass


class AuthError(GameError):
    pass


class TileUnavailable(GameError):
    pass


_PHASE_BOARD = {
    Phase.PRACTICE: "practice",
    Phase.ROUND1: "qual",
    Phase.GAME: "main",
}

_SECRET_FIELDS = {
    "api_key",
    "team_api_key",
    "anthropic_api_key",
    "access_token",
    "refresh_token",
    "token",
    "authorization",
    "cookie",
    "password",
    "secret",
    "credential",
}


def _secret_field(name: str) -> bool:
    lowered = name.lower()
    return lowered in _SECRET_FIELDS or lowered.endswith(
        ("_api_key", "_password", "_secret", "_credential")
    )


class GameClient:
    def __init__(self, config: Config):
        self.config = config
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["X-Api-Key"] = self.config.team_api_key
            self._local.session = session
        return session

    @staticmethod
    def _detail(response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                payload = payload.get("detail") or payload
            return str(payload)[:300]
        except ValueError:
            return (response.text or "").strip()[:300]

    def _json(
        self,
        response: requests.Response,
        what: str,
        unavailable: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        if response.status_code == 401:
            raise AuthError(f"{what}: API key was rejected")
        if response.status_code in unavailable:
            raise TileUnavailable(
                f"{what}: HTTP {response.status_code} - {self._detail(response)}"
            )
        if not response.ok:
            raise GameError(
                f"{what}: HTTP {response.status_code} - {self._detail(response)}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise GameError(f"{what}: expected JSON") from error
        if not isinstance(payload, dict):
            raise GameError(f"{what}: expected a JSON object")
        return payload

    def board(self) -> BoardSnapshot:
        payload = self._json(
            self._session().get(f"{self.config.base_url}/api/board", timeout=30),
            "GET /api/board",
        )
        if "you" not in payload:
            raise AuthError("GET /api/board returned no authenticated `you` block")
        phase = Phase.parse(payload.get("phase"))
        board_name = _PHASE_BOARD.get(phase)
        solved = frozenset((payload.get("you") or {}).get("solved_ids") or ())
        include_solved_practice = (
            self.config.mode == "practice_eval" and phase is Phase.PRACTICE
        )
        tiles: list[Tile] = []
        if board_name:
            for category in (payload.get("boards") or {}).get(board_name, ()):
                category_name = str(category.get("name") or "")
                for cell in category.get("tiles") or ():
                    if cell.get("locked"):
                        continue
                    ids = cell.get("open_ids")
                    if ids is None:
                        ids = [] if cell.get("claimed_by") else [cell.get("id")]
                    for task_id in ids:
                        if (
                            not task_id
                            or (task_id in solved and not include_solved_practice)
                        ):
                            continue
                        tiles.append(
                            Tile(
                                id=str(task_id),
                                category=category_name,
                                points=int(cell.get("points") or 0),
                                remaining=int(cell.get("remaining") or len(ids)),
                                total=int(cell.get("total") or len(ids)),
                                locked=bool(cell.get("locked")),
                            )
                        )
        tiles.sort(key=lambda tile: (-tile.points, tile.category, tile.id))
        return BoardSnapshot(
            phase=phase,
            tiles=tuple(tiles),
            solved_ids=solved,
            server_time=payload.get("server_time"),
            fetched_monotonic=time.monotonic(),
            raw_you=dict(payload.get("you") or {}),
        )

    def dashboard(self) -> DashboardSnapshot:
        payload = self._json(
            self._session().get(f"{self.config.base_url}/api/me", timeout=30),
            "GET /api/me",
        )
        safe = {
            key: value
            for key, value in payload.items()
            if not _secret_field(key)
        }
        return DashboardSnapshot(safe, time.monotonic())

    def task(self, task_id: str) -> TaskDetail:
        payload = self._json(
            self._session().get(
                f"{self.config.base_url}/api/task/{task_id}", timeout=30
            ),
            f"GET /api/task/{task_id}",
            unavailable=(403, 404),
        )
        return TaskDetail.from_payload(payload)

    @staticmethod
    def workdir(task_id: str) -> pathlib.Path:
        path = pathlib.Path(tempfile.gettempdir()) / f"jeopardy_{task_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def fetch_files(self, task: TaskDetail) -> pathlib.Path:
        workdir = self.workdir(task.id).resolve()
        for name in task.files:
            path = (workdir / name).resolve()
            if workdir not in path.parents:
                raise GameError(f"unsafe task filename: {name}")
            if path.exists() and path.stat().st_size:
                continue
            response = self._session().get(
                f"{self.config.base_url}/api/task/{task.id}/file/{name}",
                timeout=120,
            )
            if response.status_code == 401:
                raise AuthError(f"file download for {task.id}: key rejected")
            if response.status_code in (403, 404):
                raise TileUnavailable(f"file download for {task.id} unavailable")
            if not response.ok:
                raise GameError(
                    f"file download for {task.id}: HTTP {response.status_code}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return workdir

    def submit(self, task_id: str, answer: str) -> dict[str, Any]:
        payload = self._json(
            self._session().post(
                f"{self.config.base_url}/api/submit",
                json={"task_id": task_id, "answer": str(answer)},
                timeout=30,
            ),
            f"POST /api/submit {task_id}",
        )
        if "result" not in payload:
            raise GameError(f"POST /api/submit {task_id}: no result field")
        return payload

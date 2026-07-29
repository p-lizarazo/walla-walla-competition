from __future__ import annotations

import json
import pathlib
import threading
import time
from typing import Any


class PracticeAttemptLog:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self._lock = threading.Lock()

    def append(self, **record: Any) -> None:
        safe = {
            key: value
            for key, value in record.items()
            if key not in {"answer", "password", "cookie", "team_api_key"}
        }
        safe["timestamp"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(safe, sort_keys=True, default=str) + "\n")

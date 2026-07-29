from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_PATH = Path(__file__).with_name("playbooks.json")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _tier(value: int | str | None) -> int:
    number = int(value or 100)
    return min(5, max(1, number // 100 if number > 5 else number))


class PlaybookLoader:
    """Load and select short, reusable solver methods."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.data, dict):
            raise ValueError("playbooks must be a JSON object")

    def select(
        self,
        category: str,
        tier: int | str | None = None,
        task_signature: str | Iterable[str] | None = None,
        *,
        points: int | None = None,
        prompt: str = "",
        title: str = "",
        files: Iterable[str] = (),
        answer_format: str = "",
    ) -> tuple[str, ...]:
        selected_tier = _tier(points if points is not None else tier)
        category_key = _slug(category)
        explicit = (
            {_slug(task_signature)}
            if isinstance(task_signature, str)
            else {_slug(str(item)) for item in (task_signature or ())}
        )
        text = " ".join(
            (title, prompt, answer_format, *(str(name) for name in files))
        ).lower()

        methods: list[str] = list(self.data.get("generic", ()))
        methods.extend(self.data.get("tiers", {}).get(str(selected_tier), ()))
        methods.extend(self.data.get("categories", {}).get(category_key, ()))

        for signature in self.data.get("signatures", ()):
            signature_id = _slug(str(signature.get("id", "")))
            signature_category = _slug(str(signature.get("category", "")))
            keywords = tuple(str(item).lower() for item in signature.get("keywords", ()))
            if signature_category and signature_category != category_key:
                continue
            if signature_id in explicit or any(keyword in text for keyword in keywords):
                methods.extend(signature.get("methods", ()))

        return tuple(dict.fromkeys(str(method).strip() for method in methods if method))

    methods_for = select

    def select_for_task(
        self,
        task: Any,
        task_signature: str | Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        return self.select(
            str(getattr(task, "category", "")),
            task_signature=task_signature,
            points=int(getattr(task, "points", 0) or 0),
            prompt=str(getattr(task, "prompt", "") or ""),
            title=str(getattr(task, "title", "") or ""),
            files=tuple(getattr(task, "files", ()) or ()),
            answer_format=str(getattr(task, "answer_format", "") or ""),
        )


PlaybookLibrary = PlaybookLoader


def load_playbooks(path: str | Path = DEFAULT_PATH) -> PlaybookLoader:
    return PlaybookLoader(path)


def select_methods(
    category: str,
    tier: int | str | None = None,
    task_signature: str | Iterable[str] | None = None,
    *,
    path: str | Path = DEFAULT_PATH,
    **context: Any,
) -> tuple[str, ...]:
    return PlaybookLoader(path).select(
        category, tier, task_signature, **context
    )

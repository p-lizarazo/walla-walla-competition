from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
import sys
import time
from typing import Any
from urllib.parse import quote

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_paths import FastPathSolver
from models import TaskDetail
from tools.web import EventWebSession, WebResponse


SCHEMA_VERSION = 1


class RecordedWebSession:
    def __init__(self, path: pathlib.Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.responses = dict(payload.get("responses") or {})

    def _response(
        self,
        method: str,
        url: str,
        *,
        form: dict[str, str] | None = None,
    ) -> WebResponse:
        key = f"{method.upper()} {url}"
        record = self.responses.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"recorded web response not found: {key}")
        expected_form = record.get("expected_form")
        if expected_form is not None and dict(form or {}) != dict(expected_form):
            raise ValueError(f"recorded web form mismatch: {key}")
        headers = tuple(
            (str(name), str(value))
            for name, value in (record.get("headers") or {}).items()
        )
        return WebResponse(
            int(record.get("status") or 200),
            str(record.get("url") or url),
            headers,
            str(record.get("body") or "").encode("utf-8"),
            False,
        )

    def get(self, url: str) -> WebResponse:
        return self._response("GET", url)

    def post(self, url: str, *, form: dict[str, str]) -> WebResponse:
        return self._response("POST", url, form=form)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def stable_split(
    task_ids: list[str],
    *,
    test_fraction: float,
    seed: str,
) -> dict[str, str]:
    if not 0 <= test_fraction <= 1:
        raise ValueError("test_fraction must be between zero and one")
    if not task_ids:
        return {}
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task ids must be unique within a cell")
    if len(task_ids) == 1:
        return {
            task_ids[0]: "test" if test_fraction >= 0.5 else "train"
        }
    test_count = round(len(task_ids) * test_fraction)
    test_count = min(len(task_ids) - 1, max(1, test_count))
    ordered = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{seed}\0{task_id}".encode("utf-8")
        ).digest(),
    )
    test_ids = set(ordered[:test_count])
    return {
        task_id: "test" if task_id in test_ids else "train"
        for task_id in task_ids
    }


def safe_relative_path(name: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe task filename: {name!r}")
    return path


def category_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "uncategorized"


def board_items(
    payload: dict[str, Any],
    board_name: str,
    *,
    split_policy: str,
    test_fraction: float,
    seed: str,
) -> list[dict[str, Any]]:
    boards = payload.get("boards") or {}
    if board_name not in boards:
        available = ", ".join(sorted(boards)) or "none"
        raise ValueError(
            f"board {board_name!r} is unavailable; available boards: {available}"
        )
    items: list[dict[str, Any]] = []
    for category_index, category in enumerate(boards[board_name]):
        category_name = str(category.get("name") or "")
        for cell_index, cell in enumerate(category.get("tiles") or ()):
            variants = cell.get("variants") or ()
            task_ids = [
                str(variant.get("id"))
                for variant in variants
                if variant.get("id")
            ]
            if not task_ids:
                task_ids = [
                    str(task_id)
                    for task_id in (cell.get("open_ids") or ())
                    if task_id
                ]
            if not task_ids and cell.get("id"):
                task_ids = [str(cell["id"])]
            if split_policy == "all-train":
                splits = {task_id: "train" for task_id in task_ids}
            elif split_policy == "all-test":
                splits = {task_id: "test" for task_id in task_ids}
            else:
                splits = stable_split(
                    task_ids,
                    test_fraction=test_fraction,
                    seed=seed,
                )
            cell_key = (
                f"{board_name}:{category_index}:{cell_index}:"
                f"{category_name}:{int(cell.get('points') or 0)}:"
                f"{str(cell.get('title') or '')}"
            )
            template_key = (
                f"{category_name}\0{int(cell.get('points') or 0)}\0"
                f"{str(cell.get('title') or '')}"
            )
            for variant_index, task_id in enumerate(task_ids):
                items.append(
                    {
                        "id": task_id,
                        "board": board_name,
                        "category": category_name,
                        "points": int(cell.get("points") or 0),
                        "title": str(cell.get("title") or ""),
                        "cell_key": cell_key,
                        "template_key": hashlib.sha256(
                            template_key.encode("utf-8")
                        ).hexdigest(),
                        "variant_index": variant_index,
                        "split": splits[task_id],
                    }
                )
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("board contains duplicate task ids")
    return items


def _credentials() -> tuple[str, str]:
    base_url = os.environ.get("JEOPARDY_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("TEAM_API_KEY", "")
    if not base_url or not api_key:
        raise ValueError("JEOPARDY_BASE_URL and TEAM_API_KEY are required")
    return base_url, api_key


def _get_json(
    url: str,
    api_key: str,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = requests.get(
        url,
        headers={"X-Api-Key": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return payload


def _pull_task(
    item: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    output: pathlib.Path,
    download_files: bool,
) -> dict[str, Any]:
    task_id = str(item["id"])
    started = time.monotonic()
    payload = _get_json(
        f"{base_url}/api/task/{quote(task_id, safe='')}",
        api_key,
    )
    task_dir = output / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=False)
    file_records: list[dict[str, Any]] = []
    total_bytes = 0
    if download_files:
        for name in payload.get("files") or ():
            relative = safe_relative_path(str(name))
            response = requests.get(
                (
                    f"{base_url}/api/task/{quote(task_id, safe='')}/file/"
                    f"{quote(str(relative), safe='/')}"
                ),
                headers={"X-Api-Key": api_key},
                timeout=120,
            )
            response.raise_for_status()
            path = task_dir / pathlib.Path(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            digest = hashlib.sha256(response.content).hexdigest()
            total_bytes += len(response.content)
            file_records.append(
                {
                    "name": str(relative),
                    "bytes": len(response.content),
                    "sha256": digest,
                }
            )
    task_path = task_dir / "task.json"
    task_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prompt = str(payload.get("prompt") or "")
    return {
        **item,
        "answer_format": str(payload.get("answer_format") or "exact"),
        "claimed": bool(payload.get("claimed")),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "task_path": str(task_path.relative_to(output)),
        "files": file_records,
        "download_bytes": total_bytes,
        "pull_seconds": round(time.monotonic() - started, 6),
    }


def pull_dataset(arguments: argparse.Namespace) -> int:
    base_url, api_key = _credentials()
    output = pathlib.Path(arguments.output).resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.mkdir(parents=True)
    try:
        board = _get_json(f"{base_url}/api/board", api_key)
        items = board_items(
            board,
            arguments.board,
            split_policy=arguments.split_policy,
            test_fraction=arguments.test_fraction,
            seed=arguments.seed,
        )
        with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
            records = list(
                pool.map(
                    lambda item: _pull_task(
                        item,
                        base_url=base_url,
                        api_key=api_key,
                        output=output,
                        download_files=not arguments.no_files,
                    ),
                    items,
                )
            )
        records.sort(key=lambda item: item["id"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "base_url": base_url,
                "board": arguments.board,
                "phase": str(board.get("phase") or ""),
                "server_time": board.get("server_time"),
            },
            "split": {
                "policy": arguments.split_policy,
                "test_fraction": arguments.test_fraction,
                "seed": arguments.seed,
            },
            "download_files": not arguments.no_files,
            "tasks": records,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for split in ("train", "test"):
            ids = [
                record["id"]
                for record in records
                if record["split"] == split
            ]
            (output / f"{split}_ids.txt").write_text(
                "".join(f"{task_id}\n" for task_id in ids),
                encoding="utf-8",
            )
        categories: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            categories.setdefault(str(record["category"]), []).append(record)
        category_index: list[dict[str, Any]] = []
        for category_name, category_records in sorted(categories.items()):
            slug = category_slug(category_name)
            category_dir = output / "categories" / slug
            category_dir.mkdir(parents=True)
            templates: dict[str, dict[str, Any]] = {}
            for record in category_records:
                template = templates.setdefault(
                    str(record["template_key"]),
                    {
                        "template_key": record["template_key"],
                        "category": category_name,
                        "points": record["points"],
                        "title": record["title"],
                        "train_ids": [],
                        "test_ids": [],
                    },
                )
                template[f"{record['split']}_ids"].append(record["id"])
            category_manifest = {
                "schema_version": SCHEMA_VERSION,
                "dataset_root": "../..",
                "category": category_name,
                "tasks": category_records,
            }
            (category_dir / "manifest.json").write_text(
                json.dumps(category_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (category_dir / "templates.json").write_text(
                json.dumps(
                    sorted(
                        templates.values(),
                        key=lambda item: (item["points"], item["title"]),
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            for split in ("train", "test"):
                ids = [
                    record["id"]
                    for record in category_records
                    if record["split"] == split
                ]
                (category_dir / f"{split}_ids.txt").write_text(
                    "".join(f"{task_id}\n" for task_id in ids),
                    encoding="utf-8",
                )
            category_index.append(
                {
                    "name": category_name,
                    "slug": slug,
                    "tasks": len(category_records),
                    "train": sum(
                        record["split"] == "train"
                        for record in category_records
                    ),
                    "test": sum(
                        record["split"] == "test"
                        for record in category_records
                    ),
                    "points": sum(
                        int(record["points"]) for record in category_records
                    ),
                }
            )
        (output / "categories.json").write_text(
            json.dumps(category_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "output": str(output),
            "board": arguments.board,
            "tasks": len(records),
            "train": sum(item["split"] == "train" for item in records),
            "test": sum(item["split"] == "test" for item in records),
            "categories": len(categories),
            "bytes": sum(item["download_bytes"] for item in records),
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception:
        marker = output / "INCOMPLETE"
        marker.write_text(
            "Dataset pull failed. Remove this directory before retrying.\n",
            encoding="utf-8",
        )
        raise


def _load_manifest(dataset: pathlib.Path) -> dict[str, Any]:
    manifest = json.loads(
        (dataset / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark manifest schema")
    if not isinstance(manifest.get("tasks"), list):
        raise ValueError("benchmark manifest has no task list")
    return manifest


def _evaluate_task(
    record: dict[str, Any],
    *,
    dataset: pathlib.Path,
    base_url: str,
) -> tuple[dict[str, Any], str | None]:
    payload = json.loads(
        (dataset / record["task_path"]).read_text(encoding="utf-8")
    )
    task = TaskDetail.from_payload(payload)
    solver = FastPathSolver()
    classification = solver.classify(task)
    result_record = {
        "id": task.id,
        "split": record["split"],
        "category": task.category,
        "points": task.points,
        "title": task.title,
        "classification": classification,
        "supported": classification is not None,
    }
    if classification is None:
        return result_record, None
    started = time.monotonic()
    task_dir = (dataset / record["task_path"]).parent
    recorded_web = task_dir / "responses.json"
    web = (
        RecordedWebSession(recorded_web)
        if recorded_web.exists()
        else EventWebSession(base_url, timeout_seconds=10)
    )
    result = solver.solve(
        task,
        task_dir,
        web,
    )
    elapsed = time.monotonic() - started
    result_record["elapsed_seconds"] = round(elapsed, 6)
    result_record["error"] = result.error
    if result.candidate is None:
        result_record["solved"] = False
        return result_record, None
    result_record.update(
        {
            "solved": True,
            "answer_sha256": result.candidate.answer_sha256,
            "solver_seconds": result.candidate.elapsed_seconds,
        }
    )
    return result_record, result.candidate.answer


def answers_equal(actual: str, expected: str, answer_format: str) -> bool:
    if answer_format == "exact_ci":
        return actual.strip().casefold() == expected.strip().casefold()
    if answer_format == "numeric":
        try:
            left = Decimal(actual.strip().replace(",", "").replace("$", ""))
            right = Decimal(expected.strip().replace(",", "").replace("$", ""))
        except InvalidOperation:
            return False
        return abs(left - right) <= Decimal("0.000001")
    if answer_format == "literal":
        try:
            return ast.literal_eval(actual) == ast.literal_eval(expected)
        except (ValueError, SyntaxError):
            return False
    return " ".join(actual.split()) == " ".join(expected.split())


def _submit_practice(
    task_id: str,
    answer: str,
    *,
    base_url: str,
    api_key: str,
    next_submission_at: float,
) -> tuple[dict[str, Any], float]:
    delay = next_submission_at - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    response = requests.post(
        f"{base_url}/api/submit",
        headers={"X-Api-Key": api_key},
        json={"task_id": task_id, "answer": answer},
        timeout=30,
    )
    next_at = time.monotonic() + 3.05
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError(f"submission for {task_id} returned no result")
    if payload.get("result") == "rate_limited":
        retry = max(0.0, float(payload.get("retry_in") or 0.0))
        return _submit_practice(
            task_id,
            answer,
            base_url=base_url,
            api_key=api_key,
            next_submission_at=max(next_at, time.monotonic() + retry),
        )
    return payload, next_at


def evaluate_dataset(arguments: argparse.Namespace) -> int:
    dataset = pathlib.Path(arguments.dataset).resolve()
    manifest = _load_manifest(dataset)
    oracle_path = dataset / "oracle.json"
    oracle = (
        json.loads(oracle_path.read_text(encoding="utf-8"))
        if oracle_path.exists()
        else {}
    )
    records = [
        record
        for record in manifest["tasks"]
        if (arguments.split == "all" or record["split"] == arguments.split)
        and (
            arguments.category is None
            or category_slug(str(record["category"]))
            == category_slug(arguments.category)
        )
    ]
    if not records:
        raise ValueError(f"dataset contains no {arguments.split} tasks")
    base_url = str(manifest["source"]["base_url"]).rstrip("/")
    api_key = ""
    if arguments.submit_practice:
        live_base, api_key = _credentials()
        if live_base != base_url:
            raise ValueError("dataset and live JEOPARDY_BASE_URL differ")
        if manifest["source"]["board"] != "practice":
            raise ValueError("submissions are allowed only for practice datasets")
        board = _get_json(f"{base_url}/api/board", api_key)
        if board.get("phase") != "practice":
            raise ValueError("practice submissions require phase=practice")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        evaluated = list(
            pool.map(
                lambda record: _evaluate_task(
                    record,
                    dataset=dataset,
                    base_url=base_url,
                ),
                records,
            )
        )

    results: list[dict[str, Any]] = []
    next_submission_at = 0.0
    for result, answer in evaluated:
        expected = oracle.get(result["id"])
        if answer is not None and isinstance(expected, dict):
            result["oracle_result"] = (
                "correct"
                if answers_equal(
                    answer,
                    str(expected["answer"]),
                    str(expected.get("answer_format") or "exact"),
                )
                else "incorrect"
            )
        if arguments.submit_practice and answer is not None:
            payload, next_submission_at = _submit_practice(
                result["id"],
                answer,
                base_url=base_url,
                api_key=api_key,
                next_submission_at=next_submission_at,
            )
            result["submission_result"] = str(
                payload.get("result") or "unknown"
            )
        results.append(result)

    elapsed = time.monotonic() - started
    latencies = [
        float(result["elapsed_seconds"])
        for result in results
        if result.get("solved")
    ]
    correct = [
        result
        for result in results
        if result.get("submission_result") == "correct"
        or result.get("oracle_result") == "correct"
    ]
    summary = {
        "dataset": str(dataset),
        "split": arguments.split,
        "category": arguments.category,
        "tasks": len(results),
        "supported": sum(bool(result["supported"]) for result in results),
        "solved": sum(bool(result.get("solved")) for result in results),
        "points_total": sum(int(result["points"]) for result in results),
        "points_solved": sum(
            int(result["points"])
            for result in results
            if result.get("solved")
        ),
        "wall_seconds": round(elapsed, 6),
        "latency_p50": round(statistics.median(latencies), 6)
        if latencies
        else None,
        "latency_p95": round(percentile(latencies, 0.95), 6)
        if latencies
        else None,
        "latency_max": round(max(latencies), 6) if latencies else None,
        "submitted": sum("submission_result" in result for result in results),
        "oracle_checked": sum("oracle_result" in result for result in results),
        "correct": len(correct),
        "points_correct": sum(int(result["points"]) for result in correct),
    }
    if arguments.results:
        path = pathlib.Path(arguments.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(result, sort_keys=True) + "\n"
                for result in results
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["solved"] == summary["supported"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Freeze and evaluate Agent Jeopardy benchmark datasets."
    )
    commands = root.add_subparsers(dest="command", required=True)

    pull = commands.add_parser("pull", help="Download one currently available board.")
    pull.add_argument("--board", choices=("practice", "qual", "main"), required=True)
    pull.add_argument("--output", required=True)
    pull.add_argument(
        "--split-policy",
        choices=("within-cell", "all-train", "all-test"),
        default="within-cell",
    )
    pull.add_argument("--test-fraction", type=float, default=0.5)
    pull.add_argument("--seed", default="walla-walla-v1")
    pull.add_argument("--workers", type=int, default=8)
    pull.add_argument("--no-files", action="store_true")
    pull.set_defaults(run=pull_dataset)

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate deterministic fast paths on a frozen dataset."
    )
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--split", choices=("train", "test", "all"), default="test")
    evaluate.add_argument(
        "--category",
        help="Evaluate one category by name or slug.",
    )
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--submit-practice", action="store_true")
    evaluate.add_argument("--results")
    evaluate.set_defaults(run=evaluate_dataset)
    return root


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.run(arguments))


if __name__ == "__main__":
    raise SystemExit(main())

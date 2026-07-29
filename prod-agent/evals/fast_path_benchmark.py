from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from fast_paths import FastPathSolver
from game_client import GameClient
from models import TaskDetail
from tools.web import EventWebSession


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", choices=("qual", "practice", "main"), default="qual")
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--min-supported", type=int, default=1)
    parser.add_argument("--all-variants", action="store_true")
    arguments = parser.parse_args()

    config = Config.from_env()
    client = GameClient(config)
    solver = FastPathSolver()
    response = requests.get(
        f"{config.base_url}/api/board",
        headers={"X-Api-Key": config.team_api_key},
        timeout=15,
    )
    response.raise_for_status()
    raw = response.json()

    representatives: list[tuple[str, str, int, str]] = []
    for category in (raw.get("boards") or {}).get(arguments.board, ()):
        for cell in category.get("tiles") or ():
            variants = cell.get("variants") or ()
            task_ids = (
                [variant.get("id") for variant in variants]
                if arguments.all_variants and variants
                else [
                    variants[0].get("id")
                    if variants
                    else next(
                        iter(cell.get("open_ids") or ()),
                        cell.get("id"),
                    )
                ]
            )
            for task_id in task_ids:
                if not task_id:
                    continue
                representatives.append(
                    (
                        str(task_id),
                        str(category.get("name") or ""),
                        int(cell.get("points") or 0),
                        str(cell.get("title") or ""),
                    )
                )

    started = time.monotonic()
    supported = 0
    solved = 0
    solved_points = 0
    latencies: list[float] = []
    processed = 0
    for task_id, category, points, title in representatives:
        if time.monotonic() - started >= arguments.max_seconds:
            print("benchmark deadline reached")
            break
        processed += 1
        task = client.task(task_id)
        classification = solver.classify(task)
        if classification is None:
            print(f"fallback {task_id} {category} {points} {title}")
            continue
        supported += 1
        workdir = client.fetch_files(task)
        result = solver.solve(
            task,
            workdir,
            EventWebSession(
                config.base_url,
                timeout_seconds=config.model_timeout_seconds,
            ),
        )
        if result.candidate is None:
            print(f"failed {task_id} path={classification} error={result.error}")
            continue
        solved += 1
        solved_points += points
        latencies.append(result.candidate.elapsed_seconds)
        print(
            f"solved {task_id} path={classification} "
            f"seconds={result.candidate.elapsed_seconds:.3f}"
        )

    elapsed = time.monotonic() - started
    total = len(representatives)
    print(
        f"summary board={arguments.board} templates={total} "
        f"supported={supported} solved={solved} "
        f"points={solved_points} elapsed={elapsed:.3f}s"
    )
    if latencies:
        print(
            f"latency p50={statistics.median(latencies):.3f}s "
            f"p95={percentile(latencies, 0.95):.3f}s "
            f"max={max(latencies):.3f}s"
        )
        print(
            "projected claim throughput is submission-limited at "
            f"{60 / config.submission_interval_seconds:.1f} tiles/min"
        )
    complete = processed == total
    enough_coverage = supported >= arguments.min_supported
    return 0 if complete and enough_coverage and solved == supported else 1


if __name__ == "__main__":
    raise SystemExit(main())

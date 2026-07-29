from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import itertools
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
from scheduler import Scheduler
from tools.web import EventWebSession


@dataclass(frozen=True)
class ReplayTask:
    task_id: str
    category: str
    points: int
    deadline: float
    measured_seconds: float
    scheduler_score: float


@dataclass(frozen=True)
class Variant:
    workers: int
    dispatch: str
    submission: str


@dataclass(frozen=True)
class ReplayResult:
    points: int
    tiles: int
    last_submit: float


def dispatch_key(task: ReplayTask, policy: str) -> tuple[float, ...]:
    if policy == "deadline":
        return (task.deadline, -task.points)
    if policy == "points":
        return (-task.points, task.deadline)
    if policy == "speed":
        return (task.measured_seconds, -task.points)
    if policy == "value_per_deadline":
        return (-(task.points / max(1.0, task.deadline)), task.deadline)
    if policy == "scheduler":
        return (-task.scheduler_score, task.deadline)
    raise ValueError(f"unknown dispatch policy: {policy}")


def submission_key(
    task: ReplayTask,
    policy: str,
    now: float,
    max_deadline: float,
) -> tuple[float, ...]:
    slack = max(0.001, task.deadline - now)
    if policy == "deadline":
        return (task.deadline, -task.points)
    if policy == "points":
        return (-task.points, task.deadline)
    if policy == "value_hazard":
        hazard = 1.0 - min(1.0, task.deadline / max_deadline)
        return (-(task.points * (1.0 + hazard)), task.deadline)
    if policy == "value_per_slack":
        return (-(task.points / slack), task.deadline)
    raise ValueError(f"unknown submission policy: {policy}")


def replay(
    tasks: list[ReplayTask],
    variant: Variant,
    *,
    interval: float,
    latency_multiplier: float,
    latency_padding: float,
    startup_delay: float,
) -> ReplayResult:
    worker_ready = [startup_delay] * variant.workers
    heapq.heapify(worker_ready)
    ready: list[tuple[float, ReplayTask]] = []
    for task in sorted(tasks, key=lambda item: dispatch_key(item, variant.dispatch)):
        available = heapq.heappop(worker_ready)
        solve_seconds = (
            task.measured_seconds * latency_multiplier + latency_padding
        )
        candidate_ready = available + solve_seconds
        heapq.heappush(worker_ready, candidate_ready)
        ready.append((candidate_ready, task))
    ready.sort(key=lambda item: item[0])

    pending: list[ReplayTask] = []
    index = 0
    next_submit = startup_delay
    points = 0
    tiles = 0
    last_submit = startup_delay
    max_deadline = max(task.deadline for task in tasks)
    while index < len(ready) or pending:
        if not pending and index < len(ready):
            next_submit = max(next_submit, ready[index][0])
        while index < len(ready) and ready[index][0] <= next_submit:
            pending.append(ready[index][1])
            index += 1
        pending = [
            task for task in pending if next_submit < task.deadline
        ]
        if not pending:
            if index >= len(ready):
                break
            continue
        chosen = min(
            pending,
            key=lambda task: submission_key(
                task,
                variant.submission,
                next_submit,
                max_deadline,
            ),
        )
        pending.remove(chosen)
        points += chosen.points
        tiles += 1
        last_submit = next_submit
        next_submit += interval
    return ReplayResult(points, tiles, last_submit)


def collect_tasks(config: Config, board_name: str) -> list[ReplayTask]:
    headers = {"X-Api-Key": config.team_api_key}
    response = requests.get(
        f"{config.base_url}/api/board",
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()
    raw = response.json()
    all_claims = [
        float(variant["claimed_by"]["at"])
        for category in (raw.get("boards") or {}).get(board_name, ())
        for cell in category.get("tiles") or ()
        for variant in cell.get("variants") or ()
        if (variant.get("claimed_by") or {}).get("at") is not None
    ]
    if not all_claims:
        raise RuntimeError(f"{board_name} has no claim history to replay")
    round_start = min(all_claims)

    client = GameClient(config)
    fast_paths = FastPathSolver()
    scheduler = Scheduler(mode="scored")
    tasks: list[ReplayTask] = []
    for category in (raw.get("boards") or {}).get(board_name, ()):
        category_name = str(category.get("name") or "")
        for cell in category.get("tiles") or ():
            points = int(cell.get("points") or 0)
            estimate = scheduler.estimate(
                type(
                    "TileEstimate",
                    (),
                    {"category": category_name, "points": points},
                )()
            )
            scheduler_score = points * estimate.probability / estimate.solve_seconds
            for variant in cell.get("variants") or ():
                claim = (variant.get("claimed_by") or {}).get("at")
                if claim is None:
                    continue
                task = client.task(str(variant["id"]))
                if fast_paths.classify(task) is None:
                    continue
                workdir = client.fetch_files(task)
                result = fast_paths.solve(
                    task,
                    workdir,
                    EventWebSession(
                        config.base_url,
                        timeout_seconds=config.model_timeout_seconds,
                    ),
                )
                if result.candidate is None:
                    raise RuntimeError(
                        f"fast path failed for {task.id}: {result.error}"
                    )
                tasks.append(
                    ReplayTask(
                        task_id=task.id,
                        category=category_name,
                        points=points,
                        deadline=float(claim) - round_start,
                        measured_seconds=result.candidate.elapsed_seconds,
                        scheduler_score=scheduler_score,
                    )
                )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", choices=("qual", "main"), default="qual")
    parser.add_argument("--max-seconds", type=float, default=150.0)
    arguments = parser.parse_args()
    started = time.monotonic()
    config = Config.from_env()
    tasks = collect_tasks(config, arguments.board)

    variants = [
        Variant(workers, dispatch, submission)
        for workers, dispatch, submission in itertools.product(
            range(1, 7),
            (
                "deadline",
                "points",
                "speed",
                "value_per_deadline",
                "scheduler",
            ),
            (
                "deadline",
                "points",
                "value_hazard",
                "value_per_slack",
            ),
        )
    ]
    scenarios = list(
        itertools.product(
            (1.0, 2.0, 4.0),
            (0.1, 1.0, 3.0, 5.0),
            (0.0, 10.0, 30.0, 45.0),
        )
    )
    scored: list[tuple[int, float, int, Variant, float]] = []
    for variant in variants:
        results = [
            replay(
                tasks,
                variant,
                interval=config.submission_interval_seconds,
                latency_multiplier=multiplier,
                latency_padding=padding,
                startup_delay=startup,
            )
            for multiplier, padding, startup in scenarios
        ]
        scored.append(
            (
                min(result.points for result in results),
                statistics.mean(result.points for result in results),
                min(result.tiles for result in results),
                variant,
                max(result.last_submit for result in results),
            )
        )
        if time.monotonic() - started > arguments.max_seconds:
            raise TimeoutError("auto-optimizer exceeded its time budget")

    scored.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            abs(item[3].workers - 4),
            item[4],
        )
    )
    baseline_variant = Variant(4, "scheduler", "value_hazard")
    baseline = next(item for item in scored if item[3] == baseline_variant)
    print(
        f"tasks={len(tasks)} points={sum(task.points for task in tasks)} "
        f"scenarios={len(scenarios)} variants={len(variants)}"
    )
    print(
        "baseline "
        f"workers={baseline_variant.workers} "
        f"dispatch={baseline_variant.dispatch} "
        f"submission={baseline_variant.submission} "
        f"worst_points={baseline[0]} avg_points={baseline[1]:.1f} "
        f"worst_tiles={baseline[2]}"
    )
    for rank, item in enumerate(scored[:10], 1):
        worst_points, average_points, worst_tiles, variant, last_submit = item
        print(
            f"{rank}. workers={variant.workers} "
            f"dispatch={variant.dispatch} "
            f"submission={variant.submission} "
            f"worst_points={worst_points} "
            f"avg_points={average_points:.1f} "
            f"worst_tiles={worst_tiles} "
            f"latest_submit={last_submit:.2f}s"
        )
    print(f"elapsed={time.monotonic() - started:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

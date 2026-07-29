from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
import pathlib
import shutil
import tempfile
import threading
import time
from typing import Any

from confidence import ConfidenceEngine
from config import Config
from fast_paths import FastPathSolver
from game_client import GameClient, TileUnavailable
from models import (
    BoardSnapshot,
    Candidate,
    ConfidenceDecision,
    Evidence,
    Phase,
    Priority,
    TaskDetail,
    Tile,
)
from playbooks import PlaybookLoader
from practice_log import PracticeAttemptLog
from runtime import ProductionToolRuntime
from scheduler import Scheduler
from solver import AnthropicSolver
from submission import SubmissionLane
from tools import WebSessionPool


@dataclass(frozen=True)
class SolveJob:
    tile: Tile
    priority: Priority
    temperature: float
    solve_number: int


@dataclass(frozen=True)
class SolveOutcome:
    job: SolveJob
    task: TaskDetail | None
    candidate: Candidate | None
    error: str | None = None


@dataclass(frozen=True)
class VerifyJob:
    outcome: SolveOutcome
    decision: ConfidenceDecision


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.client = GameClient(config)
        self.scheduler = Scheduler(mode=config.mode)
        self.playbooks = PlaybookLoader(config.playbooks_path)
        self.solver = AnthropicSolver(config, self.playbooks)
        self.fast_paths = FastPathSolver()
        self.confidence = ConfidenceEngine(config)
        self.practice_log = PracticeAttemptLog(config.practice_results_path)
        self.web_sessions = WebSessionPool(
            config.base_url,
            timeout_seconds=config.model_timeout_seconds,
        )
        self._fallback_slots = threading.BoundedSemaphore(
            config.fallback_workers
        )
        self._snapshot_lock = threading.Lock()
        self._snapshot: BoardSnapshot | None = None
        self._dashboard_lock = threading.Lock()
        self._dashboard: tuple[float, dict[str, Any]] | None = None
        self._submission = SubmissionLane(
            self.client,
            config,
            self._fresh_board,
        )
        self._solve_counts: dict[str, int] = {}
        self._practice_attempted: set[str] = set()
        self._priority_by_task: dict[str, Priority] = {}
        self._scheduled_at = 0
        self._work_state_lock = threading.Lock()
        self._work_states: dict[str, str] = {}
        self._deferred_until: dict[str, float] = {}
        self._transient_failures: dict[str, int] = {}
        self._submission_failures: dict[str, int] = {}
        self._pending_submissions: list[
            tuple[SolveOutcome, ConfidenceDecision]
        ] = []

    @staticmethod
    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _set_snapshot(self, snapshot: BoardSnapshot) -> None:
        with self._snapshot_lock:
            self._snapshot = snapshot

    def _get_snapshot(self) -> BoardSnapshot | None:
        with self._snapshot_lock:
            return self._snapshot

    def _fresh_board(self, task_id: str | None = None) -> BoardSnapshot:
        snapshot = self.client.board()
        self._set_snapshot(snapshot)
        return snapshot

    def _dashboard_status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._dashboard_lock:
            if self._dashboard and now - self._dashboard[0] < 15:
                return dict(self._dashboard[1])
        try:
            payload = self.client.dashboard().payload
        except Exception as error:
            return {"available": False, "error": type(error).__name__}
        with self._dashboard_lock:
            self._dashboard = (now, dict(payload))
        return dict(payload)

    def _problem_status(self, task_id: str) -> dict[str, Any]:
        snapshot = self._get_snapshot()
        if snapshot is None:
            return {"task_id": task_id, "open": False, "reason": "no snapshot"}
        tile = next((tile for tile in snapshot.tiles if tile.id == task_id), None)
        attempt = self._submission.state(task_id)
        priority = self._priority_by_task.get(task_id)
        with self._work_state_lock:
            work_state = self._work_states.get(task_id)
        return {
            "task_id": task_id,
            "phase": snapshot.phase.value,
            "open": tile is not None,
            "solved": task_id in snapshot.solved_ids,
            "snapshot_age_seconds": round(
                time.monotonic() - snapshot.fetched_monotonic, 3
            ),
            "points": tile.points if tile else None,
            "remaining": tile.remaining if tile else None,
            "attempts": attempt.attempts,
            "incorrect_attempts": attempt.incorrect_attempts,
            "cooldown_seconds": max(
                0.0, round(attempt.cooldown_until - time.monotonic(), 3)
            ),
            "priority": round(priority.score, 5) if priority else None,
            "working_on": work_state is not None,
            "work_state": work_state,
        }

    def _set_work_state(self, task_id: str, state: str | None) -> None:
        with self._work_state_lock:
            if state is None:
                self._work_states.pop(task_id, None)
            else:
                self._work_states[task_id] = state

    def _runtime(
        self,
        task: TaskDetail,
        workdir: pathlib.Path,
        temperature: float,
        web_sessions: WebSessionPool | None = None,
    ) -> ProductionToolRuntime:
        return ProductionToolRuntime(
            task,
            workdir,
            self.config,
            web_sessions or self.web_sessions,
            self._problem_status,
            self._dashboard_status,
            temperature,
        )

    def _solve(self, job: SolveJob) -> SolveOutcome:
        started = time.monotonic()
        try:
            task = self.client.task(job.tile.id)
            workdir = self.client.fetch_files(task)
            for name in ("answer.txt", "candidate.json"):
                (workdir / name).unlink(missing_ok=True)
            fast = self.fast_paths.solve(
                task,
                workdir,
                self.web_sessions.for_task(task.id),
            )
            if fast.candidate is not None:
                self.log(
                    f"{task.id}: deterministic fast path completed in "
                    f"{fast.candidate.elapsed_seconds:.3f}s"
                )
                return SolveOutcome(job, task, fast.candidate)
            if fast.error:
                self.log(
                    f"{task.id}: deterministic fast path failed: {fast.error}; "
                    "falling back to Haiku"
                )
            if not self._fallback_slots.acquire(blocking=False):
                return SolveOutcome(
                    job, task, None, "deferred: fallback lane busy"
                )
            try:
                runtime = self._runtime(task, workdir, job.temperature)
                candidate = self.solver.solve(
                    task,
                    str(workdir),
                    job.temperature,
                    runtime,
                    should_continue=lambda: (
                        time.monotonic() - started
                        < self._fallback_deadline(job.tile)
                        and bool(self._problem_status(task.id).get("open"))
                    ),
                )
            finally:
                self._fallback_slots.release()
            return SolveOutcome(job, task, candidate)
        except TileUnavailable as error:
            return SolveOutcome(job, None, None, f"unavailable: {error}")
        except Exception as error:
            return SolveOutcome(
                job, None, None, f"{type(error).__name__}: {error}"
            )

    def _verify(self, verify_job: VerifyJob) -> SolveOutcome:
        primary = verify_job.outcome
        assert primary.task is not None and primary.candidate is not None
        alternate = self._alternate_temperature(primary.job.temperature)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"verify_{primary.task.id}_"
            ) as directory:
                workdir = pathlib.Path(directory)
                source = self.client.workdir(primary.task.id)
                for name in primary.task.files:
                    source_path = source / name
                    target_path = workdir / name
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, target_path)
                candidate = self.solver.solve(
                    primary.task,
                    str(workdir),
                    alternate,
                    self._runtime(
                        primary.task,
                        workdir,
                        alternate,
                        WebSessionPool(self.config.base_url),
                    ),
                )
            return SolveOutcome(
                replace(primary.job, temperature=alternate),
                primary.task,
                candidate,
            )
        except Exception as error:
            return SolveOutcome(
                primary.job,
                primary.task,
                None,
                f"{type(error).__name__}: {error}",
            )

    def _alternate_temperature(self, current: float) -> float:
        values = self.config.temperatures
        if len(values) == 1:
            return values[0]
        try:
            index = values.index(current)
        except ValueError:
            index = 0
        return values[(index + 1) % len(values)]

    def _temperature(self, tile: Tile, solve_number: int) -> float:
        if self.config.thinking_budget(tile.points) is not None:
            return 1.0
        index = self._scheduled_at - 1
        return self.config.temperatures[index % len(self.config.temperatures)]

    def _playable(self, snapshot: BoardSnapshot) -> bool:
        if self.config.mode == "practice_eval":
            return snapshot.phase is Phase.PRACTICE
        if self.config.mode == "auto":
            return snapshot.phase in {
                Phase.PRACTICE,
                Phase.ROUND1,
                Phase.GAME,
            }
        return snapshot.phase in {Phase.ROUND1, Phase.GAME}

    def _evaluation_active(self) -> bool:
        snapshot = self._get_snapshot()
        return (
            snapshot is not None
            and snapshot.phase is Phase.PRACTICE
            and self.config.mode in {"auto", "practice_eval"}
        )

    def _max_solves(self, tile: Tile) -> int:
        if self._evaluation_active():
            return 1
        return 2 if tile.points >= 400 else 1

    def _fallback_deadline(self, tile: Tile) -> float:
        tier_limit = {
            100: 12.0,
            200: 12.0,
            300: 18.0,
            400: 25.0,
            500: 30.0,
        }.get(tile.points, self.config.fallback_deadline_seconds)
        return min(self.config.fallback_deadline_seconds, tier_limit)

    def _retry_transient(self, task_id: str) -> bool:
        snapshot = self._get_snapshot()
        if (
            snapshot is None
            or task_id not in {tile.id for tile in snapshot.tiles}
        ):
            return False
        failures = self._transient_failures.get(task_id, 0) + 1
        self._transient_failures[task_id] = failures
        if failures > 1:
            return False
        self._solve_counts[task_id] = max(
            0, self._solve_counts.get(task_id, 1) - 1
        )
        self._deferred_until[task_id] = time.monotonic() + 2.0
        self._practice_attempted.discard(task_id)
        return True

    def _rank_available(
        self,
        snapshot: BoardSnapshot,
        reserved: set[str],
    ) -> list[tuple[Tile, Priority]]:
        attempts = self._submission.states()
        priorities = self.scheduler.rank_board(snapshot, attempts)
        self._priority_by_task = {
            priority.task_id: priority for priority in priorities
        }
        tile_by_id = {tile.id: tile for tile in snapshot.tiles}
        ranked = []
        now = time.monotonic()
        for priority in priorities:
            tile = tile_by_id[priority.task_id]
            if (
                self.config.task_filter
                and tile.id not in self.config.task_filter
            ):
                continue
            if tile.id in reserved:
                continue
            if (
                self._evaluation_active()
                and tile.id in self._practice_attempted
            ):
                continue
            if self._solve_counts.get(tile.id, 0) >= self._max_solves(tile):
                continue
            if self._deferred_until.get(tile.id, 0.0) > now:
                continue
            ranked.append((tile, priority))
        return ranked

    @staticmethod
    def _diverse_take(
        ranked: list[tuple[Tile, Priority]],
        slots: int,
        occupied_categories: set[str] | None = None,
    ) -> list[tuple[Tile, Priority]]:
        if slots <= 0:
            return []
        chosen: list[tuple[Tile, Priority]] = []
        categories = set(occupied_categories or ())
        high_value_slots = 1
        selected_high_value = 0
        for item in ranked:
            if (
                item[0].points == 500
                and item[0].category not in categories
            ):
                chosen.append(item)
                categories.add(item[0].category)
                selected_high_value += 1
                if selected_high_value == high_value_slots:
                    break
        if len(chosen) >= slots:
            return chosen[:slots]
        for item in ranked:
            if item not in chosen and item[0].category not in categories:
                chosen.append(item)
                categories.add(item[0].category)
                if len(chosen) == slots:
                    return chosen
        for item in ranked:
            if item not in chosen:
                chosen.append(item)
                if len(chosen) == slots:
                    break
        return chosen

    def _record_practice(
        self,
        outcome: SolveOutcome,
        *,
        decision: ConfidenceDecision | None = None,
        submitted: bool = False,
        result: str,
        failure_stage: str | None = None,
        error_type: str | None = None,
    ) -> None:
        if not self._evaluation_active():
            return
        candidate = outcome.candidate
        self.practice_log.append(
            experiment_id=self.config.experiment_id,
            task_id=outcome.job.tile.id,
            category=outcome.job.tile.category,
            points=outcome.job.tile.points,
            temperature=outcome.job.temperature,
            thinking_budget=self.config.thinking_budget(
                outcome.job.tile.points
            ),
            workers=self.config.workers,
            max_turns=self.config.max_turns,
            max_tokens=self.config.max_tokens,
            max_tool_output=self.config.max_tool_output,
            elapsed_seconds=(
                candidate.elapsed_seconds if candidate is not None else None
            ),
            tool_turns=candidate.tool_turns if candidate else None,
            confidence=decision.probability if decision else None,
            submitted=submitted,
            result=result,
            answer_sha256=candidate.answer_sha256 if candidate else None,
            failure_stage=failure_stage,
            error_type=error_type,
        )

    def _submit(
        self,
        outcome: SolveOutcome,
        decision: ConfidenceDecision,
    ) -> dict[str, Any]:
        assert outcome.candidate is not None
        lane_decision = None if self._evaluation_active() else decision
        payload = self._submission.try_submit(outcome.candidate, lane_decision)
        result = str(payload.get("result") or "unknown")
        self.log(
            f"{outcome.job.tile.id}: confidence={decision.probability:.3f} "
            f"submitted={payload.get('submitted', True)} result={result}"
            + (
                f" reason={payload.get('reason')}"
                if payload.get("reason")
                else ""
            )
        )
        self._record_practice(
            outcome,
            decision=decision,
            submitted=result not in {"rejected", "unknown", "rate_limited"},
            result=result,
            failure_stage=(
                "submission" if result in {"rejected", "unknown"} else None
            ),
            error_type=payload.get("rejection"),
        )
        return payload

    @staticmethod
    def _submission_priority(
        item: tuple[SolveOutcome, ConfidenceDecision],
    ) -> tuple[float, int, float]:
        outcome, _ = item
        priority = outcome.job.priority
        claim_hazard = 1.0 - priority.race_survival
        return (
            priority.net_value * (1.0 + claim_hazard),
            outcome.job.tile.points,
            priority.score,
        )

    def _dispatch_next_submission(
        self,
        submission_pool: ThreadPoolExecutor,
        submissions: dict[
            Future[dict[str, Any]],
            tuple[SolveOutcome, ConfidenceDecision],
        ],
    ) -> None:
        if submissions or not self._pending_submissions:
            return
        item = max(
            self._pending_submissions,
            key=self._submission_priority,
        )
        self._pending_submissions.remove(item)
        outcome, decision = item
        self._set_work_state(
            outcome.job.tile.id, "submission_in_flight"
        )
        future = submission_pool.submit(self._submit, outcome, decision)
        submissions[future] = item

    def _queue_submit(
        self,
        outcome: SolveOutcome,
        decision: ConfidenceDecision,
        submission_pool: ThreadPoolExecutor,
        submissions: dict[
            Future[dict[str, Any]],
            tuple[SolveOutcome, ConfidenceDecision],
        ],
    ) -> None:
        self._set_work_state(outcome.job.tile.id, "submission_pending")
        self._pending_submissions.append((outcome, decision))
        self._dispatch_next_submission(submission_pool, submissions)

    def _handle_solve(
        self,
        outcome: SolveOutcome,
        verifier_pool: ThreadPoolExecutor,
        verifiers: dict[Future[SolveOutcome], VerifyJob],
        submission_pool: ThreadPoolExecutor,
        submissions: dict[
            Future[dict[str, Any]],
            tuple[SolveOutcome, ConfidenceDecision],
        ],
    ) -> None:
        task_id = outcome.job.tile.id
        if outcome.error == "deferred: fallback lane busy":
            self._solve_counts[task_id] = max(
                0, self._solve_counts.get(task_id, 1) - 1
            )
            self._deferred_until[task_id] = time.monotonic() + 2.0
            self._practice_attempted.discard(task_id)
            self.log(
                f"{task_id}: deferred; deterministic path unavailable and "
                "the single Haiku fallback lane is busy"
            )
            self._set_work_state(task_id, None)
            return
        snapshot = self._get_snapshot()
        if (
            snapshot is not None
            and task_id not in {tile.id for tile in snapshot.tiles}
            and not self._evaluation_active()
        ):
            self.log(
                f"{task_id}: abandoned before submission; tile was claimed "
                "while solving"
            )
            self._set_work_state(task_id, None)
            return
        if outcome.error:
            if (
                not outcome.error.startswith("unavailable:")
                and self._retry_transient(task_id)
            ):
                self.log(
                    f"{task_id}: transient solve failure; one bounded retry "
                    f"scheduled ({outcome.error})"
                )
                self._set_work_state(task_id, None)
                return
            self.log(f"{task_id}: solve failed: {outcome.error}")
            self._record_practice(
                outcome,
                result="exception",
                failure_stage="solve",
                error_type=outcome.error.split(":", 1)[0],
            )
            self._set_work_state(task_id, None)
            return
        if outcome.candidate is None:
            if self._retry_transient(task_id):
                self.log(
                    f"{task_id}: deadline/no-candidate; one bounded retry "
                    "scheduled"
                )
                self._set_work_state(task_id, None)
                return
            self.log(f"{task_id}: no verified candidate")
            self._record_practice(
                outcome, result="unverified", failure_stage="candidate"
            )
            self._set_work_state(task_id, None)
            return
        urgency = max(0.0, 1.0 - outcome.job.priority.race_survival)
        decision = self.confidence.assess(
            outcome.candidate, urgency=urgency
        )
        if self._evaluation_active():
            self._queue_submit(
                outcome, decision, submission_pool, submissions
            )
            return
        if decision.should_submit:
            self._queue_submit(
                outcome, decision, submission_pool, submissions
            )
            return
        if (
            self.config.verifier_workers
            and outcome.job.tile.points >= 400
            and len(verifiers) < self.config.verifier_workers
        ):
            verify_job = VerifyJob(outcome, decision)
            future = verifier_pool.submit(self._verify, verify_job)
            verifiers[future] = verify_job
            self._set_work_state(task_id, "verifying")
            self.log(
                f"{task_id}: confidence={decision.probability:.3f}; "
                "independent verifier started"
            )
            return
        self.log(
            f"{task_id}: confidence={decision.probability:.3f} below "
            f"{decision.threshold:.3f}; not submitted"
        )
        self._set_work_state(task_id, None)

    def _handle_verify(
        self,
        verifier: SolveOutcome,
        verify_job: VerifyJob,
        submission_pool: ThreadPoolExecutor,
        submissions: dict[
            Future[dict[str, Any]],
            tuple[SolveOutcome, ConfidenceDecision],
        ],
    ) -> None:
        primary = verify_job.outcome
        assert primary.candidate is not None
        task_id = primary.job.tile.id
        if (
            verifier.candidate is None
            or verifier.candidate.answer_sha256
            != primary.candidate.answer_sha256
        ):
            self.log(f"{task_id}: independent verifier did not match")
            self._set_work_state(task_id, None)
            return
        evidence = primary.candidate.evidence
        matched = replace(
            primary.candidate,
            evidence=Evidence(
                method=evidence.method,
                deterministic_checks=evidence.deterministic_checks,
                independent_checks=(
                    *evidence.independent_checks,
                    "independent Haiku 4.5 verifier matched answer hash",
                ),
                assumptions=evidence.assumptions,
                tool_errors=evidence.tool_errors,
                input_complete=evidence.input_complete,
                direct_provenance=evidence.direct_provenance,
            ),
        )
        matched_outcome = replace(primary, candidate=matched)
        decision = self.confidence.assess(matched)
        if decision.should_submit:
            self._queue_submit(
                matched_outcome, decision, submission_pool, submissions
            )
        else:
            self.log(
                f"{task_id}: matching verifier still below confidence gate"
            )
            self._set_work_state(task_id, None)

    def run(self) -> None:
        started = time.monotonic()
        next_board_refresh = 0.0
        active: dict[Future[SolveOutcome], SolveJob] = {}
        verifiers: dict[Future[SolveOutcome], VerifyJob] = {}
        submissions: dict[
            Future[dict[str, Any]],
            tuple[SolveOutcome, ConfidenceDecision],
        ] = {}
        last_phase: Phase | None = None
        with (
            ThreadPoolExecutor(max_workers=self.config.workers) as solver_pool,
            ThreadPoolExecutor(
                max_workers=max(1, self.config.verifier_workers)
            ) as verifier_pool,
            ThreadPoolExecutor(max_workers=1) as submission_pool,
        ):
            while True:
                now = time.monotonic()
                if (
                    self.config.run_duration_seconds
                    and now - started >= self.config.run_duration_seconds
                ):
                    self.log("configured run duration reached")
                    return
                if now >= next_board_refresh:
                    try:
                        snapshot = self._fresh_board()
                    except Exception as error:
                        self.log(f"board refresh failed: {type(error).__name__}")
                        time.sleep(1)
                        continue
                    next_board_refresh = now + self.config.board_poll_seconds
                    if snapshot.phase is not last_phase:
                        previous_phase = last_phase
                        self.log(
                            f"phase={snapshot.phase.value} "
                            f"open_tiles={len(snapshot.tiles)}"
                        )
                        last_phase = snapshot.phase
                        if previous_phase is not None:
                            open_ids = {tile.id for tile in snapshot.tiles}
                            stale = [
                                item
                                for item in self._pending_submissions
                                if item[0].job.tile.id not in open_ids
                            ]
                            for item in stale:
                                self._pending_submissions.remove(item)
                                task_id = item[0].job.tile.id
                                self._set_work_state(task_id, None)
                                self.log(
                                    f"{task_id}: cancelled pending candidate "
                                    "after phase change"
                                )
                    if snapshot.phase is Phase.ENDED:
                        return
                    if self._playable(snapshot):
                        reserved = {
                            job.tile.id for job in active.values()
                        } | {
                            job.outcome.job.tile.id
                            for job in verifiers.values()
                        } | {
                            outcome.job.tile.id
                            for outcome, _ in submissions.values()
                        } | {
                            outcome.job.tile.id
                            for outcome, _ in self._pending_submissions
                        }
                        occupied_categories = {
                            job.tile.category for job in active.values()
                        } | {
                            job.outcome.job.tile.category
                            for job in verifiers.values()
                        } | {
                            outcome.job.tile.category
                            for outcome, _ in submissions.values()
                        } | {
                            outcome.job.tile.category
                            for outcome, _ in self._pending_submissions
                        }
                        slots = self.config.workers - len(active)
                        if (
                            len(submissions) + len(self._pending_submissions)
                            >= self.config.workers * 2
                        ):
                            slots = 0
                        ranked = self._rank_available(snapshot, reserved)
                        for tile, priority in self._diverse_take(
                            ranked, slots, occupied_categories
                        ):
                            solve_number = self._solve_counts.get(tile.id, 0) + 1
                            self._solve_counts[tile.id] = solve_number
                            self._scheduled_at += 1
                            job = SolveJob(
                                tile,
                                priority,
                                self._temperature(tile, solve_number),
                                solve_number,
                            )
                            active[solver_pool.submit(self._solve, job)] = job
                            self._set_work_state(tile.id, "solving")
                            if self._evaluation_active():
                                self._practice_attempted.add(tile.id)
                            self.log(
                                f"{tile.id}: scheduled points={tile.points} "
                                f"priority={priority.score:.3f} "
                                f"temp={job.temperature} "
                                f"thinking={self.config.thinking_budget(tile.points)}"
                            )

                futures = set(active) | set(verifiers) | set(submissions)
                if not futures:
                    snapshot = self._get_snapshot()
                    if (
                        self._evaluation_active()
                        and snapshot is not None
                        and self._playable(snapshot)
                        and not self._rank_available(snapshot, set())
                    ):
                        deferred = [
                            ready_at
                            for task_id, ready_at in self._deferred_until.items()
                            if ready_at > time.monotonic()
                            and any(
                                tile.id == task_id
                                for tile in snapshot.tiles
                            )
                        ]
                        if deferred:
                            time.sleep(
                                min(
                                    0.25,
                                    max(
                                        0.0,
                                        min(deferred) - time.monotonic(),
                                    ),
                                )
                            )
                            continue
                        self.log(
                            f"practice evaluation complete: "
                            f"{len(self._practice_attempted)} tile(s)"
                        )
                        return
                    time.sleep(min(0.25, self.config.board_poll_seconds))
                    continue
                done, _ = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in active:
                        active.pop(future)
                        self._handle_solve(
                            future.result(),
                            verifier_pool,
                            verifiers,
                            submission_pool,
                            submissions,
                        )
                    elif future in verifiers:
                        verify_job = verifiers.pop(future)
                        self._handle_verify(
                            future.result(),
                            verify_job,
                            submission_pool,
                            submissions,
                        )
                    else:
                        outcome, decision = submissions.pop(future)
                        task_id = outcome.job.tile.id
                        retry_queued = False
                        try:
                            payload = future.result()
                            if payload.get("result") == "rate_limited":
                                failures = (
                                    self._submission_failures.get(task_id, 0)
                                    + 1
                                )
                                self._submission_failures[task_id] = failures
                                snapshot = self._get_snapshot()
                                still_open = (
                                    snapshot is not None
                                    and any(
                                        tile.id == task_id
                                        for tile in snapshot.tiles
                                    )
                                )
                                if failures <= 1 and still_open:
                                    self.log(
                                        f"{task_id}: rate limit retries "
                                        "exhausted; one queued retry retained"
                                    )
                                    self._queue_submit(
                                        outcome,
                                        decision,
                                        submission_pool,
                                        submissions,
                                    )
                                    retry_queued = True
                                else:
                                    self.log(
                                        f"{task_id}: rate-limited candidate "
                                        "dropped after bounded retry"
                                    )
                            else:
                                self._submission_failures.pop(task_id, None)
                        except Exception as error:
                            failures = (
                                self._submission_failures.get(task_id, 0) + 1
                            )
                            self._submission_failures[task_id] = failures
                            snapshot = self._get_snapshot()
                            still_open = (
                                snapshot is not None
                                and any(
                                    tile.id == task_id
                                    for tile in snapshot.tiles
                                )
                            )
                            if failures <= 1 and still_open:
                                self.log(
                                    f"{task_id}: submission transport failed; "
                                    "one bounded retry queued "
                                    f"({type(error).__name__})"
                                )
                                self._queue_submit(
                                    outcome,
                                    decision,
                                    submission_pool,
                                    submissions,
                                )
                                retry_queued = True
                            else:
                                self.log(
                                    f"{task_id}: submission failed: "
                                    f"{type(error).__name__}: {error}"
                                )
                        finally:
                            if not retry_queued:
                                self._set_work_state(task_id, None)
                        self._dispatch_next_submission(
                            submission_pool, submissions
                        )
                    next_board_refresh = 0.0

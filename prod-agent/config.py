from __future__ import annotations

from dataclasses import dataclass
import os


def _int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _temperatures() -> tuple[float, ...]:
    values = tuple(
        float(value.strip())
        for value in os.environ.get("TEMPERATURES", "0.0,0.25,0.5").split(",")
        if value.strip()
    )
    if not values or any(value < 0 or value > 1 for value in values):
        raise ValueError("TEMPERATURES must contain values between 0 and 1")
    return values


@dataclass(frozen=True)
class Config:
    base_url: str
    team_api_key: str
    anthropic_base_url: str
    anthropic_api_key: str
    model: str
    mode: str
    workers: int
    verifier_workers: int
    cpu_workers: int
    max_turns: int
    max_tokens: int
    max_tool_output: int
    python_timeout_seconds: int
    python_memory_mb: int
    board_poll_seconds: float
    run_duration_seconds: float
    submission_interval_seconds: float
    strong_confidence_threshold: float
    urgent_confidence_floor: float
    temperatures: tuple[float, ...]
    thinking_enabled: bool
    thinking_min_points: int
    thinking_budget_400: int
    thinking_budget_500: int
    playbooks_path: str
    practice_results_path: str
    task_filter: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Config":
        base_url = os.environ.get("JEOPARDY_BASE_URL", "").rstrip("/")
        team_api_key = os.environ.get("TEAM_API_KEY", "")
        if not base_url or not team_api_key:
            raise ValueError("JEOPARDY_BASE_URL and TEAM_API_KEY are required")
        mode = os.environ.get("AGENT_MODE", "auto").lower()
        if mode not in {"auto", "scored", "practice_eval"}:
            raise ValueError(
                "AGENT_MODE must be auto, scored, or practice_eval"
            )
        strong_threshold = _float("STRONG_CONFIDENCE_THRESHOLD", 0.90)
        urgent_floor = _float("URGENT_CONFIDENCE_FLOOR", 0.80)
        if not 0 <= urgent_floor <= strong_threshold <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= urgent <= strong <= 1")
        return cls(
            base_url=base_url,
            team_api_key=team_api_key,
            anthropic_base_url=os.environ.get(
                "ANTHROPIC_BASE_URL", f"{base_url}/anthropic"
            ).rstrip("/"),
            anthropic_api_key=os.environ.get(
                "ANTHROPIC_API_KEY", team_api_key
            ),
            model="claude-haiku-4-5",
            mode=mode,
            workers=_int("WORKERS", 4),
            verifier_workers=_int("VERIFIER_WORKERS", 1, minimum=0),
            cpu_workers=_int("CPU_WORKERS", 1),
            max_turns=_int("MAX_TURNS", 20),
            max_tokens=_int("MAX_TOKENS", 4096),
            max_tool_output=_int("MAX_TOOL_OUTPUT", 12_000),
            python_timeout_seconds=_int("PYTHON_TIMEOUT_SECONDS", 60),
            python_memory_mb=_int("PYTHON_MEMORY_MB", 1024),
            board_poll_seconds=_float("BOARD_POLL_SECONDS", 3.0, minimum=0.5),
            run_duration_seconds=_float(
                "RUN_DURATION_SECONDS", 0.0, minimum=0.0
            ),
            submission_interval_seconds=_float(
                "SUBMISSION_INTERVAL_SECONDS", 3.1, minimum=3.0
            ),
            strong_confidence_threshold=strong_threshold,
            urgent_confidence_floor=urgent_floor,
            temperatures=_temperatures(),
            thinking_enabled=os.environ.get("THINKING_ENABLED", "1") == "1",
            thinking_min_points=_int("THINKING_MIN_POINTS", 400),
            thinking_budget_400=_int("THINKING_BUDGET_400", 1024),
            thinking_budget_500=_int("THINKING_BUDGET_500", 2048),
            playbooks_path=os.environ.get("PLAYBOOKS_PATH", "playbooks.json"),
            practice_results_path=os.environ.get(
                "PRACTICE_RESULTS_PATH", "practice_results.jsonl"
            ),
            task_filter=tuple(
                task_id.strip()
                for task_id in os.environ.get("TASK_FILTER", "").split(",")
                if task_id.strip()
            ),
        )

    def thinking_budget(self, points: int) -> int | None:
        if not self.thinking_enabled or points < self.thinking_min_points:
            return None
        budget = (
            self.thinking_budget_500
            if points >= 500
            else self.thinking_budget_400
        )
        if budget < 1024 or budget >= self.max_tokens:
            raise ValueError(
                "thinking budget must be at least 1024 and below MAX_TOKENS"
            )
        return budget

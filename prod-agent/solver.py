from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Protocol

from anthropic import Anthropic

from config import Config
from models import Candidate, TaskDetail
from playbooks import PlaybookLoader


@dataclass(frozen=True)
class ToolExecution:
    output: str
    is_error: bool = False


class ToolRuntime(Protocol):
    @property
    def candidate(self) -> Candidate | None: ...

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution: ...


TOOLS = [
    {
        "name": "list_files",
        "description": "List downloaded task files and sizes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read a bounded line range from a task text file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["name", "start_line", "end_line"],
        },
    },
    {
        "name": "search_files",
        "description": "Regex-search task text files without loading whole files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "glob": {"type": "string"},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "archive",
        "description": "List or safely extract a zip or tar archive in the task directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "action": {"type": "string", "enum": ["list", "extract"]},
                "destination": {"type": "string"},
            },
            "required": ["name", "action"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run bounded Python in the task directory for parsing, computation, "
            "or deterministic verification. Credentials are not inherited."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": ["code"],
        },
    },
    {
        "name": "web_request",
        "description": (
            "Make a stateful event-host-only web request. Cookies persist for "
            "this task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST"]},
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "form": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "json": {"type": "object"},
            },
            "required": ["url", "method"],
        },
    },
    {
        "name": "get_problem_status",
        "description": "Check whether this tile is still open and worth solving.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_dashboard_status",
        "description": "Read safe model-rate and agent workload status.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "record_candidate",
        "description": (
            "Record one final answer candidate with its method and checks. "
            "This does not submit. Use only after verification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "minLength": 1, "maxLength": 1000},
                "answer_file": {
                    "type": "string",
                    "description": (
                        "Task-relative file containing the exact single-line "
                        "answer. Prefer this over answer for exact tokens."
                    ),
                },
                "method": {"type": "string", "minLength": 1, "maxLength": 2000},
                "deterministic_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "independent_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "tool_errors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "input_complete": {"type": "boolean"},
                "direct_provenance": {"type": "boolean"},
            },
            "required": [
                "method",
                "deterministic_checks",
                "independent_checks",
                "assumptions",
                "tool_errors",
                "input_complete",
                "direct_provenance",
            ],
        },
    },
]
TOOL_BY_NAME = {tool["name"]: tool for tool in TOOLS}


class AnthropicSolver:
    def __init__(
        self,
        config: Config,
        playbooks: PlaybookLoader,
        client_factory: Callable[[], Any] | None = None,
    ):
        self.config = config
        self.playbooks = playbooks
        self.client_factory = client_factory

    def _client(self) -> Anthropic:
        if self.client_factory is not None:
            return self.client_factory()
        return Anthropic(
            api_key=self.config.anthropic_api_key,
            base_url=self.config.anthropic_base_url,
            timeout=self.config.model_timeout_seconds,
            max_retries=0,
        )

    def _system_prompt(
        self, task: TaskDetail, temperature: float, workdir: str
    ) -> str:
        playbook = "\n".join(
            f"- {method}" for method in self.playbooks.select_for_task(task)
        )
        return f"""You are one fast, careful solver in an Agent Jeopardy team.

Model: Claude Haiku 4.5
Temperature: {temperature}
Task: {task.id}
Category: {task.category}
Points: {task.points}
Answer format: {task.answer_format}
Task directory: {workdir}

Prompt:
{task.prompt}

Relevant practice playbook:
{playbook}

Use tools immediately. Do not guess and do not submit. For nontrivial parsing,
computation, binary handling, or validation, run Python against the actual task
materials. Preserve cookies for web flows. Check problem status before lengthy
work and again before recording a candidate.

Minimize latency and token use. Combine inspection, computation, and validation
into as few tool calls as possible; prefer one comprehensive Python call over
many incremental reads. Record the candidate immediately once verified.

When the exact answer is verified, call record_candidate once. Include concrete
checks, list every remaining assumption or tool failure, and mark whether the
answer came directly from code/file/web output. For exact tokens or computed
strings, write the answer to a task-relative file with Python and pass
answer_file instead of retyping it."""

    @staticmethod
    def _tools_for_task(task: TaskDetail) -> list[dict[str, Any]]:
        names = {"run_python", "get_problem_status", "record_candidate"}
        if task.files:
            names.update({"list_files", "read_file", "search_files", "archive"})
        if task.category.strip().lower() == "the dark web":
            names.add("web_request")
        return [tool for tool in TOOLS if tool["name"] in names]

    def solve(
        self,
        task: TaskDetail,
        workdir: str,
        temperature: float,
        runtime: ToolRuntime,
        should_continue: Callable[[], bool] | None = None,
    ) -> Candidate | None:
        client = self._client()
        thinking_budget = self.config.thinking_budget(task.points)
        inference_temperature = 1.0 if thinking_budget is not None else temperature
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Solve this tile now with tools."}
        ]
        started = time.monotonic()
        for turn in range(self.config.max_turns):
            if should_continue is not None and not should_continue():
                return runtime.candidate
            request: dict[str, Any] = dict(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=inference_temperature,
                system=self._system_prompt(
                    task, inference_temperature, workdir
                ),
                tools=self._tools_for_task(task),
                messages=messages,
            )
            if thinking_budget is not None:
                request["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            response = client.messages.create(**request)
            messages.append({"role": "assistant", "content": response.content})
            calls = [
                block for block in response.content if block.type == "tool_use"
            ]
            if not calls:
                return runtime.candidate
            results: list[dict[str, Any]] = []
            for call in calls:
                if should_continue is not None and not should_continue():
                    return runtime.candidate
                execution = runtime.execute(call.name, dict(call.input))
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": execution.output,
                }
                if execution.is_error:
                    result["is_error"] = True
                results.append(result)
            messages.append({"role": "user", "content": results})
            candidate = runtime.candidate
            if candidate is not None:
                return Candidate(
                    task_id=candidate.task_id,
                    answer=candidate.answer,
                    answer_sha256=candidate.answer_sha256,
                    evidence=candidate.evidence,
                    model_temperature=inference_temperature,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    tool_turns=turn + 1,
                )
        return runtime.candidate

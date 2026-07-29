from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

from config import Config
from models import Candidate, TaskDetail
from solver import ToolExecution
from tools import (
    CandidateWriter,
    PythonExecutor,
    TaskFiles,
    WebSessionPool,
    shared_cpu_semaphore,
)


class ProductionToolRuntime:
    def __init__(
        self,
        task: TaskDetail,
        workdir: pathlib.Path,
        config: Config,
        web_sessions: WebSessionPool,
        problem_status: Callable[[str], dict[str, Any]],
        dashboard_status: Callable[[], dict[str, Any]],
        temperature: float,
    ):
        self.task = task
        self.workdir = workdir
        self.config = config
        self.files = TaskFiles(
            workdir,
            max_output_chars=config.max_tool_output,
        )
        self.python = PythonExecutor(
            workdir,
            timeout_seconds=config.python_timeout_seconds,
            max_output_bytes=config.max_tool_output,
            memory_mb=config.python_memory_mb,
            semaphore=shared_cpu_semaphore(config.cpu_workers),
        )
        self.web = web_sessions.for_task(task.id)
        self.problem_status = problem_status
        self.dashboard_status = dashboard_status
        self.temperature = temperature
        self._candidate: Candidate | None = None

    @property
    def candidate(self) -> Candidate | None:
        return self._candidate

    def _record_candidate(self, arguments: dict[str, Any]) -> str:
        common = {
            "method": arguments["method"],
            "deterministic_checks": arguments.get("deterministic_checks", ()),
            "independent_checks": arguments.get("independent_checks", ()),
            "assumptions": arguments.get("assumptions", ()),
            "tool_errors": arguments.get("tool_errors", ()),
            "input_complete": arguments.get("input_complete", True),
            "direct_provenance": arguments.get("direct_provenance", True),
            "model_temperature": self.temperature,
        }
        answer_file = arguments.get("answer_file")
        answer = arguments.get("answer")
        if bool(answer_file) == bool(answer):
            raise ValueError("provide exactly one of answer or answer_file")
        if answer_file:
            writer = CandidateWriter(
                self.workdir,
                self.task.id,
                answer_filename=str(answer_file),
            )
            self._candidate = writer.record_existing(**common)
        else:
            writer = CandidateWriter(self.workdir, self.task.id)
            self._candidate = writer.write(str(answer), **common)
        return json.dumps(
            {
                "recorded": True,
                "answer_sha256": self._candidate.answer_sha256,
            },
            sort_keys=True,
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        try:
            if name == "list_files":
                output = self.files.list_text()
            elif name == "read_file":
                output = self.files.read(
                    arguments["name"],
                    start_line=int(arguments["start_line"]),
                    end_line=int(arguments["end_line"]),
                )
            elif name == "search_files":
                output = self.files.search_text(
                    arguments["pattern"],
                    glob=arguments.get("glob", "*"),
                    regex=True,
                    max_matches=int(arguments.get("max_matches", 200)),
                )
            elif name == "archive":
                if arguments["action"] == "list":
                    output = self.files.archive_text(arguments["name"])
                else:
                    extracted = self.files.extract_archive(
                        arguments["name"],
                        destination=arguments.get("destination", "extracted"),
                    )
                    output = "\n".join(extracted) or "(nothing extracted)"
            elif name == "run_python":
                result = self.python.run(
                    arguments["code"],
                    timeout_seconds=int(
                        arguments.get(
                            "timeout", self.config.python_timeout_seconds
                        )
                    ),
                )
                output = result.render(self.config.max_tool_output)
            elif name == "web_request":
                response = self.web.request(
                    arguments["method"],
                    arguments["url"],
                    headers=arguments.get("headers"),
                    form=arguments.get("form"),
                    json=arguments.get("json"),
                )
                output = response.render(self.config.max_tool_output)
            elif name == "get_problem_status":
                output = json.dumps(
                    self.problem_status(self.task.id),
                    sort_keys=True,
                    default=str,
                )
            elif name == "get_dashboard_status":
                output = json.dumps(
                    self.dashboard_status(), sort_keys=True, default=str
                )
            elif name == "record_candidate":
                output = self._record_candidate(arguments)
            else:
                raise ValueError(f"unknown tool: {name}")
            return ToolExecution(output)
        except Exception as error:
            return ToolExecution(f"{type(error).__name__}: {error}", True)

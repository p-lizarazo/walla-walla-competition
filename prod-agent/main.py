"""A bounded, tool-using solver for Agent Jeopardy tasks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import requests

import jeopardy as jp

VERBOSE = os.environ.get("VERBOSE") == "1"
TASK_FILTER = [
    task_id.strip()
    for task_id in os.environ.get("TASK_FILTER", "").split(",")
    if task_id.strip()
]
MAX_TILES = int(os.environ.get("MAX_TILES", "6"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "20"))
WORKERS = int(os.environ.get("WORKERS", "3"))
TEMPERATURES = tuple(
    float(value)
    for value in os.environ.get("TEMPERATURES", "0.0,0.25,0.5").split(",")
)
MAX_TOOL_OUTPUT = 12_000
ANSWER_FILE = "answer.txt"
MODEL = "claude-haiku-4-5"
LEARNINGS_PATH = pathlib.Path(__file__).with_name("learnings.jsonl")

if not TEMPERATURES or any(not 0 <= value <= 1 for value in TEMPERATURES):
    raise ValueError("TEMPERATURES must contain one or more values in [0, 1].")
if WORKERS < 1:
    raise ValueError("WORKERS must be at least 1.")

_api_lock = threading.Lock()
_submission_lock = threading.Lock()
_learning_lock = threading.Lock()
_next_submission_at = 0.0


def _bounded(value: str) -> str:
    if len(value) <= MAX_TOOL_OUTPUT:
        return value
    return f"{value[:MAX_TOOL_OUTPUT]}\n...[output truncated]"


def _append_learning(task_id: str, detail: dict, temperature: float, event: str,
                     summary: str, **extra: Any) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_id": task_id,
        "category": detail.get("category"),
        "points": detail.get("points"),
        "answer_format": detail.get("answer_format", "exact"),
        "model": MODEL,
        "temperature": temperature,
        "event": event,
        "summary": summary[:2_000],
        **extra,
    }
    with _learning_lock:
        with LEARNINGS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _resolve_task_path(workdir: pathlib.Path, name: str) -> pathlib.Path:
    path = (workdir / name).resolve()
    if path != workdir and workdir not in path.parents:
        raise ValueError("Paths must stay inside this task's work directory.")
    return path


def _read_file(workdir: pathlib.Path, name: str, start_line: int,
               end_line: int) -> str:
    path = _resolve_task_path(workdir, name)
    if not path.is_file():
        raise ValueError(f"{name!r} is not a file.")
    if path.stat().st_size > 2_000_000:
        raise ValueError(
            "This file is large. Analyze it with run_python instead of "
            "loading it into the model context."
        )
    lines = path.read_text(errors="replace").splitlines()
    if start_line < 1 or end_line < start_line:
        raise ValueError("Line numbers must be positive and ordered.")
    return _bounded("\n".join(lines[start_line - 1:end_line]))


def _list_files(workdir: pathlib.Path) -> str:
    files = [
        f"{path.relative_to(workdir)} ({path.stat().st_size} bytes)"
        for path in sorted(workdir.rglob("*"))
        if path.is_file()
    ]
    return _bounded("\n".join(files) or "(no files)")


def _run_python(workdir: pathlib.Path, code: str, timeout: int) -> str:
    timeout = max(1, min(timeout, 60))
    environment = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workdir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout} seconds."
    output = completed.stdout
    if completed.stderr:
        output += f"\nSTDERR:\n{completed.stderr}"
    return _bounded(f"exit={completed.returncode}\n{output}")


def _web_request(session: requests.Session, url: str, method: str,
                 form: dict[str, str] | None,
                 body: dict[str, Any] | None) -> str:
    base = urlsplit(jp.BASE)
    target = urlsplit(url)
    if target.scheme != base.scheme or target.netloc != base.netloc:
        raise ValueError("Web requests may only target the event host.")
    if method not in {"GET", "POST"}:
        raise ValueError("Only GET and POST are allowed.")
    response = session.request(
        method,
        url,
        data=form if method == "POST" else None,
        json=body if method == "POST" else None,
        timeout=30,
        allow_redirects=True,
    )
    content_type = response.headers.get("content-type", "")
    response_body = response.text if "text" in content_type or "json" in content_type else response.content.hex()
    return _bounded(
        f"status={response.status_code}\nurl={response.url}\n"
        f"content-type={content_type}\n\n{response_body}"
    )


TOOLS = [
    {
        "name": "list_files",
        "description": "List every downloaded task file and its size.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": (
            "Read a text-file line range. Use run_python for binary files or "
            "large files."
        ),
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
        "name": "run_python",
        "description": (
            "Run Python in the task directory. Use this to inspect, extract, "
            "parse, calculate, validate, or solve. To submit an exact result "
            "without transcription, write the computed answer as one line to "
            f"{ANSWER_FILE!r}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["code"],
        },
    },
    {
        "name": "web_request",
        "description": (
            "Make a GET or POST request to the event web task host. Cookies "
            "persist throughout this task. Inspect every response carefully."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST"]},
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
        "name": "submit_answer_file",
        "description": (
            f"Submit the single computed answer stored in {ANSWER_FILE!r}. "
            "Use only after independently verifying it. This submits at most "
            "once for this task."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "record_learning",
        "description": (
            "Record a concise reusable lesson about the method and verification "
            "you used. Call this just before submit_answer_file; do not include "
            "the final answer, passwords, or other secrets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "required": ["summary"],
        },
    },
]


def _system_prompt(detail: dict, workdir: pathlib.Path, temperature: float) -> str:
    return f"""You are an autonomous, careful solver for one Agent Jeopardy tile.

Task id: {detail["id"]}
Category: {detail.get("category")}
Points: {detail.get("points")}
Answer format: {detail.get("answer_format", "exact")}
Task directory: {workdir}
Inference model: Claude Haiku 4.5 (temperature {temperature})

Task:
{detail["prompt"]}

Solve the task with tools. Do not guess. Inspect actual files or web responses,
write code for all nontrivial parsing and computation, and independently check
the result. You have no internet beyond the event-host web task tool.

To submit, your Python computation must write the exact final value as a
single line to answer.txt in the task directory, then call
record_learning with a concise reusable description of the approach and
verification, then call submit_answer_file. That tool reads the computed string
directly, avoiding any model transcription. Submit at most once. Do not report
an answer in text: either use the submit tool after verification, or explain
why you cannot verify it."""


def _tool_result(tool_use_id: str, output: str, is_error: bool = False) -> dict:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": output,
    }
    if is_error:
        result["is_error"] = True
    return result


def _submit_rate_safe(task_id: str, answer: str) -> dict:
    global _next_submission_at
    with _submission_lock:
        remaining = _next_submission_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        result = jp.submit(task_id, answer)
        _next_submission_at = time.monotonic() + 3
        if result.get("result") == "rate_limited":
            retry_in = max(float(result.get("retry_in", 3)), 3)
            jp.log(f"{task_id}: rate-limited; retrying computed answer in "
                   f"{retry_in:.1f}s")
            time.sleep(retry_in)
            result = jp.submit(task_id, answer)
            _next_submission_at = time.monotonic() + 3
        return result


def solve_tile(task_id: str, temperature: float) -> bool:
    with _api_lock:
        detail = jp.task(task_id)
        workdir = jp.workdir(task_id)
        names = jp.fetch_files(task_id, detail, workdir)
    answer_path = workdir / ANSWER_FILE
    answer_path.unlink(missing_ok=True)
    jp.log(f"{task_id}: solving {detail.get('category')} ({detail.get('points')}pt), "
           f"temp={temperature}, files={names or 'none'}")

    client = jp.anthropic_client()
    web_session = requests.Session()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Start solving. Use the tools now."}
    ]
    submitted = False
    started_at = time.monotonic()

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=temperature,
            system=_system_prompt(detail, workdir, temperature),
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        calls = [block for block in response.content if block.type == "tool_use"]
        if VERBOSE:
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            if text:
                jp.log(f"{task_id}: model: {text}")
        if not calls:
            jp.log(f"{task_id}: stopped without a verified submission")
            return False

        tool_results = []
        for call in calls:
            try:
                if call.name == "list_files":
                    output = _list_files(workdir)
                elif call.name == "read_file":
                    output = _read_file(workdir, **call.input)
                elif call.name == "run_python":
                    output = _run_python(
                        workdir,
                        call.input["code"],
                        call.input.get("timeout", 30),
                    )
                elif call.name == "web_request":
                    output = _web_request(
                        web_session,
                        call.input["url"],
                        call.input["method"],
                        call.input.get("form"),
                        call.input.get("json"),
                    )
                elif call.name == "submit_answer_file":
                    if submitted:
                        raise ValueError("This tile has already been submitted once.")
                    if not answer_path.is_file():
                        raise ValueError("answer.txt does not exist.")
                    answer = answer_path.read_text().strip()
                    if not answer or "\n" in answer or len(answer) > 1000:
                        raise ValueError(
                            "answer.txt must contain one nonempty answer line "
                            "of at most 1000 characters."
                        )
                    result = _submit_rate_safe(task_id, answer)
                    submitted = True
                    _append_learning(
                        task_id,
                        detail,
                        temperature,
                        "submission",
                        "Submitted a computed, locally verified answer.",
                        result=result.get("result"),
                        answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                        elapsed_seconds=round(time.monotonic() - started_at, 2),
                        tool_turns=turn + 1,
                    )
                    output = json.dumps(result, sort_keys=True)
                    jp.log(f"{task_id}: submitted computed answer -> "
                           f"{result.get('result')}")
                    tool_results.append(_tool_result(call.id, output))
                    return result.get("result") == "correct"
                elif call.name == "record_learning":
                    _append_learning(
                        task_id,
                        detail,
                        temperature,
                        "method",
                        call.input["summary"],
                    )
                    output = "Learning recorded."
                else:
                    raise ValueError(f"Unknown tool {call.name!r}.")
                tool_results.append(_tool_result(call.id, output))
            except Exception as error:  # Surface per-tool failures to the model.
                tool_results.append(_tool_result(call.id, repr(error), True))
        messages.append({"role": "user", "content": tool_results})

    jp.log(f"{task_id}: exhausted {MAX_TURNS} tool turns without submitting")
    return False


def pick_tiles(board: dict) -> list[str]:
    tiles = jp.open_tiles(board)
    open_now = {tile["id"] for tile in tiles}
    if TASK_FILTER:
        missing = [task_id for task_id in TASK_FILTER if task_id not in open_now]
        if missing:
            jp.log(f"TASK_FILTER: unavailable, skipping {missing}")
        return [task_id for task_id in TASK_FILTER if task_id in open_now]

    # Spend parallel capacity on the highest-value tiles first, but spread the
    # initial wave across categories so one slow problem type cannot block it.
    picked: list[str] = []
    for points in sorted({tile.get("points", 0) for tile in tiles}, reverse=True):
        tier = sorted(
            (tile for tile in tiles if tile.get("points", 0) == points),
            key=lambda tile: (tile.get("category", ""), tile["id"]),
        )
        categories: set[str] = set()
        for tile in tier:
            if tile["category"] not in categories:
                picked.append(tile["id"])
                categories.add(tile["category"])
                if len(picked) >= MAX_TILES:
                    return picked
        for tile in tier:
            if tile["id"] not in picked:
                picked.append(tile["id"])
                if len(picked) >= MAX_TILES:
                    return picked
    return picked


def _attempt_tile(task_id: str, temperature: float) -> bool:
    try:
        return solve_tile(task_id, temperature)
    except jp.AuthError:
        raise
    except jp.TileUnavailable as error:
        jp.log(f"{task_id}: unavailable — {error}")
    except Exception as error:  # Keep the unattended agent progressing.
        jp.log(f"{task_id}: failed — {error!r}")
    return False


def main() -> None:
    board = jp.board()
    tiles = pick_tiles(board)
    jobs = [
        (task_id, TEMPERATURES[index % len(TEMPERATURES)])
        for index, task_id in enumerate(tiles)
    ]
    jp.log(f"phase={board.get('phase')}: attempting {len(jobs)} tile(s) with "
           f"{min(WORKERS, len(jobs))} parallel Haiku 4.5 workers: {jobs}")
    solved = 0
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(jobs))) as executor:
        futures = {
            executor.submit(_attempt_tile, task_id, temperature): task_id
            for task_id, temperature in jobs
        }
        for future in as_completed(futures):
            solved += future.result()
    jp.log(f"solver: {solved}/{len(jobs)} attempted tiles solved")


if __name__ == "__main__":
    try:
        main()
    except jp.AuthError as error:
        raise SystemExit(f"[auth] {error}")

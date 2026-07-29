from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Iterable

from models import Candidate, Evidence

from .files import resolve_task_path


class EvidenceError(ValueError):
    pass


def validate_answer(answer: str, *, max_chars: int = 1_000) -> str:
    if not isinstance(answer, str):
        raise EvidenceError("answer must be a string")
    if answer.endswith("\r\n"):
        answer = answer[:-2]
    elif answer.endswith("\n"):
        answer = answer[:-1]
    if not answer:
        raise EvidenceError("answer must not be empty")
    if "\r" in answer or "\n" in answer:
        raise EvidenceError("answer must be a single line")
    if len(answer) > max_chars:
        raise EvidenceError(f"answer must contain at most {max_chars} characters")
    if "\x00" in answer:
        raise EvidenceError("answer may not contain NUL bytes")
    return answer


def _items(name: str, values: Iterable[str], *, limit: int = 100) -> tuple[str, ...]:
    if isinstance(values, str):
        raise EvidenceError(f"{name} must be an iterable of strings, not a string")
    result = tuple(values)
    if len(result) > limit:
        raise EvidenceError(f"{name} contains more than {limit} entries")
    for value in result:
        if not isinstance(value, str) or not value or len(value) > 2_000:
            raise EvidenceError(
                f"each {name} entry must contain 1 to 2000 characters"
            )
    return result


class CandidateWriter:
    """Persist an answer and its evidence; this class has no submission capability."""

    def __init__(
        self,
        workdir: str | os.PathLike[str],
        task_id: str,
        *,
        answer_filename: str = "answer.txt",
        candidate_filename: str = "candidate.json",
        max_answer_chars: int = 1_000,
    ) -> None:
        if not isinstance(task_id, str) or not task_id or len(task_id) > 200:
            raise EvidenceError("task_id must contain 1 to 200 characters")
        self.root = Path(workdir).resolve(strict=True)
        if not self.root.is_dir():
            raise EvidenceError("task workdir is not a directory")
        self.task_id = task_id
        self.answer_filename = answer_filename
        self.candidate_filename = candidate_filename
        self.answer_path = resolve_task_path(self.root, self.answer_filename)
        self.candidate_path = resolve_task_path(self.root, self.candidate_filename)
        if (
            isinstance(max_answer_chars, bool)
            or not isinstance(max_answer_chars, int)
            or max_answer_chars < 1
        ):
            raise EvidenceError("max_answer_chars must be positive")
        self.max_answer_chars = max_answer_chars

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.new")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _candidate(
        self,
        answer: str,
        *,
        method: str,
        deterministic_checks: Iterable[str],
        independent_checks: Iterable[str],
        assumptions: Iterable[str],
        tool_errors: Iterable[str],
        input_complete: bool,
        direct_provenance: bool,
        model_temperature: float,
        elapsed_seconds: float,
        tool_turns: int,
    ) -> Candidate:
        if not isinstance(method, str) or not method or len(method) > 2_000:
            raise EvidenceError("method must contain 1 to 2000 characters")
        if isinstance(tool_turns, bool) or not isinstance(tool_turns, int) or tool_turns < 0:
            raise EvidenceError("tool_turns must be a non-negative integer")
        if (
            isinstance(model_temperature, bool)
            or not isinstance(model_temperature, (int, float))
            or not 0 <= model_temperature <= 1
        ):
            raise EvidenceError("model_temperature must be between 0 and 1")
        if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds < 0:
            raise EvidenceError("elapsed_seconds must be non-negative")
        if not isinstance(input_complete, bool) or not isinstance(
            direct_provenance, bool
        ):
            raise EvidenceError(
                "input_complete and direct_provenance must be booleans"
            )
        evidence = Evidence(
            method=method,
            deterministic_checks=_items(
                "deterministic_checks", deterministic_checks
            ),
            independent_checks=_items("independent_checks", independent_checks),
            assumptions=_items("assumptions", assumptions),
            tool_errors=_items("tool_errors", tool_errors),
            input_complete=input_complete,
            direct_provenance=direct_provenance,
        )
        return Candidate(
            task_id=self.task_id,
            answer=answer,
            answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            evidence=evidence,
            model_temperature=float(model_temperature),
            elapsed_seconds=float(elapsed_seconds),
            tool_turns=tool_turns,
        )

    def _record(self, candidate: Candidate) -> None:
        self.candidate_path = resolve_task_path(
            self.root, self.candidate_filename
        )
        payload = asdict(candidate)
        payload["schema_version"] = 1
        data = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        self._atomic_write(self.candidate_path, data)

    def write(
        self,
        answer: str,
        *,
        method: str,
        deterministic_checks: Iterable[str] = (),
        independent_checks: Iterable[str] = (),
        assumptions: Iterable[str] = (),
        tool_errors: Iterable[str] = (),
        input_complete: bool = True,
        direct_provenance: bool = True,
        model_temperature: float = 0.0,
        elapsed_seconds: float = 0.0,
        tool_turns: int = 0,
    ) -> Candidate:
        answer = validate_answer(answer, max_chars=self.max_answer_chars)
        candidate = self._candidate(
            answer,
            method=method,
            deterministic_checks=deterministic_checks,
            independent_checks=independent_checks,
            assumptions=assumptions,
            tool_errors=tool_errors,
            input_complete=input_complete,
            direct_provenance=direct_provenance,
            model_temperature=model_temperature,
            elapsed_seconds=elapsed_seconds,
            tool_turns=tool_turns,
        )
        self.answer_path = resolve_task_path(self.root, self.answer_filename)
        self.candidate_path = resolve_task_path(self.root, self.candidate_filename)
        self._atomic_write(self.answer_path, (answer + "\n").encode("utf-8"))
        self._record(candidate)
        return candidate

    def record_existing(
        self,
        *,
        method: str,
        deterministic_checks: Iterable[str] = (),
        independent_checks: Iterable[str] = (),
        assumptions: Iterable[str] = (),
        tool_errors: Iterable[str] = (),
        input_complete: bool = True,
        direct_provenance: bool = True,
        model_temperature: float = 0.0,
        elapsed_seconds: float = 0.0,
        tool_turns: int = 0,
    ) -> Candidate:
        self.answer_path = resolve_task_path(
            self.root, self.answer_filename, must_exist=True
        )
        self.candidate_path = resolve_task_path(self.root, self.candidate_filename)
        if not self.answer_path.is_file():
            raise EvidenceError(f"{self.answer_path.name} does not exist")
        raw = self.answer_path.read_bytes()
        if len(raw) > self.max_answer_chars * 4 + 2:
            raise EvidenceError("answer file exceeds configured size limit")
        try:
            answer = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceError("answer file must be UTF-8") from error
        answer = validate_answer(answer, max_chars=self.max_answer_chars)
        candidate = self._candidate(
            answer,
            method=method,
            deterministic_checks=deterministic_checks,
            independent_checks=independent_checks,
            assumptions=assumptions,
            tool_errors=tool_errors,
            input_complete=input_complete,
            direct_provenance=direct_provenance,
            model_temperature=model_temperature,
            elapsed_seconds=elapsed_seconds,
            tool_turns=tool_turns,
        )
        self._record(candidate)
        return candidate


CandidateEvidenceWriter = CandidateWriter
EvidenceWriter = CandidateWriter


def write_candidate(
    workdir: str | os.PathLike[str],
    task_id: str,
    answer: str,
    **evidence: object,
) -> Candidate:
    return CandidateWriter(workdir, task_id).write(
        answer, **evidence  # type: ignore[arg-type]
    )

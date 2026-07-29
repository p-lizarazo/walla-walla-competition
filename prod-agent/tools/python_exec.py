from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import BinaryIO

from .files import resolve_task_path

class PythonExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def render(self, max_chars: int | None = None) -> str:
        status = (
            f"timed out; exit={self.exit_code}"
            if self.timed_out
            else f"exit={self.exit_code}"
        )
        sections = [status]
        if self.stdout:
            sections.append(self.stdout)
        if self.stderr:
            sections.append(f"STDERR:\n{self.stderr}")
        text = "\n".join(sections)
        if max_chars is not None and len(text) > max_chars:
            marker = "\n...[output truncated]"
            if max_chars <= len(marker):
                return marker[:max_chars]
            return text[: max(0, max_chars - len(marker))] + marker
        return text


_SEMAPHORE_LOCK = threading.Lock()
_SHARED_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}


def shared_cpu_semaphore(slots: int) -> threading.BoundedSemaphore:
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise ValueError("CPU slots must be a positive integer")
    with _SEMAPHORE_LOCK:
        return _SHARED_SEMAPHORES.setdefault(
            slots, threading.BoundedSemaphore(slots)
        )


_POSIX_BOOTSTRAP = r"""
import sys
try:
    import resource
except ImportError:
    resource = None

if resource is not None:
    cpu_seconds = int(sys.argv[1])
    memory_bytes = int(sys.argv[2])
    if hasattr(resource, "RLIMIT_CPU"):
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        target_hard = cpu_seconds + 1
        if hard != resource.RLIM_INFINITY:
            target_hard = min(target_hard, hard)
        resource.setrlimit(
            resource.RLIMIT_CPU, (min(cpu_seconds, target_hard), target_hard)
        )
    # macOS rejects RLIMIT_AS below the interpreter's existing mapped space.
    # The hosted Linux container enforces it; local macOS relies on timeout.
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = memory_bytes
        if hard != resource.RLIM_INFINITY:
            target = min(target, hard)
        resource.setrlimit(resource.RLIMIT_AS, (target, target))
    if hasattr(resource, "RLIMIT_CORE"):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

source = sys.stdin.buffer.read()
exec(compile(source, "<solver-python>", "exec"), {"__name__": "__main__"})
"""


class _Capture:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.data = bytearray()
        self.truncated = False

    def drain(self, stream: BinaryIO) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                available = self.cap - len(self.data)
                if available > 0:
                    self.data.extend(chunk[:available])
                if len(chunk) > available:
                    self.truncated = True
        finally:
            stream.close()

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


class PythonExecutor:
    """Run bounded Python subprocesses rooted in one task work directory."""

    def __init__(
        self,
        workdir: str | os.PathLike[str],
        *,
        timeout_seconds: int = 60,
        max_output_bytes: int = 12_000,
        memory_mb: int = 512,
        cpu_slots: int = 1,
        semaphore: threading.Semaphore | None = None,
        max_code_bytes: int = 1_000_000,
    ) -> None:
        self.root = Path(workdir).resolve(strict=True)
        if not self.root.is_dir():
            raise PythonExecutionError("task workdir is not a directory")
        self.timeout_seconds = self._positive("timeout_seconds", timeout_seconds)
        self.max_output_bytes = self._positive(
            "max_output_bytes", max_output_bytes
        )
        self.memory_mb = self._positive("memory_mb", memory_mb)
        self.max_code_bytes = self._positive("max_code_bytes", max_code_bytes)
        self.semaphore = semaphore or shared_cpu_semaphore(cpu_slots)

    @staticmethod
    def _positive(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _environment(self, cwd: Path) -> dict[str, str]:
        environment = {
            "HOME": str(self.root),
            "TMPDIR": str(cwd),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PATH": os.defpath,
        }
        for name in ("LANG", "LC_ALL", "TZ"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        return environment

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows fallback
                process.kill()
        except ProcessLookupError:
            pass

    def run(
        self,
        code: str,
        *,
        timeout_seconds: int | None = None,
        cwd: str = ".",
    ) -> ExecutionResult:
        if not isinstance(code, str):
            raise PythonExecutionError("code must be a string")
        if len(code.encode("utf-8")) > self.max_code_bytes:
            raise PythonExecutionError("code exceeds configured size limit")
        timeout = min(
            self.timeout_seconds,
            self._positive(
                "timeout_seconds",
                self.timeout_seconds
                if timeout_seconds is None
                else timeout_seconds,
            ),
        )
        run_cwd = resolve_task_path(self.root, cwd, must_exist=True)
        if not run_cwd.is_dir():
            raise PythonExecutionError("execution cwd is not a directory")
        cpu_seconds = max(1, timeout)
        started = time.monotonic()
        with self.semaphore:
            command = [sys.executable, "-I", "-c", code]
            if os.name == "posix":
                command = [
                    sys.executable,
                    "-I",
                    "-c",
                    _POSIX_BOOTSTRAP,
                    str(cpu_seconds),
                    str(self.memory_mb * 1024 * 1024),
                ]
            try:
                process = subprocess.Popen(
                    command,
                    cwd=run_cwd,
                    env=self._environment(run_cwd),
                    stdin=subprocess.PIPE if os.name == "posix" else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=os.name == "posix",
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise PythonExecutionError(
                    f"could not start Python subprocess: {error}"
                ) from error
            assert process.stdout is not None and process.stderr is not None
            if process.stdin is not None:
                try:
                    process.stdin.write(code.encode("utf-8"))
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            stdout = _Capture(self.max_output_bytes)
            stderr = _Capture(self.max_output_bytes)
            threads = [
                threading.Thread(target=stdout.drain, args=(process.stdout,), daemon=True),
                threading.Thread(target=stderr.drain, args=(process.stderr,), daemon=True),
            ]
            for thread in threads:
                thread.start()
            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate(process)
                exit_code = process.wait()
            for thread in threads:
                thread.join(timeout=2)
            combined = stdout.text() + stderr.text()
            truncated = stdout.truncated or stderr.truncated
            if len(combined.encode("utf-8")) > self.max_output_bytes:
                remaining = self.max_output_bytes
                out_bytes = stdout.text().encode("utf-8")[:remaining]
                remaining -= len(out_bytes)
                err_bytes = stderr.text().encode("utf-8")[:remaining]
                out_text = out_bytes.decode("utf-8", errors="ignore")
                err_text = err_bytes.decode("utf-8", errors="ignore")
                truncated = True
            else:
                out_text = stdout.text()
                err_text = stderr.text()
        return ExecutionResult(
            exit_code=exit_code,
            stdout=out_text,
            stderr=err_text,
            timed_out=timed_out,
            truncated=truncated,
            elapsed_seconds=round(time.monotonic() - started, 6),
        )


def run_python(
    workdir: str | os.PathLike[str],
    code: str,
    *,
    timeout_seconds: int = 60,
    max_output_bytes: int = 12_000,
    memory_mb: int = 512,
    cpu_slots: int = 1,
    cwd: str = ".",
) -> str:
    executor = PythonExecutor(
        workdir,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        memory_mb=memory_mb,
        cpu_slots=cpu_slots,
    )
    return executor.run(code, cwd=cwd).render(max_output_bytes)

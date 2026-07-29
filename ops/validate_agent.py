"""Verify that agent.zip is something the hosted runner can actually boot.

`build_agent.py` checks the archive's shape. This checks its behavior: every
shipped module is compiled and imported from an extracted copy with nothing
else on the path, which is the only reliable way to catch a helper module that
was refactored away but is still imported. That failure otherwise surfaces
only as a ModuleNotFoundError in /api/agent/logs after a deploy.
"""

from __future__ import annotations

import argparse
import pathlib
import py_compile
import subprocess
import sys
import tempfile
import zipfile

MAX_COMPRESSED_BYTES = 20 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
STUB_ENV = {
    "JEOPARDY_BASE_URL": "https://validate.invalid",
    "TEAM_API_KEY": "team_validation_placeholder",
    "ANTHROPIC_BASE_URL": "https://validate.invalid/anthropic",
    "ANTHROPIC_API_KEY": "team_validation_placeholder",
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
}
FORBIDDEN_SUFFIXES = (".pyc", "practice_results.jsonl", "learnings.jsonl")


def _fail(problems: list[str], message: str) -> None:
    problems.append(message)


def check_layout(archive: zipfile.ZipFile, problems: list[str]) -> None:
    names = archive.namelist()
    if "main.py" not in names:
        _fail(problems, "main.py is missing from the zip root")
    for name in names:
        if name.startswith("/") or ".." in pathlib.PurePosixPath(name).parts:
            _fail(problems, f"unsafe archive path: {name}")
        if "__pycache__" in name or name.endswith(FORBIDDEN_SUFFIXES):
            _fail(problems, f"build artifact should not ship: {name}")
        if pathlib.PurePosixPath(name).name.startswith(".env"):
            _fail(problems, f"environment file should not ship: {name}")
    top_level = {pathlib.PurePosixPath(name).parts[0] for name in names}
    if len(top_level) == 1 and top_level.pop().endswith(("agent", "prod-agent")):
        _fail(problems, "every entry is nested in one directory; main.py must be at the root")


def check_size(path: pathlib.Path, archive: zipfile.ZipFile, problems: list[str]) -> None:
    compressed = path.stat().st_size
    uncompressed = sum(item.file_size for item in archive.infolist())
    if compressed > MAX_COMPRESSED_BYTES:
        _fail(problems, f"compressed size {compressed} exceeds 20 MB")
    if uncompressed > MAX_UNCOMPRESSED_BYTES:
        _fail(problems, f"uncompressed size {uncompressed} exceeds 200 MB")
    print(f"size: {compressed} bytes compressed, {uncompressed} bytes uncompressed")


def check_compiles(root: pathlib.Path, problems: list[str]) -> list[str]:
    modules: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True, cfile=str(path) + "c")
        except py_compile.PyCompileError as error:
            _fail(problems, f"syntax error in {path.relative_to(root)}: {error}")
            continue
        relative = path.relative_to(root)
        if relative.name == "__init__.py":
            modules.append(".".join(relative.parts[:-1]))
        else:
            modules.append(".".join(relative.parts)[: -len(".py")])
    for stale in root.rglob("*.pyc"):
        stale.unlink()
    return [module for module in modules if module]


def check_imports(root: pathlib.Path, modules: list[str], problems: list[str]) -> None:
    """Import each module with only the extracted zip on sys.path."""
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=root,
            env=dict(STUB_ENV, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            continue
        stderr = result.stderr.strip().splitlines()
        detail = stderr[-1] if stderr else "unknown import failure"
        _fail(problems, f"`import {module}` failed: {detail}")


def check_entrypoint(root: pathlib.Path, problems: list[str]) -> None:
    """The runner executes `python -u main.py`, so prove that call path works."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; sys.argv=['main.py']; "
            "runpy.run_path('main.py', run_name='__not_main__')",
        ],
        cwd=root,
        env=dict(STUB_ENV, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip().splitlines()
        _fail(problems, f"main.py is not loadable: {stderr[-1] if stderr else '?'}")


def validate(path: pathlib.Path) -> int:
    problems: list[str] = []
    if not path.is_file():
        print(f"validate: {path} does not exist", file=sys.stderr)
        return 1
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            print("validate: archive is corrupt", file=sys.stderr)
            return 1
        check_layout(archive, problems)
        check_size(path, archive, problems)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive.extractall(root)
            modules = check_compiles(root, problems)
            check_imports(root, modules, problems)
            check_entrypoint(root, problems)
            print(f"validated {len(modules)} module(s): {', '.join(sorted(modules))}")
    if problems:
        print("\nagent.zip is NOT submittable:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nagent.zip is submittable: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        nargs="?",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "agent.zip"),
    )
    return validate(pathlib.Path(parser.parse_args().archive).resolve())


if __name__ == "__main__":
    sys.exit(main())

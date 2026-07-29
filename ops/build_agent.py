"""Package prod-agent/ into the agent.zip the event runner expects.

The runner unzips the archive and executes `python -u main.py` from its root,
so the agent's modules are flattened to the archive root here; the repository
keeps them in `prod-agent/` for everything else.

Packaging is an exclude list, not an allowlist, on purpose. A module that is
accidentally *omitted* kills the agent on boot with ModuleNotFoundError, while
an extra pure-Python module costs a few kilobytes. So new runtime modules ship
automatically, and `validate_agent.py` is what proves the result actually
boots.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPOSITORY / "prod-agent"
DEFAULT_OUTPUT = REPOSITORY / "agent.zip"

EXCLUDED_DIRECTORIES = frozenset({"tests", "evals", "__pycache__", ".git", ".venv"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".jsonl", ".log", ".zip", ".md")
MAX_COMPRESSED_BYTES = 20 * 1024 * 1024


def is_excluded(path: pathlib.Path, relative: pathlib.PurePath) -> bool:
    directories = relative.parts[:-1]
    if EXCLUDED_DIRECTORIES.intersection(directories):
        return True
    if any(part.startswith(".") for part in directories):
        return True
    name = relative.name
    if name.startswith("."):
        return True
    if name.endswith(EXCLUDED_SUFFIXES):
        return True
    return not path.is_file()


def collect(source: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    entries: list[tuple[pathlib.Path, str]] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if is_excluded(path, relative):
            continue
        entries.append((path, relative.as_posix()))
    return entries


def build(source: pathlib.Path, output: pathlib.Path) -> int:
    if not source.is_dir():
        print(f"build: {source} does not exist", file=sys.stderr)
        return 1
    entries = collect(source)
    names = {name for _, name in entries}
    if "main.py" not in names:
        print("build: main.py must exist at the root of prod-agent/", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path, name in entries:
            # Fixed timestamp and mode keep the archive byte-reproducible.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    size = output.stat().st_size
    if size > MAX_COMPRESSED_BYTES:
        print(f"build: {size} bytes exceeds the 20 MB limit", file=sys.stderr)
        return 1
    for name in sorted(names):
        print(f"  + {name}")
    print(f"built {output} ({size} bytes, {len(entries)} files)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()
    return build(
        pathlib.Path(arguments.source).resolve(),
        pathlib.Path(arguments.output).resolve(),
    )


if __name__ == "__main__":
    sys.exit(main())

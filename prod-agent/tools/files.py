from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Iterable
import zipfile


class FileToolError(ValueError):
    """Raised when a file operation is unsafe or exceeds a configured bound."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int


@dataclass(frozen=True)
class SearchMatch:
    path: str
    line: int
    text: str


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _inside(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def resolve_task_path(
    workdir: str | os.PathLike[str],
    name: str | os.PathLike[str] = ".",
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a relative task path without permitting traversal or symlink escape."""
    root = Path(workdir).resolve(strict=True)
    if not root.is_dir():
        raise FileToolError("task workdir is not a directory")
    raw = os.fspath(name)
    if "\x00" in raw:
        raise FileToolError("paths may not contain NUL bytes")
    requested = Path(raw)
    if requested.is_absolute():
        raise FileToolError("task paths must be relative")
    try:
        path = (root / requested).resolve(strict=must_exist)
    except (FileNotFoundError, RuntimeError, OSError) as error:
        raise FileToolError(f"cannot resolve task path {raw!r}: {error}") from error
    if not _inside(root, path):
        raise FileToolError("paths must stay inside the task workdir")
    return path


def _bounded_text(text: str, limit: int) -> tuple[str, bool]:
    limit = _positive("max_output_chars", limit)
    if len(text) <= limit:
        return text, False
    marker = "\n...[output truncated]"
    if limit <= len(marker):
        return marker[:limit], True
    return text[: max(0, limit - len(marker))] + marker, True


def _relative(root: Path, path: Path) -> str:
    value = path.relative_to(root).as_posix()
    return value or "."


class TaskFiles:
    """Bounded, read-only task-file inspection plus safe archive extraction."""

    def __init__(
        self,
        workdir: str | os.PathLike[str],
        *,
        max_output_chars: int = 12_000,
        max_entries: int = 500,
        max_read_bytes: int = 2_000_000,
        max_search_files: int = 500,
        max_search_matches: int = 200,
        max_search_bytes: int = 20_000_000,
        max_archive_entries: int = 2_000,
        max_extract_bytes: int = 100_000_000,
    ) -> None:
        self.root = Path(workdir).resolve(strict=True)
        if not self.root.is_dir():
            raise FileToolError("task workdir is not a directory")
        self.max_output_chars = _positive("max_output_chars", max_output_chars)
        self.max_entries = _positive("max_entries", max_entries)
        self.max_read_bytes = _positive("max_read_bytes", max_read_bytes)
        self.max_search_files = _positive("max_search_files", max_search_files)
        self.max_search_matches = _positive(
            "max_search_matches", max_search_matches
        )
        self.max_search_bytes = _positive("max_search_bytes", max_search_bytes)
        self.max_archive_entries = _positive(
            "max_archive_entries", max_archive_entries
        )
        self.max_extract_bytes = _positive("max_extract_bytes", max_extract_bytes)

    def resolve(
        self, name: str | os.PathLike[str] = ".", *, must_exist: bool = False
    ) -> Path:
        return resolve_task_path(self.root, name, must_exist=must_exist)

    def list(
        self,
        path: str = ".",
        *,
        recursive: bool = True,
        pattern: str | None = None,
        limit: int | None = None,
    ) -> tuple[FileEntry, ...]:
        base = self.resolve(path, must_exist=True)
        if not base.is_dir():
            raise FileToolError(f"{path!r} is not a directory")
        cap = min(
            self.max_entries,
            _positive("limit", self.max_entries if limit is None else limit),
        )
        iterator: Iterable[Path] = base.rglob("*") if recursive else base.iterdir()
        entries: list[FileEntry] = []
        scanned = 0
        scan_cap = max(cap * 20, cap)
        for candidate in iterator:
            scanned += 1
            if scanned > scan_cap:
                break
            if len(entries) >= cap:
                break
            try:
                resolved = candidate.resolve(strict=True)
                info = candidate.lstat()
            except (FileNotFoundError, OSError):
                continue
            if not _inside(self.root, resolved) or not stat.S_ISREG(info.st_mode):
                continue
            relative = _relative(self.root, resolved)
            if pattern and not fnmatch.fnmatch(relative, pattern):
                continue
            entries.append(FileEntry(relative, info.st_size))
        return tuple(entries)

    def list_text(self, *args: object, **kwargs: object) -> str:
        entries = self.list(*args, **kwargs)
        text = "\n".join(f"{entry.path} ({entry.size} bytes)" for entry in entries)
        text, _ = _bounded_text(text or "(no files)", self.max_output_chars)
        return text

    def read(
        self,
        name: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        encoding: str = "utf-8",
    ) -> str:
        start_line = _positive("start_line", start_line)
        if end_line is not None:
            end_line = _positive("end_line", end_line)
            if end_line < start_line:
                raise FileToolError("end_line must not precede start_line")
        path = self.resolve(name, must_exist=True)
        if not path.is_file():
            raise FileToolError(f"{name!r} is not a file")
        size = path.stat().st_size
        if size > self.max_read_bytes:
            raise FileToolError(
                f"{name!r} is {size} bytes; read limit is {self.max_read_bytes}"
            )
        try:
            with path.open("r", encoding=encoding, errors="replace") as stream:
                selected = [
                    line.rstrip("\r\n")
                    for number, line in enumerate(stream, 1)
                    if number >= start_line
                    and (end_line is None or number <= end_line)
                ]
        except (LookupError, OSError) as error:
            raise FileToolError(f"cannot read {name!r}: {error}") from error
        text, _ = _bounded_text("\n".join(selected), self.max_output_chars)
        return text

    def search(
        self,
        query: str,
        *,
        path: str = ".",
        glob: str = "*",
        regex: bool = False,
        case_sensitive: bool = True,
        max_matches: int | None = None,
    ) -> tuple[SearchMatch, ...]:
        if not isinstance(query, str) or not query or len(query) > 1_000:
            raise FileToolError("query must contain 1 to 1000 characters")
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            expression = re.compile(query if regex else re.escape(query), flags)
        except re.error as error:
            raise FileToolError(f"invalid regular expression: {error}") from error
        cap = min(
            self.max_search_matches,
            _positive(
                "max_matches",
                self.max_search_matches if max_matches is None else max_matches,
            ),
        )
        matches: list[SearchMatch] = []
        files = self.list(
            path,
            recursive=True,
            pattern=glob,
            limit=self.max_search_files,
        )
        searched_bytes = 0
        for entry in files:
            if entry.size > self.max_read_bytes:
                continue
            if searched_bytes + entry.size > self.max_search_bytes:
                break
            searched_bytes += entry.size
            candidate = self.resolve(entry.path, must_exist=True)
            try:
                with candidate.open(
                    "r", encoding="utf-8", errors="replace"
                ) as stream:
                    for number, line in enumerate(stream, 1):
                        if expression.search(line):
                            text = line.rstrip("\r\n")
                            matches.append(
                                SearchMatch(
                                    entry.path,
                                    number,
                                    text[: min(2_000, self.max_output_chars)],
                                )
                            )
                            if len(matches) >= cap:
                                return tuple(matches)
            except (OSError, UnicodeError):
                continue
        return tuple(matches)

    def search_text(self, *args: object, **kwargs: object) -> str:
        matches = self.search(*args, **kwargs)
        text = "\n".join(
            f"{match.path}:{match.line}:{match.text}" for match in matches
        )
        text, _ = _bounded_text(text or "(no matches)", self.max_output_chars)
        return text

    @staticmethod
    def _member_parts(name: str) -> tuple[str, ...]:
        normalized = name.replace("\\", "/")
        if "\x00" in normalized:
            raise FileToolError("archive member contains a NUL byte")
        member = PurePosixPath(normalized)
        if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
            raise FileToolError(f"unsafe archive member: {name!r}")
        if member.parts and member.parts[0].endswith(":"):
            raise FileToolError(f"unsafe archive member: {name!r}")
        return member.parts

    def _archive_kind(self, archive: Path) -> str:
        if zipfile.is_zipfile(archive):
            return "zip"
        try:
            if tarfile.is_tarfile(archive):
                return "tar"
        except OSError:
            pass
        raise FileToolError("supported archives are ZIP and TAR variants")

    def list_archive(
        self, name: str, *, limit: int | None = None
    ) -> tuple[FileEntry, ...]:
        archive = self.resolve(name, must_exist=True)
        if not archive.is_file():
            raise FileToolError(f"{name!r} is not a file")
        cap = min(
            self.max_archive_entries,
            _positive(
                "limit", self.max_archive_entries if limit is None else limit
            ),
        )
        entries: list[FileEntry] = []
        try:
            if self._archive_kind(archive) == "zip":
                with zipfile.ZipFile(archive) as bundle:
                    for seen, info in enumerate(bundle.infolist(), 1):
                        if seen > self.max_archive_entries:
                            break
                        if len(entries) >= cap:
                            break
                        if info.is_dir():
                            continue
                        self._member_parts(info.filename)
                        mode = info.external_attr >> 16
                        file_type = stat.S_IFMT(mode)
                        if file_type not in (0, stat.S_IFREG):
                            raise FileToolError(
                                f"archive member is not a regular file: {info.filename!r}"
                            )
                        entries.append(FileEntry(info.filename, info.file_size))
            else:
                with tarfile.open(archive, mode="r:*") as bundle:
                    for seen, info in enumerate(bundle, 1):
                        if seen > self.max_archive_entries:
                            break
                        if len(entries) >= cap:
                            break
                        if info.isdir():
                            continue
                        self._member_parts(info.name)
                        if not info.isfile():
                            raise FileToolError(
                                f"archive member is not a regular file: {info.name!r}"
                            )
                        entries.append(FileEntry(info.name, info.size))
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as error:
            raise FileToolError(f"cannot list archive {name!r}: {error}") from error
        return tuple(entries)

    def archive_text(self, name: str, *, limit: int | None = None) -> str:
        entries = self.list_archive(name, limit=limit)
        text = "\n".join(f"{entry.path} ({entry.size} bytes)" for entry in entries)
        text, _ = _bounded_text(text or "(empty archive)", self.max_output_chars)
        return text

    def _destination(self, destination: str, member_name: str) -> Path:
        parts = self._member_parts(member_name)
        base = self.resolve(destination)
        target = base.joinpath(*parts).resolve(strict=False)
        if not _inside(self.root, target):
            raise FileToolError(f"unsafe archive member: {member_name!r}")
        return target

    @staticmethod
    def _copy_bounded(source: io.BufferedReader, target: Path, remaining: int) -> int:
        written = 0
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as output:
                while True:
                    chunk = source.read(min(64 * 1024, remaining - written + 1))
                    if not chunk:
                        return written
                    written += len(chunk)
                    if written > remaining:
                        raise FileToolError("archive exceeds extraction byte limit")
                    output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def extract_archive(
        self,
        name: str,
        *,
        destination: str = ".",
        members: Iterable[str] | None = None,
        overwrite: bool = False,
    ) -> tuple[str, ...]:
        archive = self.resolve(name, must_exist=True)
        if not archive.is_file():
            raise FileToolError(f"{name!r} is not a file")
        base = self.resolve(destination)
        base.mkdir(parents=True, exist_ok=True)
        if isinstance(members, (str, bytes)):
            raise FileToolError("members must be an iterable of archive names")
        wanted = set(members) if members is not None else None
        extracted: list[str] = []
        total = 0
        seen = 0

        def prepare(member_name: str, declared_size: int) -> Path | None:
            nonlocal total
            if wanted is not None and member_name not in wanted:
                return None
            if len(extracted) >= self.max_archive_entries:
                raise FileToolError("archive exceeds extraction entry limit")
            total += declared_size
            if total > self.max_extract_bytes:
                raise FileToolError("archive exceeds extraction byte limit")
            target = self._destination(destination, member_name)
            if target.exists():
                if not overwrite:
                    raise FileToolError(f"refusing to overwrite {member_name!r}")
                if not target.is_file() or target.is_symlink():
                    raise FileToolError(f"unsafe overwrite target: {member_name!r}")
                target.unlink()
            return target

        try:
            if self._archive_kind(archive) == "zip":
                with zipfile.ZipFile(archive) as bundle:
                    for info in bundle.infolist():
                        seen += 1
                        if seen > self.max_archive_entries:
                            raise FileToolError(
                                "archive exceeds extraction entry limit"
                            )
                        if info.is_dir():
                            continue
                        self._member_parts(info.filename)
                        mode = info.external_attr >> 16
                        file_type = stat.S_IFMT(mode)
                        if file_type not in (0, stat.S_IFREG):
                            raise FileToolError(
                                f"archive member is not a regular file: {info.filename!r}"
                            )
                        target = prepare(info.filename, info.file_size)
                        if target is None:
                            continue
                        with bundle.open(info, "r") as source:
                            actual = self._copy_bounded(
                                source, target, self.max_extract_bytes - (total - info.file_size)
                            )
                        if actual != info.file_size:
                            raise FileToolError(
                                f"archive member size changed: {info.filename!r}"
                            )
                        extracted.append(_relative(self.root, target))
            else:
                with tarfile.open(archive, mode="r:*") as bundle:
                    for info in bundle:
                        seen += 1
                        if seen > self.max_archive_entries:
                            raise FileToolError(
                                "archive exceeds extraction entry limit"
                            )
                        if info.isdir():
                            continue
                        self._member_parts(info.name)
                        if not info.isfile():
                            raise FileToolError(
                                f"archive member is not a regular file: {info.name!r}"
                            )
                        target = prepare(info.name, info.size)
                        if target is None:
                            continue
                        source = bundle.extractfile(info)
                        if source is None:
                            raise FileToolError(f"cannot read archive member {info.name!r}")
                        with source:
                            actual = self._copy_bounded(
                                source, target, self.max_extract_bytes - (total - info.size)
                            )
                        if actual != info.size:
                            raise FileToolError(
                                f"archive member size changed: {info.name!r}"
                            )
                        extracted.append(_relative(self.root, target))
        except FileToolError:
            raise
        except (zipfile.BadZipFile, tarfile.TarError, OSError, RuntimeError) as error:
            raise FileToolError(f"cannot extract archive {name!r}: {error}") from error
        if wanted is not None:
            missing = wanted.difference(
                entry.path for entry in self.list_archive(name)
            )
            if missing:
                raise FileToolError(
                    "archive members not found: " + ", ".join(sorted(missing))
                )
        return tuple(extracted)


def list_files(workdir: str | os.PathLike[str], **kwargs: object) -> str:
    return TaskFiles(workdir).list_text(**kwargs)


def read_file(
    workdir: str | os.PathLike[str], name: str, **kwargs: object
) -> str:
    return TaskFiles(workdir).read(name, **kwargs)


def search_files(
    workdir: str | os.PathLike[str], query: str, **kwargs: object
) -> str:
    return TaskFiles(workdir).search_text(query, **kwargs)


def list_archive(
    workdir: str | os.PathLike[str], name: str, **kwargs: object
) -> str:
    return TaskFiles(workdir).archive_text(name, **kwargs)


def extract_archive(
    workdir: str | os.PathLike[str], name: str, **kwargs: object
) -> tuple[str, ...]:
    return TaskFiles(workdir).extract_archive(name, **kwargs)

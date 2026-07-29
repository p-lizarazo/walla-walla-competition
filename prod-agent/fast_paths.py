from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from html.parser import HTMLParser
import io
import pathlib
import re
import time
from urllib.parse import urljoin
import zipfile

from models import Candidate, Evidence, TaskDetail
from tools.web import EventWebSession


@dataclass(frozen=True)
class FastPathResult:
    matched: bool
    candidate: Candidate | None = None
    error: str | None = None


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[dict[str, object]] = []
        self.links: list[str] = []
        self._form: dict[str, object] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "form":
            self._form = {
                "action": values.get("action") or "",
                "method": (values.get("method") or "GET").upper(),
                "inputs": {},
            }
            self.forms.append(self._form)
        elif tag == "input" and self._form is not None:
            name = values.get("name")
            if name:
                inputs = self._form["inputs"]
                assert isinstance(inputs, dict)
                inputs[name] = values.get("value") or ""
        elif tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._form = None


class FastPathSolver:
    """Deterministic solvers for repeated competition prompt templates."""

    _TOKEN = re.compile(r"\bVAULT-[A-Z0-9]{12}\b")

    def classify(self, task: TaskDetail) -> str | None:
        for handler in self._handlers():
            if handler(task, None, None):
                return handler.__name__.removeprefix("_")
        return None

    def _handlers(self):
        return (
            self._policy_section,
            self._leaderboard,
            self._recurrence_sum,
            self._infeasible_hash_work,
            self._regional_sales,
            self._encrypted_bundle,
            self._members_vault,
        )

    def solve(
        self,
        task: TaskDetail,
        workdir: pathlib.Path,
        web: EventWebSession,
    ) -> FastPathResult:
        started = time.monotonic()
        for handler in self._handlers():
            if not handler(task, None, None):
                continue
            try:
                answer, checks = handler(task, workdir, web)
                elapsed = round(time.monotonic() - started, 3)
                return FastPathResult(
                    matched=True,
                    candidate=self._candidate(
                        task, answer, handler.__name__, checks, elapsed
                    ),
                )
            except Exception as error:
                return FastPathResult(
                    matched=True,
                    error=f"{type(error).__name__}: {error}",
                )
        return FastPathResult(matched=False)

    @staticmethod
    def _candidate(
        task: TaskDetail,
        answer: str,
        method: str,
        checks: tuple[str, ...],
        elapsed: float,
    ) -> Candidate:
        answer = answer.strip()
        if not answer or "\n" in answer:
            raise ValueError("fast-path answer must be one nonempty line")
        return Candidate(
            task_id=task.id,
            answer=answer,
            answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            evidence=Evidence(
                method=f"deterministic fast path: {method}",
                deterministic_checks=checks,
                input_complete=True,
                direct_provenance=True,
            ),
            model_temperature=0.0,
            elapsed_seconds=elapsed,
            tool_turns=0,
        )

    @staticmethod
    def _policy_section(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"maximum reimbursement for a single claim.*?"
            r"under Section\s+(\d+\.\d+)",
            task.prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return False
        if workdir is None:
            return True
        section = match.group(1)
        path = workdir / task.files[0]
        text = path.read_text(encoding="utf-8", errors="replace")
        block_match = re.search(
            rf"(?ms)^Section\s+{re.escape(section)}\b.*?"
            rf"(?=^Section\s+\d+\.\d+\b|\Z)",
            text,
        )
        if not block_match:
            raise ValueError(f"Section {section} was not found")
        values = re.findall(
            r"maximum reimbursement for a single claim.*?\$\s*([\d,]+)",
            block_match.group(0),
            re.IGNORECASE | re.DOTALL,
        )
        if len(values) != 1:
            raise ValueError("target section did not contain exactly one limit")
        answer = values[0].replace(",", "")
        return answer, (
            f"isolated Section {section} by heading boundaries",
            "found exactly one reimbursement limit in the target section",
            "removed currency punctuation without changing the numeric value",
        )

    @staticmethod
    def _leaderboard(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"leaderboard\((\[.*?\])\)\s+return\?",
            task.prompt,
            re.DOTALL,
        )
        if not match:
            return False
        if workdir is None:
            return True
        pairs = ast.literal_eval(match.group(1))
        if (
            not isinstance(pairs, list)
            or not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], (int, float))
                for item in pairs
            )
        ):
            raise ValueError("leaderboard input has an unexpected shape")
        spec = (workdir / "spec.md").read_text(
            encoding="utf-8", errors="replace"
        ).lower()
        if (
            "score descending" not in spec
            or "alphabetically ascending" not in spec
        ):
            raise ValueError("leaderboard specification is unfamiliar")
        result = [name for name, _ in sorted(pairs, key=lambda item: (-item[1], item[0]))]
        return repr(result), (
            "parsed the literal input with ast.literal_eval",
            "applied score-descending and name-ascending ordering from spec.md",
            "rendered the result with Python repr",
        )

    @staticmethod
    def _recurrence_sum(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        pattern = re.compile(
            r"x0\s*=\s*(\d+).*?"
            r"x_\(n\+1\)\s*=\s*\((\d+)\s*\*\s*x_n\s*\+\s*(\d+)\)"
            r"\s*mod\s*(\d+).*?"
            r"apply the recurrence\s+([\d,]+)\s+times.*?"
            r"divisible by\s+(\d+)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(task.prompt)
        if not match:
            return False
        if workdir is None:
            return True
        value, multiplier, increment, modulus, count, divisor = (
            int(group.replace(",", "")) for group in match.groups()
        )
        total = 0
        divisible_count = 0
        for _ in range(count):
            value = (multiplier * value + increment) % modulus
            if value % divisor == 0:
                total += value
                divisible_count += 1
        return str(total), (
            f"iterated the recurrence exactly {count} times",
            f"checked divisibility for every generated term ({divisible_count} matched)",
            "summed with Python arbitrary-precision integers",
        )

    @staticmethod
    def _infeasible_hash_work(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"SHA-256 digest begins with\s+(\d+)\s+zero BITS",
            task.prompt,
            re.IGNORECASE,
        )
        if (
            not match
            or "IMPOSSIBLE" not in task.prompt
            or "WORKFACTOR" not in task.prompt
        ):
            return False
        bits = int(match.group(1))
        if bits < 60:
            return False
        if workdir is None:
            return True
        return "IMPOSSIBLE WORKFACTOR", (
            f"a random SHA-256 prefix of {bits} zero bits needs 2^{bits} trials on average",
            "the prompt explicitly assigns infeasible expected compute to WORKFACTOR",
            "the threshold is far beyond the event runtime and hardware budget",
        )

    @staticmethod
    def _regional_sales(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        if (
            "regional_sales.zip" not in task.prompt
            or "usd_per_unit" not in task.prompt
            or "FIRST occurrence" not in task.prompt
        ):
            return False
        if workdir is None:
            return True
        archive_path = workdir / "regional_sales.zip"
        revenue_names = {"amount", "amt", "value", "gross"}
        id_names = {"txn_id", "order_id", "ref", "record_id"}
        seen: set[str] = set()
        total = Decimal("0")
        kept = 0
        with zipfile.ZipFile(archive_path) as archive:
            fx_rows = csv.DictReader(
                io.StringIO(archive.read("fx.csv").decode("utf-8-sig"))
            )
            rates = {
                row["region"].strip(): Decimal(row["usd_per_unit"].strip())
                for row in fx_rows
            }
            data_names = sorted(
                name
                for name in archive.namelist()
                if re.fullmatch(r"sales_[A-Z]{2}_\d+\.csv", pathlib.PurePosixPath(name).name)
            )
            if not data_names:
                raise ValueError("archive contains no regional sales files")
            for name in data_names:
                region_match = re.search(r"sales_([A-Z]{2})_", pathlib.PurePosixPath(name).name)
                assert region_match is not None
                region = region_match.group(1)
                lines = archive.read(name).decode("utf-8-sig").splitlines()
                header_index = next(
                    index
                    for index, line in enumerate(lines)
                    if not line.startswith("#")
                )
                rows = csv.reader(lines[header_index:])
                header = next(rows)
                revenue_columns = [
                    index
                    for index, column in enumerate(header)
                    if column.strip() in revenue_names
                ]
                id_columns = [
                    index
                    for index, column in enumerate(header)
                    if column.strip() in id_names
                ]
                if len(revenue_columns) != 1 or len(id_columns) != 1:
                    raise ValueError(f"{name} has ambiguous required columns")
                revenue_index = revenue_columns[0]
                id_index = id_columns[0]
                for row in rows:
                    if len(row) != len(header) or not row or row[0] == "TOTAL":
                        continue
                    transaction_id = row[id_index].strip()
                    if transaction_id in seen:
                        continue
                    seen.add(transaction_id)
                    amount = Decimal(
                        row[revenue_index].strip().replace(",", "")
                    )
                    total += amount * rates[region]
                    kept += 1
        answer = format(
            total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f"
        )
        return answer, (
            f"processed {len(data_names)} files in alphabetical order",
            f"deduplicated transaction ids before conversion ({kept} rows kept)",
            "used Decimal arithmetic and rounded only the final USD total",
        )

    @classmethod
    def _encrypted_bundle(
        cls,
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        if (
            "bundle.zip" not in task.prompt
            or "start with note.txt" not in task.prompt
        ):
            return False
        if workdir is None:
            return True
        with zipfile.ZipFile(workdir / "bundle.zip") as archive:
            note = archive.read("note.txt").decode("utf-8", errors="replace")
            if (
                "largest .dat file" not in note
                or "reversed" not in note
                or "XOR-encrypted" not in note
            ):
                raise ValueError("bundle instructions are unfamiliar")
            data_files = [
                info for info in archive.infolist() if info.filename.endswith(".dat")
            ]
            if not data_files:
                raise ValueError("bundle contains no .dat files")
            largest = max(data_files, key=lambda info: info.file_size)
            password = pathlib.PurePosixPath(largest.filename).stem[::-1].encode()
            encrypted = archive.read("inner.blob")
        decrypted = bytes(
            byte ^ password[index % len(password)]
            for index, byte in enumerate(encrypted)
        )
        with zipfile.ZipFile(io.BytesIO(decrypted)) as inner:
            text = "\n".join(
                inner.read(name).decode("utf-8", errors="replace")
                for name in inner.namelist()
            )
        tokens = cls._TOKEN.findall(text)
        if len(tokens) != 1:
            raise ValueError("decrypted archive did not contain exactly one token")
        return tokens[0], (
            f"selected the largest data file ({largest.filename})",
            "derived the repeating XOR password exactly as note.txt specified",
            "opened the decrypted ZIP and found exactly one VAULT token",
        )

    @classmethod
    def _members_vault(
        cls,
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"(https?://\S+).*?username:\s*(\S+)\s+password:\s*(\S+)",
            task.prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if not match or "members-only vault" not in task.prompt.lower():
            return False
        if workdir is None:
            return True
        assert web is not None
        start_url, username, password = match.groups()
        first = web.get(start_url)
        parser = _PageParser()
        parser.feed(first.text)
        if not parser.forms:
            raise ValueError("login page contained no form")
        form = next(
            (
                item
                for item in parser.forms
                if "login" in str(item["action"]).lower()
            ),
            parser.forms[0],
        )
        fields = dict(form["inputs"])
        username_field = next(
            (name for name in fields if "user" in name.lower()), "username"
        )
        password_field = next(
            (name for name in fields if "pass" in name.lower()), "password"
        )
        fields[username_field] = username
        fields[password_field] = password
        action = urljoin(first.url, str(form["action"]))
        logged_in = web.post(action, form=fields)
        pages = [logged_in]
        token = cls._TOKEN.search(logged_in.text)
        if token is None:
            parser = _PageParser()
            parser.feed(logged_in.text)
            vault_links = [
                urljoin(logged_in.url, link)
                for link in parser.links
                if "vault" in link.lower()
            ]
            if not vault_links:
                vault_links = [urljoin(logged_in.url, "vault")]
            pages.append(web.get(vault_links[0]))
            token = cls._TOKEN.search(pages[-1].text)
        if token is None:
            raise ValueError("authenticated pages contained no VAULT token")
        return token.group(0), (
            "loaded the login form and preserved its hidden fields",
            "submitted the supplied credentials in a persistent cookie session",
            "opened the vault and extracted a format-validated token",
        )

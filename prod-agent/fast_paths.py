from __future__ import annotations

import ast
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
from html.parser import HTMLParser
import io
import json
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
            self._two_hops,
            self._minute_details,
            self._living_document,
            self._leaderboard,
            self._recurrence_sum,
            self._infeasible_hash_work,
            self._dirty_ledger,
            self._event_horizon,
            self._join_the_dots,
            self._session_hunter,
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
    def _two_hops(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        clause_match = re.search(
            r"Clause\s+(\d+\.\d+)\s+defines the monthly service fee",
            task.prompt,
            re.IGNORECASE,
        )
        if not clause_match or "contract.txt" not in task.files:
            return False
        if workdir is None:
            return True
        text = (workdir / "contract.txt").read_text(
            encoding="utf-8", errors="replace"
        )

        def clause(number: str) -> str:
            match = re.search(
                rf"(?ms)^Clause\s+{re.escape(number)}\s+-.*?"
                rf"(?=^Clause\s+\d+\.\d+\s+-|^Appendix\s+[A-Z]\b|\Z)",
                text,
            )
            if match is None:
                raise ValueError(f"Clause {number} was not found")
            return match.group(0)

        fee_clause = clause(clause_match.group(1))
        formula = re.search(
            r"base rate listed in Appendix\s+([A-Z]).*?tier\s+"
            r"([A-Za-z]+)\s+customers.*?multiplied by the adjustment "
            r"multiplier set forth in\s+Clause\s+(\d+\.\d+)",
            fee_clause,
            re.IGNORECASE | re.DOTALL,
        )
        if formula is None:
            raise ValueError("monthly fee formula is unfamiliar")
        appendix_letter, tier, multiplier_clause = formula.groups()
        appendix = re.search(
            rf"(?ms)^Appendix\s+{appendix_letter}\b.*?"
            rf"(?=^Appendix\s+[A-Z]\b|\Z)",
            text,
        )
        if appendix is None:
            raise ValueError(f"Appendix {appendix_letter} was not found")
        rate_match = re.search(
            rf"Tier\s+{re.escape(tier)}:\s*\$\s*([\d,]+(?:\.\d+)?)",
            appendix.group(0),
            re.IGNORECASE,
        )
        multiplier_match = re.search(
            r"adjustment multiplier.*?(\d+(?:\.\d+)?)",
            clause(multiplier_clause),
            re.IGNORECASE | re.DOTALL,
        )
        if rate_match is None or multiplier_match is None:
            raise ValueError("fee rate or multiplier was not found")
        rate = Decimal(rate_match.group(1).replace(",", ""))
        multiplier = Decimal(multiplier_match.group(1))
        answer = format(
            (rate * multiplier).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            "f",
        )
        return answer, (
            f"isolated fee formula in Clause {clause_match.group(1)}",
            f"resolved tier {tier} in Appendix {appendix_letter}",
            f"resolved multiplier in Clause {multiplier_clause} and multiplied exactly",
        )

    @staticmethod
    def _minute_details(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"assigned to ([A-Za-z]+) during ([A-Za-z]+) (\d{4})",
            task.prompt,
            re.IGNORECASE,
        )
        if not match or not any(
            name.startswith("notes_") and name.endswith(".md")
            for name in task.files
        ):
            return False
        if workdir is None:
            return True
        person, month_name, year_text = match.groups()
        month = datetime.strptime(month_name, "%B").month
        filename = f"notes_{int(year_text):04d}-{month:02d}.md"
        if filename not in task.files:
            raise ValueError(f"expected notes file is missing: {filename}")
        text = (workdir / filename).read_text(
            encoding="utf-8", errors="replace"
        )
        tickets = set(
            re.findall(
                rf"(?im)^\s*-\s*{re.escape(person)}\s*:.*?"
                r"\b([A-Z]{2}-\d{4})\b",
                text,
            )
        )
        if len(tickets) != 1:
            raise ValueError(
                "target month did not contain exactly one matching ticket"
            )
        answer = next(iter(tickets))
        return answer, (
            f"selected only {filename} from the twelve monthly files",
            f"matched action-item assignee {person!r} exactly",
            "found exactly one ticket with the required XX-1234 format",
        )

    @staticmethod
    def _living_document(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"limit.*?for Clause\s+(\d+\.\d+)\s+on\s+(\d{4}-\d{2}-\d{2})",
            task.prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if (
            not match
            or "regulation.txt" not in task.files
            or "AMENDMENTS" not in task.prompt
            or "revoke" not in task.prompt.lower()
        ):
            return False
        if workdir is None:
            return True
        clause_number, query_date_text = match.groups()
        query_date = datetime.strptime(query_date_text, "%Y-%m-%d").date()
        text = (workdir / "regulation.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        original_block = re.search(
            rf"(?ms)^Clause\s+{re.escape(clause_number)}\b.*?"
            rf"(?=^Clause\s+\d+\.\d+\b|^AMENDMENTS\b|\Z)",
            text,
        )
        if original_block is None:
            raise ValueError(f"Clause {clause_number} was not found")
        original_limit = re.search(
            r"limit of\s*\$\s*([\d,]+)",
            original_block.group(0),
            re.IGNORECASE,
        )
        if original_limit is None:
            raise ValueError("original clause limit was not found")
        amendments_text = text.split("AMENDMENTS", 1)
        if len(amendments_text) != 2:
            raise ValueError("AMENDMENTS section was not found")
        amendments: dict[int, dict[str, object]] = {}
        pattern = re.compile(
            r"(?ms)^\s*Amendment\s+(\d+)\s+"
            r"\(published\s+(\d{4}-\d{2}-\d{2});\s*"
            r"effective\s+(\d{4}-\d{2}-\d{2})\):\s*"
            r"(.*?)(?=^\s*Amendment\s+\d+\s+\(|\Z)"
        )
        for amendment_match in pattern.finditer(amendments_text[1]):
            number = int(amendment_match.group(1))
            action = amendment_match.group(4).strip()
            revoke_match = re.search(
                r"Amendment\s+(\d+)\s+is\s+hereby\s+revoked",
                action,
                re.IGNORECASE,
            )
            limit_match = re.search(
                rf"Clause\s+{re.escape(clause_number)}\s+is\s+amended.*?"
                r"limit(?:\s+of|\s+is)\s*\$\s*([\d,]+)",
                action,
                re.IGNORECASE | re.DOTALL,
            )
            amendments[number] = {
                "effective": datetime.strptime(
                    amendment_match.group(3), "%Y-%m-%d"
                ).date(),
                "revokes": (
                    int(revoke_match.group(1))
                    if revoke_match is not None
                    else None
                ),
                "limit": (
                    limit_match.group(1).replace(",", "")
                    if limit_match is not None
                    else None
                ),
            }
        revokers: dict[int, list[int]] = defaultdict(list)
        for number, amendment in amendments.items():
            revoked = amendment["revokes"]
            if isinstance(revoked, int):
                revokers[revoked].append(number)

        memo: dict[int, bool] = {}

        def active(number: int, visiting: frozenset[int] = frozenset()) -> bool:
            if number in memo:
                return memo[number]
            if number in visiting:
                raise ValueError("cyclic amendment revocations are ambiguous")
            value = not any(
                active(revoker, visiting | {number})
                for revoker in revokers.get(number, ())
            )
            memo[number] = value
            return value

        candidates = [
            (
                amendment["effective"],
                int(number),
                str(amendment["limit"]),
            )
            for number, amendment in amendments.items()
            if amendment["limit"] is not None
            and amendment["effective"] <= query_date
            and active(number)
        ]
        answer = (
            max(candidates)[2]
            if candidates
            else original_limit.group(1).replace(",", "")
        )
        return answer, (
            f"parsed original Clause {clause_number} and every amendment record",
            "resolved amendment revocations recursively, independent of filing order",
            f"selected the latest active effective date on or before {query_date_text}",
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
    def _dirty_ledger(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"amount column is messy.*?status is exactly ['\"]([^'\"]+)['\"]",
            task.prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if not match or "transactions.csv" not in task.files:
            return False
        if workdir is None:
            return True
        target_status = match.group(1)
        total = Decimal("0")
        included = 0
        skipped = 0
        with (workdir / "transactions.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != target_status:
                    continue
                raw = (row.get("amount") or "").strip()
                cleaned = re.sub(
                    r"(?i)\bUSD\b", "", raw
                ).replace("$", "").replace(",", "").strip()
                try:
                    amount = Decimal(cleaned)
                except InvalidOperation:
                    skipped += 1
                    continue
                total += amount
                included += 1
        answer = format(
            total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f"
        )
        return answer, (
            f"filtered rows by exact status {target_status!r}",
            f"parsed and summed {included} valid currency values",
            f"skipped {skipped} blank or non-numeric matching rows",
        )

    @staticmethod
    def _event_horizon(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"events of type ['\"]([^'\"]+)['\"].*?"
            r"UTC date from (\d{4}-\d{2}-\d{2}) through "
            r"(\d{4}-\d{2}-\d{2}) inclusive",
            task.prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if not match or "events.jsonl" not in task.files:
            return False
        if workdir is None:
            return True
        event_type, start_date, end_date = match.groups()
        counts: dict[str, int] = defaultdict(int)
        rows = 0
        with (workdir / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if (
                    event.get("type") == event_type
                    and start_date <= str(event.get("ts", ""))[:10] <= end_date
                ):
                    counts[str(event["user"])] += 1
                    rows += 1
        if not counts:
            raise ValueError("no matching events were found")
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            raise ValueError("event winner is not unique")
        return ordered[0][0], (
            f"parsed every JSONL record and matched {rows} events",
            f"applied inclusive UTC dates {start_date} through {end_date}",
            "verified that the highest event count is unique",
        )

    @staticmethod
    def _join_the_dots(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        lowered_prompt = task.prompt.lower()
        if (
            "orders.jsonl" not in task.files
            or "users.csv" not in task.files
            or "keep only the first occurrence" not in lowered_prompt
            or "exclude orders whose status is 'refunded'"
            not in lowered_prompt
        ):
            return False
        if workdir is None:
            return True
        users: dict[str, str] = {}
        with (workdir / "users.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                users[str(row["id"])] = str(row["email"])
        seen: set[str] = set()
        totals: dict[str, int] = defaultdict(int)
        kept = 0
        with (workdir / "orders.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                order = json.loads(line)
                order_id = str(order["order_id"])
                if order_id in seen:
                    continue
                seen.add(order_id)
                if order.get("status") == "refunded":
                    continue
                totals[str(order["user_id"])] += int(order["amount_cents"])
                kept += 1
        if not totals:
            raise ValueError("no eligible orders were found")
        ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            raise ValueError("highest user total is not unique")
        winner_id = ordered[0][0]
        if winner_id not in users:
            raise ValueError("winning user is missing from users.csv")
        return users[winner_id], (
            f"deduplicated {len(seen)} order ids in file order",
            f"summed {kept} non-refunded first occurrences",
            "verified a unique highest total and joined it to users.csv",
        )

    @staticmethod
    def _session_hunter(
        task: TaskDetail,
        workdir: pathlib.Path | None,
        web: EventWebSession | None,
    ) -> tuple[str, tuple[str, ...]] | bool:
        match = re.search(
            r"during ([A-Za-z]+) (\d{4}).*?"
            r"(\d+)(?:st|nd|rd|th)-highest"
            r"(?:\s+[A-Za-z-]+)*\s+count",
            task.prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if (
            not match
            or "server.log" not in task.files
            or "gap of more than 30 minutes" not in task.prompt
        ):
            return False
        if workdir is None:
            return True
        month_name, year_text, rank_text = match.groups()
        target_month = datetime.strptime(month_name, "%B").month
        target_year = int(year_text)
        target_rank = int(rank_text)
        events: dict[str, list[datetime]] = defaultdict(list)
        timestamp_patterns = (
            (
                re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"),
                "%Y-%m-%dT%H:%M:%SZ",
            ),
            (
                re.compile(
                    r"\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} \+0000"
                ),
                "%d/%b/%Y:%H:%M:%S +0000",
            ),
            (
                re.compile(r"[A-Za-z]{3}\s+\d{1,2} \d{2}:\d{2}:\d{2}"),
                "%b %d %H:%M:%S",
            ),
        )
        with (workdir / "server.log").open(
            encoding="utf-8", errors="replace"
        ) as handle:
            for line in handle:
                user_match = re.search(r"\buser=([^\s]+)", line)
                if user_match is None:
                    continue
                parsed: datetime | None = None
                for pattern, date_format in timestamp_patterns:
                    timestamp_match = pattern.search(line)
                    if timestamp_match is None:
                        continue
                    parsed = datetime.strptime(
                        timestamp_match.group(0), date_format
                    )
                    if date_format == "%b %d %H:%M:%S":
                        parsed = parsed.replace(year=target_year)
                    parsed = parsed.replace(tzinfo=timezone.utc)
                    break
                if parsed is not None:
                    events[user_match.group(1)].append(parsed)
        session_days: list[tuple[str, int]] = []
        for user, timestamps in events.items():
            starts: set[datetime.date] = set()
            previous: datetime | None = None
            for timestamp in sorted(timestamps):
                if previous is None or timestamp - previous > timedelta(minutes=30):
                    if (
                        timestamp.year == target_year
                        and timestamp.month == target_month
                    ):
                        starts.add(timestamp.date())
                previous = timestamp
            session_days.append((user, len(starts)))
        ordered = sorted(
            session_days, key=lambda item: (-item[1], item[0])
        )
        if target_rank < 1 or target_rank > len(ordered):
            raise ValueError("requested session rank is outside the user list")
        return ordered[target_rank - 1][0], (
            f"parsed and normalized {sum(map(len, events.values()))} log events",
            "sorted each user's events and split sessions at gaps over 30 minutes",
            f"ranked distinct session-start days for {month_name} {target_year}",
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

from __future__ import annotations

import argparse
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import pathlib
import random
import string
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.jeopardy_benchmark import SCHEMA_VERSION, category_slug
from evals.synthetic_suite import _write_task


def two_hops(task_id: str, rng: random.Random):
    appendix = rng.choice(string.ascii_uppercase[3:10])
    tier = rng.choice(("Bronze", "Silver", "Gold", "Platinum"))
    multiplier_clause = f"{rng.randint(2, 8)}.{rng.randint(1, 9)}"
    rate = Decimal(rng.randint(10_000, 800_000)) / Decimal("100")
    multiplier = Decimal(rng.randint(125, 550)) / Decimal("100")
    answer = (rate * multiplier).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    contract = (
        f"Clause {multiplier_clause} - Adjustment\n"
        f"The adjustment multiplier applicable to fee computations is "
        f"{multiplier}.\n\n"
        "Clause 14.2 - Monthly Fee\n"
        f"The monthly service fee shall equal the base rate listed in Appendix "
        f"{appendix} for tier {tier} customers, multiplied by the adjustment "
        f"multiplier set forth in Clause {multiplier_clause}.\n\n"
        f"Appendix {appendix} - Current Rates\n"
        f"Tier Bronze: $1,000.00 per month\n"
        f"Tier Silver: $2,000.00 per month\n"
        f"Tier Gold: $3,000.00 per month\n"
        f"Tier Platinum: $4,000.00 per month\n"
    )
    contract = contract.replace(
        f"Tier {tier}: ${ {'Bronze':'1,000.00','Silver':'2,000.00','Gold':'3,000.00','Platinum':'4,000.00'}[tier]}",
        f"Tier {tier}: ${rate:,.2f}",
    )
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Ancient Scrolls",
        "title": "Two Hops",
        "points": 200,
        "prompt": (
            "Download contract.txt. Clause 14.2 defines the monthly service "
            "fee. Compute the fee in dollars; it requires cross-referencing "
            "an appendix and another clause."
        ),
        "files": ["contract.txt"],
        "answer_format": "numeric",
        "claimed": False,
    }
    return payload, {"contract.txt": contract.encode()}, format(answer, "f")


def minute_details(task_id: str, rng: random.Random):
    people = ("Ingrid", "Elena", "Marcus", "Sylvia", "Ronan")
    person = rng.choice(people)
    month = rng.randint(1, 12)
    ticket = "".join(rng.choice(string.ascii_uppercase) for _ in range(2))
    ticket += f"-{rng.randint(1000, 9999)}"
    files = {}
    for current_month in range(1, 13):
        lines = [f"## Weekly - 2026-{current_month:02d}-07", "### Action Items"]
        lines.append("- Other: routine task (ticket AA-1000)")
        if current_month == month:
            lines.append(f"- {person}: complete target work (ticket {ticket})")
        elif rng.random() < 0.5:
            lines.append(f"- {person}: distractor work (ticket ZZ-9999)")
        name = f"notes_2026-{current_month:02d}.md"
        files[name] = ("\n".join(lines) + "\n").encode()
    month_name = date(2026, month, 1).strftime("%B")
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Ancient Scrolls",
        "title": "Minute Details",
        "points": 300,
        "prompt": (
            "Download the twelve monthly notes files. Exactly one action item "
            f"was assigned to {person} during {month_name} 2026; it references "
            "a ticket code of the form XX-1234."
        ),
        "files": sorted(files),
        "answer_format": "exact",
        "claimed": False,
    }
    return payload, files, ticket


def living_document(task_id: str, rng: random.Random):
    clause = f"{rng.randint(2, 9)}.{rng.randint(1, 9)}"
    original = rng.randrange(10_000, 40_000, 250)
    first = rng.randrange(40_000, 60_000, 250)
    revoked = rng.randrange(60_000, 80_000, 250)
    latest = rng.randrange(80_000, 100_000, 250)
    query = date(2025, 1, 15) + timedelta(days=rng.randint(-20, 20))
    entries = [
        (
            7,
            date(2024, 1, 10),
            f'Clause {clause} is amended to read: "The limit is ${first:,}."',
        ),
        (
            12,
            date(2024, 6, 5),
            f'Clause {clause} is amended to read: "The limit is ${revoked:,}."',
        ),
        (18, date(2024, 8, 1), "Amendment 12 is hereby revoked."),
        (
            23,
            date(2024, 11, 20),
            f'Clause {clause} is amended to read: "The limit is ${latest:,}."',
        ),
    ]
    rng.shuffle(entries)
    amendments = "\n\n".join(
        f"Amendment {number} (published {(effective - timedelta(days=30)):%Y-%m-%d}; "
        f"effective {effective:%Y-%m-%d}): {action}"
        for number, effective, action in entries
    )
    text = (
        f"Clause {clause} - Original\n"
        f"Transactions under this clause are subject to a limit of "
        f"${original:,}.\n\nAMENDMENTS\n{amendments}\n"
    )
    active = [
        (date(2024, 1, 10), first),
        (date(2024, 11, 20), latest),
    ]
    eligible = [item for item in active if item[0] <= query]
    answer = max(eligible)[1] if eligible else original
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Ancient Scrolls",
        "title": "Living Document",
        "points": 400,
        "prompt": (
            "Download regulation.txt. The AMENDMENTS section is not "
            "chronological; use effective dates, and disregard every revoked "
            f"amendment. What limit was in force for Clause {clause} on "
            f"{query:%Y-%m-%d}?"
        ),
        "files": ["regulation.txt"],
        "answer_format": "numeric",
        "claimed": False,
    }
    return payload, {"regulation.txt": text.encode()}, str(answer)


GENERATORS = (
    ("AT", "Two Hops", 200, two_hops),
    ("AM", "Minute Details", 300, minute_details),
    ("AL", "Living Document", 400, living_document),
)


def generate(arguments: argparse.Namespace) -> int:
    output = pathlib.Path(arguments.output).resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.mkdir(parents=True)
    records = []
    oracle = {}
    for code, title, points, generator in GENERATORS:
        for split, count in (
            ("train", arguments.train_per_template),
            ("test", arguments.test_per_template),
        ):
            for index in range(1, count + 1):
                task_id = f"SYN-{code}-{split.upper()}-{index:03d}"
                payload, files, answer = generator(
                    task_id,
                    random.Random(f"{arguments.seed}\0{task_id}"),
                )
                task_path, file_records = _write_task(output, payload, files)
                records.append(
                    {
                        "id": task_id,
                        "board": "synthetic",
                        "category": payload["category"],
                        "points": points,
                        "title": title,
                        "cell_key": f"synthetic:{code}",
                        "template_key": hashlib.sha256(
                            f"{payload['category']}\0{points}\0{title}".encode()
                        ).hexdigest(),
                        "variant_index": index - 1,
                        "split": split,
                        "answer_format": payload["answer_format"],
                        "claimed": False,
                        "prompt_sha256": hashlib.sha256(
                            payload["prompt"].encode()
                        ).hexdigest(),
                        "task_path": task_path,
                        "files": file_records,
                        "download_bytes": sum(
                            record["bytes"] for record in file_records
                        ),
                        "pull_seconds": 0.0,
                    }
                )
                oracle[task_id] = {
                    "answer": answer,
                    "answer_format": payload["answer_format"],
                }
    records.sort(key=lambda item: item["id"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "base_url": "http://127.0.0.1",
            "board": "synthetic",
            "phase": "offline",
            "server_time": None,
        },
        "split": {
            "policy": "seeded-generator",
            "seed": arguments.seed,
        },
        "download_files": True,
        "tasks": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "oracle.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True) + "\n"
    )
    category_dir = output / "categories" / category_slug("Ancient Scrolls")
    category_dir.mkdir(parents=True)
    (category_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_root": "../..",
                "category": "Ancient Scrolls",
                "tasks": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "tasks": len(records),
                "train": sum(item["split"] == "train" for item in records),
                "test": sum(item["split"] == "test" for item in records),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="walla-walla-ancient-v1")
    parser.add_argument("--train-per-template", type=int, default=3)
    parser.add_argument("--test-per-template", type=int, default=2)
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

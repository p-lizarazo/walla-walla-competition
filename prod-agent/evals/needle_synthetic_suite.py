from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import io
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.jeopardy_benchmark import SCHEMA_VERSION, category_slug
from evals.synthetic_suite import _write_task


def dirty_ledger(task_id: str, rng: random.Random):
    target = rng.choice(("returned", "cancelled"))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["transaction_id", "status", "amount"])
    total = Decimal("0")
    for index in range(80):
        status = rng.choice(("returned", "cancelled", "paid"))
        if index % 17 == 0:
            value = rng.choice(("", "missing", "N/A"))
        else:
            amount = Decimal(rng.randint(1, 250_000)) / Decimal("100")
            value = rng.choice(
                (
                    f"${amount:,.2f}",
                    f"{amount:.2f}",
                    f"  {amount:,.2f} USD",
                )
            )
            if status == target:
                total += amount
        writer.writerow([f"T{index:04d}", status, value])
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Needle in the Haystack",
        "title": "Dirty Ledger",
        "points": 100,
        "prompt": (
            "Download transactions.csv. The amount column is messy: values "
            "may contain dollar signs, commas, whitespace, or USD; skip blank "
            f"and non-numeric values. Sum rows whose status is exactly '{target}'."
        ),
        "files": ["transactions.csv"],
        "answer_format": "numeric",
        "claimed": False,
    }
    return payload, {"transactions.csv": output.getvalue().encode()}, f"{total:.2f}"


def event_horizon(task_id: str, rng: random.Random):
    event_type = rng.choice(("upload", "purchase", "login"))
    start = datetime(2026, rng.randint(2, 9), 5)
    end = start + timedelta(days=rng.randint(10, 18))
    users = [f"user{index:03d}" for index in range(5)]
    winner = rng.choice(users)
    rows = []
    for user in users:
        count = 14 if user == winner else rng.randint(3, 10)
        for index in range(count):
            timestamp = start + timedelta(
                seconds=rng.randint(0, int((end - start).total_seconds()) + 86399)
            )
            rows.append(
                {
                    "user": user,
                    "type": event_type,
                    "ts": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
        rows.append(
            {
                "user": user,
                "type": "noise",
                "ts": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    rng.shuffle(rows)
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Needle in the Haystack",
        "title": "Event Horizon",
        "points": 200,
        "prompt": (
            "Download events.jsonl. Which user has the most events of type "
            f"'{event_type}' with a UTC date from {start:%Y-%m-%d} through "
            f"{end:%Y-%m-%d} inclusive? There is a unique winner."
        ),
        "files": ["events.jsonl"],
        "answer_format": "exact_ci",
        "claimed": False,
    }
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    return payload, {"events.jsonl": content}, winner


def join_the_dots(task_id: str, rng: random.Random):
    users = {
        str(index): f"user{index}@example.test"
        for index in range(1, 7)
    }
    winner = rng.choice(list(users))
    totals = {user_id: rng.randint(1_000, 8_000) for user_id in users}
    totals[winner] = 20_000
    orders = []
    for index, (user_id, total) in enumerate(totals.items()):
        order_id = f"O{index:04d}"
        orders.append(
            {
                "order_id": order_id,
                "user_id": int(user_id),
                "amount_cents": total,
                "status": "paid",
            }
        )
        orders.append(
            {
                "order_id": order_id,
                "user_id": int(rng.choice(list(users))),
                "amount_cents": 99_999,
                "status": "paid",
            }
        )
    orders.append(
        {
            "order_id": "REFUND",
            "user_id": int(rng.choice(list(users))),
            "amount_cents": 999_999,
            "status": "refunded",
        }
    )
    users_csv = io.StringIO()
    writer = csv.writer(users_csv)
    writer.writerow(["id", "email", "signup"])
    for user_id, email in users.items():
        writer.writerow([user_id, email, "2026-01-01"])
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Needle in the Haystack",
        "title": "Join the Dots",
        "points": 300,
        "prompt": (
            "Download users.csv and orders.jsonl. Keep only the FIRST "
            "occurrence of each order_id, exclude orders whose status is "
            "'refunded', sum amount_cents per user, and answer with the "
            "email of the unique highest total."
        ),
        "files": ["orders.jsonl", "users.csv"],
        "answer_format": "exact_ci",
        "claimed": False,
    }
    order_content = "".join(json.dumps(row) + "\n" for row in orders).encode()
    return (
        payload,
        {
            "orders.jsonl": order_content,
            "users.csv": users_csv.getvalue().encode(),
        },
        users[winner],
    )


def session_hunter(task_id: str, rng: random.Random):
    users = [f"u{index:04d}" for index in range(1, 7)]
    rng.shuffle(users)
    counts = [6, 5, 4, 3, 2, 1]
    target_rank = rng.choice((2, 3, 4))
    answer = users[target_rank - 1]
    lines = []
    formats = (
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%b/%Y:%H:%M:%S +0000",
        "%b %d %H:%M:%S",
    )
    for user, count in zip(users, counts):
        for day in range(1, count + 1):
            timestamp = datetime(2026, 9, day, 8, 0, 0)
            date_format = formats[day % len(formats)]
            lines.append(f"{timestamp.strftime(date_format)} user={user}")
            lines.append(
                f"{(timestamp + timedelta(minutes=10)).strftime(date_format)} "
                f"user={user}"
            )
        lines.append(f"2026-08-15T12:00:00Z user={user}")
    rng.shuffle(lines)
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Needle in the Haystack",
        "title": "Session Hunter",
        "points": 400,
        "prompt": (
            "Download server.log. Timestamps use ISO, Apache, or syslog "
            "formats; syslog lines omit the year, which is 2026. A gap of "
            "more than 30 minutes starts a new session. During September "
            f"2026, which user has the {target_rank}th-highest session-day count?"
        ),
        "files": ["server.log"],
        "answer_format": "exact_ci",
        "claimed": False,
    }
    return payload, {"server.log": ("\n".join(lines) + "\n").encode()}, answer


GENERATORS = (
    ("ND", "Dirty Ledger", 100, dirty_ledger),
    ("NE", "Event Horizon", 200, event_horizon),
    ("NJ", "Join the Dots", 300, join_the_dots),
    ("NS", "Session Hunter", 400, session_hunter),
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
                rng = random.Random(f"{arguments.seed}\0{task_id}")
                payload, files, answer = generator(task_id, rng)
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
    category_dir = output / "categories" / category_slug(
        "Needle in the Haystack"
    )
    category_dir.mkdir(parents=True)
    (category_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_root": "../..",
                "category": "Needle in the Haystack",
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
    parser.add_argument("--seed", default="walla-walla-needle-v1")
    parser.add_argument("--train-per-template", type=int, default=3)
    parser.add_argument("--test-per-template", type=int, default=2)
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

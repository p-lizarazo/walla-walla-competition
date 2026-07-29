from __future__ import annotations

import argparse
import ast
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import io
import json
import pathlib
import random
import string
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.jeopardy_benchmark import SCHEMA_VERSION, category_slug


CATEGORIES = (
    ("Ancient Scrolls", "A", 100, "Buried Clause"),
    ("Cryptic", "C", 200, "Locked Box"),
    ("Heavy Compute", "H", 100, "Congruential Cascade"),
    ("Needle in the Haystack", "N", 500, "Thirty Systems"),
    ("Ship It", "S", 200, "Spec vs Code"),
    ("The Dark Web", "W", 200, "Members Only"),
)


def _token(rng: random.Random, prefix: str, length: int = 12) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(length))


def _write_task(
    root: pathlib.Path,
    payload: dict,
    files: dict[str, bytes],
) -> tuple[str, list[dict]]:
    task_dir = root / "tasks" / payload["id"]
    task_dir.mkdir(parents=True)
    records = []
    for name, content in files.items():
        path = task_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append(
            {
                "name": name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    task_path = task_dir / "task.json"
    task_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(task_path.relative_to(root)), records


def _ancient(
    task_id: str, rng: random.Random
) -> tuple[dict, dict[str, bytes], str]:
    target_major = rng.randint(6, 18)
    target_minor = rng.randint(1, 9)
    target = f"{target_major}.{target_minor}"
    answer = str(rng.randrange(1250, 9750, 25))
    sections = []
    for major in range(1, 21):
        for minor in range(1, 10):
            value = answer if f"{major}.{minor}" == target else str(
                rng.randrange(1000, 9900, 25)
            )
            sections.append(
                f"Section {major}.{minor} - Expense rule\n"
                f"The maximum reimbursement for a single claim under this "
                f"section is ${int(value):,}.\n"
            )
    prompt = (
        "Download policy_manual.txt, a long corporate expense policy. "
        "What is the maximum reimbursement for a single claim, in dollars, "
        f"under Section {target}? Many sections state similar limits; make "
        "sure you read the right one. Answer with just the number."
    )
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Ancient Scrolls",
        "title": "Buried Clause",
        "points": 100,
        "prompt": prompt,
        "files": ["policy_manual.txt"],
        "answer_format": "numeric",
        "claimed": False,
    }
    return payload, {"policy_manual.txt": "\n".join(sections).encode()}, answer


def _cryptic(
    task_id: str, rng: random.Random
) -> tuple[dict, dict[str, bytes], str]:
    answer = _token(rng, "VAULT")
    stem = "".join(rng.choice(string.ascii_lowercase) for _ in range(7))
    password = stem[::-1].encode()
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        inner.writestr("token.txt", answer)
    encrypted = bytes(
        byte ^ password[index % len(password)]
        for index, byte in enumerate(inner_buffer.getvalue())
    )
    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as archive:
        archive.writestr(
            "note.txt",
            "Use the largest .dat file, reversed, as the password. "
            "inner.blob is XOR-encrypted.",
        )
        archive.writestr(f"{stem}.dat", b"x" * 128)
        archive.writestr("small.dat", b"x" * 4)
        archive.writestr("inner.blob", encrypted)
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Cryptic",
        "title": "Locked Box",
        "points": 200,
        "prompt": (
            "Download bundle.zip and start with note.txt inside it. Follow its "
            "instructions to reach the token. Answer with the token exactly "
            "(format VAULT-XXXXXXXXXXXX)."
        ),
        "files": ["bundle.zip"],
        "answer_format": "exact",
        "claimed": False,
    }
    return payload, {"bundle.zip": outer_buffer.getvalue()}, answer


def _heavy(
    task_id: str, rng: random.Random
) -> tuple[dict, dict[str, bytes], str]:
    value = rng.randint(1, 2_000_000_000)
    multiplier = rng.randint(100_000, 999_999)
    increment = rng.randint(100_000, 999_999)
    modulus = 2_147_483_647
    count = rng.randint(20_000, 40_000)
    divisor = rng.choice((5, 7, 11))
    initial = value
    total = 0
    for _ in range(count):
        value = (multiplier * value + increment) % modulus
        if value % divisor == 0:
            total += value
    prompt = (
        f"A sequence is defined by x0 = {initial}; "
        f"x_(n+1) = ({multiplier} * x_n + {increment}) mod {modulus}. "
        f"Apply the recurrence {count:,} times starting from x0. Compute the "
        f"sum of all generated terms divisible by {divisor}. Answer with the "
        "sum as a plain integer."
    )
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Heavy Compute",
        "title": "Congruential Cascade",
        "points": 100,
        "prompt": prompt,
        "files": [],
        "answer_format": "numeric",
        "claimed": False,
    }
    return payload, {}, str(total)


def _needle(
    task_id: str, rng: random.Random
) -> tuple[dict, dict[str, bytes], str]:
    rates = {
        "AA": Decimal("1.25"),
        "BB": Decimal("0.80"),
        "CC": Decimal("2.10"),
        "DD": Decimal("0.55"),
    }
    archive_buffer = io.BytesIO()
    seen: set[str] = set()
    total = Decimal("0")
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            "fx.csv",
            "region,currency,usd_per_unit\n"
            + "".join(
                f"{region},CUR{region},{rate}\n"
                for region, rate in rates.items()
            ),
        )
        transaction_pool = [f"T{index:04d}" for index in range(20)]
        for file_index, region in enumerate(sorted(rates), 1):
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["order_id", "amount"])
            for _ in range(8):
                transaction_id = rng.choice(transaction_pool)
                amount = Decimal(rng.randint(100, 25_000)) / Decimal("10")
                writer.writerow([transaction_id, format(amount, "f")])
                if transaction_id not in seen:
                    seen.add(transaction_id)
                    total += amount * rates[region]
            archive.writestr(
                f"sales_{region}_{file_index:03d}.csv",
                output.getvalue(),
            )
    answer = format(
        total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f"
    )
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Needle in the Haystack",
        "title": "Thirty Systems",
        "points": 500,
        "prompt": (
            "Download regional_sales.zip. It contains fx.csv with "
            "usd_per_unit and regional sales exports. Keep only the FIRST "
            "occurrence of each transaction id in filename order, convert "
            "with the region rate, and round only the final total."
        ),
        "files": ["regional_sales.zip"],
        "answer_format": "numeric",
        "claimed": False,
    }
    return payload, {"regional_sales.zip": archive_buffer.getvalue()}, answer


def _ship(
    task_id: str, rng: random.Random
) -> tuple[dict, dict[str, bytes], str]:
    names = rng.sample(
        [
            "amber",
            "cedar",
            "ember",
            "gable",
            "heron",
            "indigo",
            "umbra",
            "zephyr",
        ],
        7,
    )
    pairs = [(name, rng.choice((12, 14, 18, 24, 30))) for name in names]
    expected = [name for name, _ in sorted(pairs, key=lambda item: (-item[1], item[0]))]
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "Ship It",
        "title": "Spec vs Code",
        "points": 200,
        "prompt": (
            "Download lib.py and spec.md. lib.py contains exactly one bug. "
            f"Per the spec, what should leaderboard({pairs!r}) return? "
            "Answer with the exact Python repr."
        ),
        "files": ["lib.py", "spec.md"],
        "answer_format": "literal",
        "claimed": False,
    }
    files = {
        "lib.py": (
            b"def leaderboard(rows):\n"
            b"    return [name for name, score in sorted(rows)]\n"
        ),
        "spec.md": (
            b"Order entries by score descending. For equal scores, order "
            b"names alphabetically ascending.\n"
        ),
    }
    return payload, files, repr(expected)


def _web(
    task_id: str, rng: random.Random
) -> tuple[dict, dict[str, bytes], str]:
    answer = _token(rng, "VAULT")
    username = "agent_" + "".join(
        rng.choice(string.ascii_lowercase) for _ in range(7)
    )
    password = "".join(
        rng.choice(string.ascii_letters + string.digits) for _ in range(14)
    )
    csrf = hashlib.sha256(task_id.encode()).hexdigest()[:12]
    start = f"http://127.0.0.1/web/{task_id}/"
    login = start + "login"
    member = start + "member/"
    vault = start + "vault"
    responses = {
        "responses": {
            f"GET {start}": {
                "status": 200,
                "url": start,
                "headers": {"Content-Type": "text/html"},
                "body": (
                    '<form method="post" action="login">'
                    f'<input name="csrf" value="{csrf}">'
                    '<input name="username"><input name="password"></form>'
                ),
            },
            f"POST {login}": {
                "status": 200,
                "url": member,
                "headers": {"Content-Type": "text/html"},
                "expected_form": {
                    "csrf": csrf,
                    "username": username,
                    "password": password,
                },
                "body": '<a href="../vault">Open vault</a>',
            },
            f"GET {vault}": {
                "status": 200,
                "url": vault,
                "headers": {"Content-Type": "text/html"},
                "body": f"<p>{answer}</p>",
            },
        }
    }
    payload = {
        "id": task_id,
        "board": "synthetic",
        "category": "The Dark Web",
        "title": "Members Only",
        "points": 200,
        "prompt": (
            f"A token sits in the members-only vault at {start} "
            f"Your credentials — username: {username} password: {password} "
            "Log in via the form, keep your session cookie, and open the vault."
        ),
        "files": [],
        "answer_format": "exact",
        "claimed": False,
    }
    return (
        payload,
        {"responses.json": json.dumps(responses, indent=2).encode()},
        answer,
    )


GENERATORS = {
    "A": _ancient,
    "C": _cryptic,
    "H": _heavy,
    "N": _needle,
    "S": _ship,
    "W": _web,
}


def generate(arguments: argparse.Namespace) -> int:
    output = pathlib.Path(arguments.output).resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.mkdir(parents=True)
    records = []
    oracle = {}
    for category, letter, points, title in CATEGORIES:
        for split, count in (
            ("train", arguments.train_per_category),
            ("test", arguments.test_per_category),
        ):
            for index in range(1, count + 1):
                task_id = f"SYN-{letter}-{split.upper()}-{index:03d}"
                seed = f"{arguments.seed}\0{task_id}"
                rng = random.Random(seed)
                payload, files, answer = GENERATORS[letter](task_id, rng)
                task_path, file_records = _write_task(output, payload, files)
                template_key = hashlib.sha256(
                    f"{category}\0{points}\0{title}".encode()
                ).hexdigest()
                records.append(
                    {
                        "id": task_id,
                        "board": "synthetic",
                        "category": category,
                        "points": points,
                        "title": title,
                        "cell_key": f"synthetic:{letter}",
                        "template_key": template_key,
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "oracle.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    categories = {}
    for record in records:
        categories.setdefault(record["category"], []).append(record)
    for category, category_records in categories.items():
        directory = output / "categories" / category_slug(category)
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_root": "../..",
                    "category": category,
                    "tasks": category_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for split in ("train", "test"):
            (directory / f"{split}_ids.txt").write_text(
                "".join(
                    f"{record['id']}\n"
                    for record in category_records
                    if record["split"] == split
                ),
                encoding="utf-8",
            )
    print(
        json.dumps(
            {
                "output": str(output),
                "tasks": len(records),
                "train": sum(record["split"] == "train" for record in records),
                "test": sum(record["split"] == "test" for record in records),
                "categories": len(categories),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate seeded synthetic Agent Jeopardy tasks."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="walla-walla-synthetic-v1")
    parser.add_argument("--train-per-category", type=int, default=4)
    parser.add_argument("--test-per-category", type=int, default=2)
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

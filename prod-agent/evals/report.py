from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import statistics


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize practice telemetry.")
    parser.add_argument("path", nargs="?", default="telemetry.jsonl")
    args = parser.parse_args()

    events = []
    with open(args.path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    submissions = [
        event for event in events if event.get("event") == "submission"
    ]
    outcomes = Counter(event.get("result", "unknown") for event in submissions)
    latencies = [
        float(event["elapsed_seconds"])
        for event in submissions
        if event.get("elapsed_seconds") is not None
    ]
    grouped: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for event in submissions:
        key = (str(event.get("category", "")), int(event.get("points") or 0))
        grouped[key][str(event.get("result", "unknown"))] += 1

    print(f"events={len(events)} submissions={len(submissions)}")
    print("outcomes=" + json.dumps(outcomes, sort_keys=True))
    if latencies:
        print(
            "latency_seconds="
            + json.dumps(
                {
                    "mean": round(statistics.mean(latencies), 2),
                    "p50": round(percentile(latencies, 0.50), 2),
                    "p95": round(percentile(latencies, 0.95), 2),
                },
                sort_keys=True,
            )
        )
    for key in sorted(grouped):
        print(f"{key[0]} {key[1]}: {dict(grouped[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one bounded production-agent practice experiment."
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--temperatures", default="0.0,0.25,0.5")
    parser.add_argument("--duration", type=int, default=600)
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.update(
        {
            "AGENT_MODE": "practice_eval",
            "WORKERS": str(args.workers),
            "MAX_TILES": str(args.max_tiles),
            "TEMPERATURES": args.temperatures,
            "RUN_DURATION_SECONDS": str(args.duration),
        }
    )
    return subprocess.call(
        [sys.executable, "-u", "main.py"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())

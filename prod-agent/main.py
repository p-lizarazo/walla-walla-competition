from __future__ import annotations

from config import Config
from orchestrator import Orchestrator


def main() -> None:
    Orchestrator(Config.from_env()).run()


if __name__ == "__main__":
    main()

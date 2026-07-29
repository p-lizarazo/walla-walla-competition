from __future__ import annotations

import unittest

from models import Priority, Tile
from orchestrator import Orchestrator


def pair(task_id: str, category: str, points: int, score: float):
    tile = Tile(task_id, category, points)
    priority = Priority(task_id, score, 0.9, 10, 1, points, ())
    return tile, priority


class OrchestratorTests(unittest.TestCase):
    def test_parallel_wave_reserves_capacity_for_500s(self) -> None:
        ranked = [
            pair("fast200", "Cryptic", 200, 10),
            pair("five-a", "Heavy Compute", 500, 4),
            pair("five-b", "The Dark Web", 500, 3),
            pair("other", "Ship It", 300, 2),
        ]

        chosen = Orchestrator._diverse_take(ranked, 2)

        self.assertEqual(chosen[0][0].points, 500)
        self.assertEqual(len(chosen), 2)


if __name__ == "__main__":
    unittest.main()

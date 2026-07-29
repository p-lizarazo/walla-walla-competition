from __future__ import annotations

import unittest
import threading

from models import BoardSnapshot, Phase, Priority, Tile
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

    def test_parallel_wave_fills_missing_category_lane(self) -> None:
        ranked = [
            pair("cryptic", "Cryptic", 300, 10),
            pair("web", "The Dark Web", 300, 9),
            pair("ship", "Ship It", 300, 8),
        ]

        chosen = Orchestrator._diverse_take(
            ranked, 1, occupied_categories={"Cryptic", "The Dark Web"}
        )

        self.assertEqual(chosen[0][0].category, "Ship It")

    def test_live_snapshot_drives_cooperative_cancellation(self) -> None:
        orchestrator = object.__new__(Orchestrator)
        orchestrator.config = type("Config", (), {"mode": "auto"})()
        orchestrator._snapshot_lock = threading.Lock()
        orchestrator._snapshot = BoardSnapshot(
            phase=Phase.GAME,
            tiles=(Tile("open", "Cryptic", 100),),
            solved_ids=frozenset(),
            server_time=None,
            fetched_monotonic=0,
        )

        self.assertTrue(orchestrator._tile_is_open("open"))
        self.assertFalse(orchestrator._tile_is_open("claimed"))


if __name__ == "__main__":
    unittest.main()

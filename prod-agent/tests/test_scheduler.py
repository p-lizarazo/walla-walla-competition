import unittest

from models import AttemptState, BoardSnapshot, Phase, Tile
from scheduler import (
    PerformanceEstimate,
    Scheduler,
    calibrated_default,
    expected_net_points,
)


class SchedulerTests(unittest.TestCase):
    @staticmethod
    def board(phase):
        return BoardSnapshot(
            phase=phase,
            tiles=(
                Tile("five", "Cryptic", 500),
                Tile("four", "Ship It", 400),
            ),
            solved_ids=frozenset(),
            server_time=None,
            fetched_monotonic=1,
        )

    def test_expected_value_includes_wrong_answer_penalty(self):
        self.assertEqual(expected_net_points(400, 1.0), 400)
        self.assertEqual(expected_net_points(400, 0.0), -100)
        self.assertAlmostEqual(expected_net_points(400, 0.8), 300)

    def test_defaults_are_calibrated_by_category_and_tier(self):
        easy = calibrated_default("Cryptic", 100)
        hard = calibrated_default("Cryptic", 500)
        other = calibrated_default("Heavy Compute", 500)
        self.assertGreater(easy.probability, hard.probability)
        self.assertLess(easy.solve_seconds, hard.solve_seconds)
        self.assertNotEqual(hard, other)

    def test_practice_calibration_can_update_ranking_dynamically(self):
        scheduler = Scheduler()
        tile = Tile("tile", "Cryptic", 500)
        before = scheduler.priority(tile, race_survival=1)
        scheduler.update_calibration("Cryptic", 500, 0.95, 20)
        after = scheduler.priority(tile, race_survival=1)
        self.assertGreater(after.score, before.score)

    def test_ranking_is_expected_points_per_second_not_points_only(self):
        tiles = [
            Tile("slow500", "Heavy Compute", 500),
            Tile("fast400", "Cryptic", 400),
        ]
        scheduler = Scheduler(
            {
                ("Heavy Compute", 500): PerformanceEstimate(0.45, 180),
                ("Cryptic", 400): PerformanceEstimate(0.95, 20),
            }
        )
        ranked = scheduler.rank(
            tiles, race_survival={"slow500": 1, "fast400": 1}
        )
        self.assertEqual(ranked[0].task_id, "fast400")

    def test_initial_500_boost_breaks_otherwise_close_tie(self):
        tiles = [
            Tile("five", "Cryptic", 500),
            Tile("four", "Cryptic", 400),
        ]
        scheduler = Scheduler(initial_500_boost=1.15)
        ranked = scheduler.rank(
            tiles,
            probabilities={"five": 0.9, "four": 0.9},
            solve_seconds={"five": 50, "four": 40},
            race_survival={"five": 1, "four": 1},
        )
        self.assertEqual(ranked[0].task_id, "five")
        ranked = scheduler.rank(
            tiles,
            probabilities={"five": 0.9, "four": 0.9},
            solve_seconds={"five": 50, "four": 40},
            race_survival={"five": 1, "four": 1},
            initial_wave=False,
        )
        self.assertEqual(ranked[0].task_id, "four")

    def test_cooldown_tiles_are_omitted_and_misses_reduce_priority(self):
        tiles = [
            Tile("cooling", "Ship It", 300),
            Tile("missed", "Ship It", 300),
            Tile("fresh", "Ship It", 300),
        ]
        attempts = {
            "cooling": AttemptState(cooldown_until=101),
            "missed": AttemptState(attempts=1, incorrect_attempts=1),
        }
        ranked = Scheduler(clock=lambda: 100).rank(
            tiles,
            attempts,
            probabilities={tile.id: 0.8 for tile in tiles},
            solve_seconds={tile.id: 30 for tile in tiles},
            race_survival={tile.id: 1 for tile in tiles},
        )
        self.assertEqual(
            [priority.task_id for priority in ranked], ["fresh", "missed"]
        )

    def test_scored_rounds_use_identical_policy(self):
        scheduler = Scheduler()
        overrides = {
            "probabilities": {"five": 0.8, "four": 0.9},
            "solve_seconds": {"five": 50, "four": 30},
            "race_survival": {"five": 0.8, "four": 0.8},
        }
        qualifier = scheduler.rank_board(
            self.board(Phase.ROUND1), **overrides
        )
        finale = scheduler.rank_board(self.board(Phase.GAME), **overrides)
        self.assertEqual(qualifier, finale)

    def test_practice_is_explicit_evaluation_mode(self):
        with self.assertRaises(ValueError):
            Scheduler().rank_board(self.board(Phase.PRACTICE))
        practice = Scheduler(mode="practice_eval").rank_board(
            self.board(Phase.PRACTICE)
        )
        self.assertTrue(practice)
        with self.assertRaises(ValueError):
            Scheduler(mode="practice_eval").rank_board(
                self.board(Phase.GAME)
            )


if __name__ == "__main__":
    unittest.main()

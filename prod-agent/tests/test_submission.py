import hashlib
import threading
import time
from types import SimpleNamespace
import unittest

from models import BoardSnapshot, Candidate, Evidence, Phase, Tile
from submission import (
    CooldownActive,
    DuplicateAnswer,
    SubmissionLane,
    SubmissionRejected,
    TileNotOpen,
)


CONFIG = SimpleNamespace(
    submission_interval_seconds=3.1,
    board_poll_seconds=1.0,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def submit(self, task_id, answer):
        self.calls.append((task_id, answer))
        return self.results.pop(0)


def candidate(task_id, answer, method="computed"):
    return Candidate(
        task_id=task_id,
        answer=answer,
        answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
        evidence=Evidence(
            method=method,
            deterministic_checks=("validated",),
            independent_checks=("recomputed",),
        ),
        model_temperature=0,
        elapsed_seconds=1,
        tool_turns=1,
    )


class SubmissionLaneTests(unittest.TestCase):
    def test_interval_and_duplicate_hash_are_centralized(self):
        clock = FakeClock()
        client = FakeClient(
            [{"result": "correct"}, {"result": "correct"}]
        )
        lane = SubmissionLane(
            client, CONFIG, lambda task_id: True,
            clock=clock, sleep=clock.sleep,
        )
        first = candidate("a", "one")
        lane.submit(first)
        with self.assertRaises(DuplicateAnswer):
            lane.submit(first)
        lane.submit(candidate("b", "two"))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(clock.sleeps), 1)
        self.assertAlmostEqual(clock.sleeps[0], 3.1)

    def test_wrong_cooldown_doubles(self):
        clock = FakeClock()
        client = FakeClient(
            [{"result": "incorrect"}, {"result": "incorrect"}]
        )
        lane = SubmissionLane(
            client, CONFIG, lambda task_id: True,
            clock=clock, sleep=clock.sleep,
        )
        lane.submit(candidate("a", "one"))
        self.assertEqual(lane.state("a").cooldown_until, 130)
        with self.assertRaises(CooldownActive):
            lane.submit(candidate("a", "two"))
        clock.now = 130
        lane.submit(candidate("a", "two"))
        self.assertEqual(lane.state("a").cooldown_until, 190)
        self.assertEqual(lane.state("a").incorrect_attempts, 2)
        self.assertEqual(lane.state("a").last_method, "computed")
        self.assertEqual(len(lane.state("a").submitted_hashes), 2)

    def test_wrong_cooldown_is_capped_at_480_seconds(self):
        clock = FakeClock()
        lane = SubmissionLane(
            FakeClient([{"result": "incorrect"}] * 6),
            CONFIG,
            lambda task_id: True,
            clock=clock,
            sleep=clock.sleep,
        )
        for index in range(6):
            lane.submit(candidate("a", str(index)))
            cooldown = lane.state("a").cooldown_until - clock.now
            clock.now += cooldown
        self.assertEqual(cooldown, 480)

    def test_rate_limit_retries_and_records_one_attempt(self):
        clock = FakeClock()
        client = FakeClient(
            [
                {"result": "rate_limited", "retry_in": 5},
                {"result": "correct"},
            ]
        )
        lane = SubmissionLane(
            client, CONFIG, lambda task_id: True,
            clock=clock, sleep=clock.sleep,
        )
        result = lane.submit(candidate("a", "one"))
        self.assertEqual(result["result"], "correct")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(clock.sleeps, [5])
        self.assertEqual(lane.state("a").attempts, 1)

    def test_board_is_checked_immediately_before_post(self):
        client = FakeClient([{"result": "correct"}])
        lane = SubmissionLane(client, CONFIG, lambda task_id: False)
        with self.assertRaises(TileNotOpen):
            lane.submit(candidate("gone", "one"))
        self.assertEqual(client.calls, [])

    def test_stale_board_snapshot_is_rejected(self):
        clock = FakeClock()
        snapshot = BoardSnapshot(
            phase=Phase.GAME,
            tiles=(Tile("a", "Cryptic", 500),),
            solved_ids=frozenset(),
            server_time=None,
            fetched_monotonic=90,
        )
        client = FakeClient([{"result": "correct"}])
        lane = SubmissionLane(
            client, CONFIG, lambda task_id: snapshot,
            clock=clock, sleep=clock.sleep, board_max_age_seconds=2,
        )
        with self.assertRaises(TileNotOpen):
            lane.submit(candidate("a", "one"))
        self.assertEqual(client.calls, [])

    def test_practice_snapshot_requires_explicit_evaluation_mode(self):
        clock = FakeClock()
        snapshot = BoardSnapshot(
            phase=Phase.PRACTICE,
            tiles=(Tile("a", "Cryptic", 500),),
            solved_ids=frozenset(),
            server_time=None,
            fetched_monotonic=clock.now,
        )
        scored_config = SimpleNamespace(
            submission_interval_seconds=3.1,
            board_poll_seconds=1,
            mode="scored",
        )
        scored = SubmissionLane(
            FakeClient([{"result": "correct"}]),
            scored_config,
            lambda task_id: snapshot,
            clock=clock,
            sleep=clock.sleep,
        )
        with self.assertRaises(TileNotOpen):
            scored.submit(candidate("a", "one"))

        practice_config = SimpleNamespace(
            submission_interval_seconds=3.1,
            board_poll_seconds=1,
            mode="practice_eval",
        )
        client = FakeClient([{"result": "correct"}])
        practice = SubmissionLane(
            client,
            practice_config,
            lambda task_id: snapshot,
            clock=clock,
            sleep=clock.sleep,
        )
        self.assertEqual(
            practice.submit(candidate("a", "one"))["result"], "correct"
        )

    def test_raw_values_and_bad_hashes_cannot_bypass_candidate_lane(self):
        lane = SubmissionLane(
            FakeClient([{"result": "correct"}]),
            CONFIG,
            lambda task_id: True,
        )
        with self.assertRaises(TypeError):
            lane.submit("task", "answer")
        bad = candidate("a", "one")
        bad = Candidate(
            **{**bad.__dict__, "answer_sha256": "not-the-answer-hash"}
        )
        with self.assertRaises(SubmissionRejected):
            lane.submit(bad)

    def test_client_calls_are_serialized_across_threads(self):
        class BlockingClient:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def submit(self, task_id, answer):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.01)
                with self.lock:
                    self.active -= 1
                return {"result": "correct"}

        config = SimpleNamespace(
            submission_interval_seconds=0,
            board_poll_seconds=1,
        )
        client = BlockingClient()
        lane = SubmissionLane(client, config, lambda task_id: True)
        threads = [
            threading.Thread(
                target=lane.submit, args=(candidate(str(index), str(index)),)
            )
            for index in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(client.max_active, 1)


if __name__ == "__main__":
    unittest.main()

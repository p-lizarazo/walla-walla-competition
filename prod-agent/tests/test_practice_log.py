from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from practice_log import PracticeAttemptLog


class PracticeAttemptLogTests(unittest.TestCase):
    def test_logs_good_and_bad_attempts_without_raw_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "attempts.jsonl"
            log = PracticeAttemptLog(str(path))
            log.append(task_id="PR-A1", result="correct", answer="secret")
            log.append(task_id="PR-A2", result="incorrect", confidence=0.7)

            rows = [
                json.loads(line) for line in path.read_text().splitlines()
            ]

        self.assertEqual([row["result"] for row in rows], ["correct", "incorrect"])
        self.assertNotIn("answer", rows[0])


if __name__ == "__main__":
    unittest.main()

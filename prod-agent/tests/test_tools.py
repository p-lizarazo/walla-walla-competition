from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import unittest
import uuid
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DashboardSnapshot, TaskDetail
from tools import (
    BoardStatusProvider,
    CandidateWriter,
    DashboardStatusProvider,
    EventWebSession,
    FileToolError,
    ProblemStatusProvider,
    PythonExecutor,
    TaskFiles,
    WebToolError,
    resolve_task_path,
    shared_cpu_semaphore,
)

TEST_ROOT = PROJECT_ROOT / "tests" / ".tool-test-work"


def tearDownModule() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


class WorkdirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.workdir = TEST_ROOT / uuid.uuid4().hex
        self.workdir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


class FileToolsTests(WorkdirTestCase):
    def test_resolve_rejects_traversal_and_symlink_escape(self) -> None:
        outside = TEST_ROOT / f"outside-{uuid.uuid4().hex}"
        outside.mkdir()
        self.addCleanup(shutil.rmtree, outside, True)
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        self.assertRaises(
            FileToolError, resolve_task_path, self.workdir, "../outside"
        )
        try:
            (self.workdir / "link").symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError):
            return
        self.assertRaises(
            FileToolError, resolve_task_path, self.workdir, "link/secret.txt"
        )

    def test_list_read_and_search_are_bounded(self) -> None:
        (self.workdir / "a.txt").write_text(
            "alpha\nneedle one\nneedle two\n", encoding="utf-8"
        )
        tools = TaskFiles(
            self.workdir,
            max_output_chars=30,
            max_search_matches=1,
        )
        self.assertEqual(tools.list()[0].path, "a.txt")
        self.assertEqual(tools.read("a.txt", start_line=2, end_line=2), "needle one")
        matches = tools.search("needle")
        self.assertEqual(len(matches), 1)
        self.assertLessEqual(len(tools.search_text("needle")), 30)

    def test_archive_listing_and_extraction_reject_traversal(self) -> None:
        good = self.workdir / "good.zip"
        with zipfile.ZipFile(good, "w") as bundle:
            bundle.writestr("nested/data.txt", "value")
        tools = TaskFiles(self.workdir)
        self.assertEqual(tools.list_archive("good.zip")[0].path, "nested/data.txt")
        self.assertEqual(
            tools.extract_archive("good.zip", destination="out"),
            ("out/nested/data.txt",),
        )
        self.assertEqual(
            (self.workdir / "out/nested/data.txt").read_text(encoding="utf-8"),
            "value",
        )

        bad = self.workdir / "bad.zip"
        with zipfile.ZipFile(bad, "w") as bundle:
            bundle.writestr("../escape.txt", "bad")
        with self.assertRaises(FileToolError):
            tools.extract_archive("bad.zip")


class PythonExecutorTests(WorkdirTestCase):
    def test_execution_confines_cwd_strips_credentials_and_caps_output(self) -> None:
        os.environ["TEAM_API_KEY"] = "should-not-leak"
        self.addCleanup(os.environ.pop, "TEAM_API_KEY", None)
        executor = PythonExecutor(
            self.workdir,
            timeout_seconds=2,
            max_output_bytes=160,
            memory_mb=512,
            cpu_slots=1,
        )
        result = executor.run(
            "import os\n"
            "print(os.getcwd())\n"
            "print(os.environ.get('TEAM_API_KEY'))\n"
            "print('x' * 500)\n"
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(str(self.workdir), result.stdout)
        self.assertIn("None", result.stdout)
        self.assertTrue(result.truncated)
        self.assertLessEqual(
            len((result.stdout + result.stderr).encode("utf-8")), 160
        )
        with self.assertRaises(FileToolError):
            executor.run("pass", cwd="..")

    def test_timeout_and_shared_semaphore(self) -> None:
        executor = PythonExecutor(
            self.workdir, timeout_seconds=1, memory_mb=512, cpu_slots=1
        )
        result = executor.run("import time; time.sleep(5)")
        self.assertTrue(result.timed_out)
        self.assertIs(shared_cpu_semaphore(1), shared_cpu_semaphore(1))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/set":
            self.send_response(302)
            self.send_header("Set-Cookie", "step=ready; Path=/")
            self.send_header("Location", "/check")
            self.end_headers()
            return
        if self.path == "/check":
            body = self.headers.get("Cookie", "").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/large":
            body = b"x" * 200
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", self.headers.get("Content-Type", ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class WebToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_cookies_post_json_and_body_cap(self) -> None:
        session = EventWebSession(self.base, max_body_bytes=32)
        self.assertIn("step=ready", session.get("/set").text)
        response = session.post("/echo", json={"answer": 42})
        self.assertEqual(json.loads(response.text), {"answer": 42})
        large = session.get("/large")
        self.assertTrue(large.truncated)
        self.assertEqual(len(large.body), 32)

    def test_rejects_other_hosts_and_methods(self) -> None:
        session = EventWebSession(self.base)
        with self.assertRaises(WebToolError):
            session.get("https://example.com/")
        with self.assertRaises(WebToolError):
            session.request("DELETE", "/check")


class StatusAndEvidenceTests(WorkdirTestCase):
    def test_game_status_is_round_agnostic_and_practice_is_explicit(self) -> None:
        qualifier = BoardStatusProvider(
            snapshot={"phase": "round1", "boards": {"qual": []}}
        ).read()
        finale = BoardStatusProvider(
            snapshot={"phase": "game", "boards": {"main": []}}
        ).read()
        self.assertEqual(qualifier["competition_mode"], "scored")
        self.assertEqual(finale["competition_mode"], "scored")
        self.assertEqual(qualifier["playable_board"], "qual")
        self.assertEqual(finale["playable_board"], "main")
        self.assertNotIn("strategy", qualifier)
        self.assertNotIn("strategy", finale)

        practice = {"phase": "practice", "boards": {"practice": []}}
        self.assertIsNone(
            BoardStatusProvider(snapshot=practice).read()["playable_board"]
        )
        evaluation = BoardStatusProvider(
            snapshot=practice, evaluation_mode=True
        ).read()
        self.assertEqual(evaluation["playable_board"], "practice")
        self.assertEqual(evaluation["competition_mode"], "practice_eval")

    def test_status_providers_copy_and_scrub_snapshots(self) -> None:
        task = TaskDetail(
            id="PR-X",
            category="Test",
            title="Title",
            points=100,
            prompt="Prompt",
            files=("a.txt",),
            answer_format="exact",
        )
        problem = ProblemStatusProvider(snapshot=task)
        first = problem.read()
        first["title"] = "changed"
        self.assertEqual(problem.read()["title"], "Title")

        dashboard = DashboardStatusProvider(
            snapshot=DashboardSnapshot(
                {"score": 10, "api_key": "secret", "nested": {"token": "secret"}},
                time.monotonic(),
            )
        ).read()
        self.assertEqual(dashboard["payload"], {"score": 10, "nested": {}})

    def test_candidate_writer_validates_hashes_and_records_evidence(self) -> None:
        writer = CandidateWriter(self.workdir, "PR-X")
        candidate = writer.write(
            "ANSWER\n",
            method="parsed the input",
            deterministic_checks=("recomputed total",),
            independent_checks=("cross-checked parser",),
            assumptions=("input format is complete",),
            tool_errors=("one optional probe failed",),
            model_temperature=0.25,
            elapsed_seconds=1.5,
            tool_turns=3,
        )
        self.assertEqual(candidate.answer, "ANSWER")
        self.assertEqual(
            candidate.answer_sha256,
            hashlib.sha256(b"ANSWER").hexdigest(),
        )
        record = json.loads((self.workdir / "candidate.json").read_text())
        self.assertEqual(record["evidence"]["method"], "parsed the input")
        self.assertEqual((self.workdir / "answer.txt").read_text(), "ANSWER\n")
        with self.assertRaises(ValueError):
            writer.write("two\nlines\n", method="invalid")


if __name__ == "__main__":
    unittest.main()

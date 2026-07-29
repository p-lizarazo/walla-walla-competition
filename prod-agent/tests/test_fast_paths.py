from __future__ import annotations

import io
import pathlib
import tempfile
import unittest
import zipfile

from fast_paths import FastPathSolver
from models import TaskDetail
from tools.web import WebResponse


class FakeWebSession:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str) -> WebResponse:
        if url.endswith("/vault"):
            body = b"<p>VAULT-ABC123DEF456</p>"
        else:
            body = (
                b'<form method="post" action="login">'
                b'<input name="csrf" value="safe">'
                b'<input name="username"><input name="password"></form>'
            )
        return WebResponse(
            200,
            url,
            (("Content-Type", "text/html; charset=utf-8"),),
            body,
            False,
        )

    def post(self, url: str, *, form: dict[str, str]) -> WebResponse:
        self.posts.append((url, form))
        return WebResponse(
            200,
            "https://example.test/member/",
            (("Content-Type", "text/html; charset=utf-8"),),
            b'<a href="/vault">Vault</a>',
            False,
        )


def task(
    prompt: str,
    *,
    files: tuple[str, ...] = (),
    category: str = "test",
) -> TaskDetail:
    return TaskDetail(
        id="Q-X",
        category=category,
        title="test",
        points=100,
        prompt=prompt,
        files=files,
        answer_format="exact",
    )


class FastPathSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = FastPathSolver()

    def test_policy_section_reads_only_requested_section(self):
        prompt = (
            "What is the maximum reimbursement for a single claim, in dollars, "
            "under Section 10.6?"
        )
        text = """
Section 10.5 - Other
The maximum reimbursement for a single claim under this section is $99,999.
Section 10.6 - Target
The maximum reimbursement for a single claim under this section is $4,250.
Section 10.7 - Other
The maximum reimbursement for a single claim under this section is $88,888.
"""
        with tempfile.TemporaryDirectory() as directory:
            pathlib.Path(directory, "policy_manual.txt").write_text(text)
            result = self.solver.solve(
                task(prompt, files=("policy_manual.txt",)),
                pathlib.Path(directory),
                FakeWebSession(),
            )
        self.assertEqual(result.candidate.answer, "4250")

    def test_leaderboard_applies_documented_tie_break(self):
        prompt = (
            "Per the spec, what should "
            "leaderboard([('z', 3), ('a', 3), ('m', 8)]) return?"
        )
        with tempfile.TemporaryDirectory() as directory:
            pathlib.Path(directory, "spec.md").write_text(
                "Order by score descending; equal scores alphabetically ascending."
            )
            result = self.solver.solve(
                task(prompt, files=("lib.py", "spec.md")),
                pathlib.Path(directory),
                FakeWebSession(),
            )
        self.assertEqual(result.candidate.answer, "['m', 'a', 'z']")

    def test_recurrence_sum_matches_direct_computation(self):
        prompt = (
            "A sequence is defined by x0 = 1; "
            "x_(n+1) = (2 * x_n + 1) mod 17. "
            "Apply the recurrence 5 times starting from x0. "
            "Compute the sum of all terms divisible by 3."
        )
        result = self.solver.solve(
            task(prompt),
            pathlib.Path("."),
            FakeWebSession(),
        )
        self.assertEqual(result.candidate.answer, "30")

    def test_infeasible_hash_work_uses_required_reason_code(self):
        prompt = (
            "Find an ASCII string whose SHA-256 digest begins with 68 zero BITS. "
            "If infeasible answer IMPOSSIBLE with one reason code: WORKFACTOR."
        )
        result = self.solver.solve(
            task(prompt),
            pathlib.Path("."),
            FakeWebSession(),
        )
        self.assertEqual(
            result.candidate.answer, "IMPOSSIBLE WORKFACTOR"
        )

    def test_regional_sales_deduplicates_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = pathlib.Path(directory, "regional_sales.zip")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "fx.csv",
                    "region,currency,usd_per_unit\nAA,X,2\nBB,Y,0.5\n",
                )
                archive.writestr(
                    "sales_AA_002.csv",
                    "# junk\norder_id,amount\none,\"1,000\"\ntwo,20\n",
                )
                archive.writestr(
                    "sales_BB_001.csv",
                    "ref,gross\none,999\nthree,10\nTOTAL,99999\nbad,row,extra\n",
                )
            prompt = (
                "Download regional_sales.zip. fx.csv has usd_per_unit. "
                "Keep only the FIRST occurrence of each transaction id."
            )
            result = self.solver.solve(
                task(prompt, files=("regional_sales.zip",)),
                pathlib.Path(directory),
                FakeWebSession(),
            )
        self.assertEqual(result.candidate.answer, "2045.00")

    def test_encrypted_bundle_follows_note(self):
        password = b"noclaf"
        inner_buffer = io.BytesIO()
        with zipfile.ZipFile(inner_buffer, "w") as inner:
            inner.writestr("token.txt", "VAULT-ABC123DEF456")
        encrypted = bytes(
            byte ^ password[index % len(password)]
            for index, byte in enumerate(inner_buffer.getvalue())
        )
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(pathlib.Path(directory, "bundle.zip"), "w") as archive:
                archive.writestr(
                    "note.txt",
                    "Use the largest .dat file, reversed, as the password. "
                    "inner.blob is XOR-encrypted.",
                )
                archive.writestr("falcon.dat", b"x" * 20)
                archive.writestr("small.dat", b"x")
                archive.writestr("inner.blob", encrypted)
            result = self.solver.solve(
                task(
                    "Download bundle.zip and start with note.txt.",
                    files=("bundle.zip",),
                ),
                pathlib.Path(directory),
                FakeWebSession(),
            )
        self.assertEqual(result.candidate.answer, "VAULT-ABC123DEF456")

    def test_members_vault_preserves_form_fields(self):
        web = FakeWebSession()
        result = self.solver.solve(
            task(
                "A token sits in the members-only vault at "
                "https://example.test/start/ username: alice password: secret"
            ),
            pathlib.Path("."),
            web,
        )
        self.assertEqual(result.candidate.answer, "VAULT-ABC123DEF456")
        self.assertEqual(
            web.posts,
            [
                (
                    "https://example.test/start/login",
                    {"csrf": "safe", "username": "alice", "password": "secret"},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()

from types import SimpleNamespace
import unittest

from confidence import (
    ConfidenceEngine,
    PracticeCalibration,
    urgent_threshold,
)
from models import Evidence


CONFIG = SimpleNamespace(
    strong_confidence_threshold=0.90,
    urgent_confidence_floor=0.80,
)


class ConfidenceTests(unittest.TestCase):
    def test_verified_direct_complete_evidence_is_strong(self):
        evidence = Evidence(
            method="parser",
            deterministic_checks=("exact total", "unique indices"),
            independent_checks=("second implementation",),
        )
        decision = ConfidenceEngine(CONFIG).assess(evidence)
        self.assertGreaterEqual(decision.probability, 0.90)
        self.assertTrue(decision.should_submit)

    def test_confidence_uses_practice_calibration(self):
        evidence = Evidence(
            method="parser",
            deterministic_checks=("schema valid",),
            independent_checks=("round trip",),
        )
        weak = ConfidenceEngine(
            CONFIG, {"parser": PracticeCalibration(0, 8)}
        ).assess(evidence)
        strong = ConfidenceEngine(
            CONFIG, {"parser": PracticeCalibration(8, 8)}
        ).assess(evidence)
        self.assertGreater(strong.probability, weak.probability)

    def test_missing_inputs_errors_and_assumptions_cap_confidence(self):
        evidence = Evidence(
            method="guess",
            deterministic_checks=("format only",),
            assumptions=("missing field means zero",),
            tool_errors=("parser crashed",),
            input_complete=False,
            direct_provenance=False,
        )
        decision = ConfidenceEngine(CONFIG, 0.99).assess(
            evidence, urgent=True
        )
        self.assertLess(decision.probability, decision.threshold)
        self.assertFalse(decision.should_submit)

    def test_unchecked_answer_is_capped_even_with_good_practice(self):
        decision = ConfidenceEngine(CONFIG, 0.99).assess(
            Evidence(method="memorized")
        )
        self.assertLess(decision.probability, 0.90)

    def test_urgent_threshold_stays_inside_configured_bounds(self):
        self.assertEqual(urgent_threshold(CONFIG, -1), 0.90)
        self.assertEqual(urgent_threshold(CONFIG, 2), 0.80)
        self.assertAlmostEqual(urgent_threshold(CONFIG, 0.5), 0.85)


if __name__ == "__main__":
    unittest.main()

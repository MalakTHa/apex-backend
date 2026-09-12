import hashlib
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import main

ORIGINAL = "def total(values: list[int]):\n    result = 0\n    for value in values:\n        result += value\n    return result\n"
CANDIDATE = "def total(values: list[int]):\n    return sum(values)\n"


class CandidateSimulationTests(unittest.TestCase):
    def request(self, **overrides):
        payload = dict(code=ORIGINAL, function_name="total", optimization_source="ai",
                       candidate_function_code=CANDIDATE,
                       candidate_fingerprint=hashlib.sha256(CANDIDATE.encode()).hexdigest())
        payload.update(overrides)
        with TestClient(main.app) as client:
            return client.post("/simulate", json=payload).json()

    def test_exact_candidate_static_analysis_without_optimization_or_runtime(self):
        with patch("main.ReplacementEngine") as engine, patch("main.suggest_ai_optimization") as ai, \
                patch("ai_optimizer.Groq") as groq, patch("main.verify_with_docker") as runtime, \
                patch("main.apply_after_override") as override:
            result = self.request()
        for mock in (engine, ai, groq, runtime, override):
            mock.assert_not_called()
        self.assertEqual(result["optimized_function_code"], CANDIDATE)
        self.assertEqual(result["simulation_kind"], "theoretical_complexity_projection")
        self.assertEqual(result["after"], main.analyze_function_node("total", main.ast.parse(CANDIDATE).body[0]))
        self.assertEqual(result["before"], main.analyze_function_node("total", main.ast.parse(ORIGINAL).body[0]))
        self.assertEqual(result["points"], main.build_simulation_points(result["before"], result["after"], 10, 100, 10))
        self.assertNotIn("verification_status", result)  # Static projection does not grant verification.

    def test_range_changes_keep_exact_candidate(self):
        first = self.request()
        second = self.request(start_n=20, end_n=60, step=20)
        self.assertEqual(first["candidate_fingerprint"], second["candidate_fingerprint"])
        self.assertEqual(second["optimized_function_code"], CANDIDATE)
        self.assertEqual([point["n"] for point in second["points"]], [20, 40, 60])

    def test_invalid_fingerprint_missing_or_changed_candidate_rejected(self):
        for override in ({"candidate_fingerprint": None}, {"candidate_function_code": None},
                         {"candidate_fingerprint": "0"*64}, {"candidate_function_code": CANDIDATE + "\n"}):
            with self.subTest(override=override), patch("main.ReplacementEngine") as engine:
                self.assertIn("error", self.request(**override))
                engine.assert_not_called()

    def test_invalid_syntax_name_and_api_rejected(self):
        for candidate in ("broken syntax !", "def other(values: list[int]): return 1", "def total(x: int): return 1"):
            result = self.request(candidate_function_code=candidate, candidate_fingerprint=hashlib.sha256(candidate.encode()).hexdigest())
            self.assertIn("error", result)

    def test_legacy_simple_and_replacement_routes_unchanged(self):
        for kind, builder in (("simple", "build_simple_simulation_result"), ("replacement", "build_replacement_simulation_result")):
            original_builder = getattr(main, builder)
            with patch("main." + builder, wraps=original_builder) as spy, patch("main.suggest_ai_optimization") as ai:
                result = self.request(optimization_source=None, optimization_type=kind, candidate_function_code=None, candidate_fingerprint=None)
            spy.assert_called_once()
            ai.assert_not_called()
            self.assertNotIn("error", result)
            self.assertEqual(result["optimization_type"], kind)
            self.assertNotIn("candidate_fingerprint", result)


if __name__ == "__main__":
    unittest.main()

import ast
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


SOURCE = "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n"
CANDIDATE = "def total(values):\n    return sum(values, 0)\n"
TRUSTED = "def unique(values):\n    result = []\n    for value in values:\n        if value not in result:\n            result.append(value)\n    return result\n"


class AIFallbackTests(unittest.TestCase):
    def setUp(self):
        # Every normal test is offline even if a real .env exists locally.
        patcher = patch("main.suggest_ai_optimization", return_value={
            "status": "candidate", "can_optimize": True,
            "optimized_function_code": CANDIDATE, "reason": "Use a reduction.",
            "error_code": None,
        })
        self.ai = patcher.start()
        self.addCleanup(patcher.stop)
        sdk = patch("ai_optimizer.Groq", side_effect=AssertionError("No real Groq in tests"))
        self.groq = sdk.start()
        self.addCleanup(sdk.stop)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def post(self, code=SOURCE, name="total", path="/optimize/replacement", **extra):
        response = self.client.post(path, json={"code": code, "function_name": name, **extra})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_trusted_pattern_does_not_call_ai_and_preserves_legacy_result(self):
        node = ast.parse(TRUSTED).body[0]
        engine = main.ReplacementEngine("unique")
        optimized = engine.optimize(node)
        full_code = main.replace_function_code(TRUSTED, "unique", optimized)
        before = main.analyze_function_node("unique", node)
        after = main.apply_after_override(main.analyze_function_node("unique", ast.parse(full_code).body[0]), engine.after_override)
        result = self.post(TRUSTED, "unique")
        legacy = {"function": "unique", "optimization_type": "replacement", "optimized_function_code": optimized,
                  "full_code": full_code, "changes": engine.changes, "before": before, "after": after, "reason": engine.reason}
        self.assertEqual({k: result[k] for k in legacy}, legacy)
        self.assertEqual(result["optimization_source"], "trusted_pattern")
        self.assertEqual(result["verification_status"], "trusted_pattern")
        self.assertTrue(result["pattern_applied"])
        self.assertFalse(result["ai_used"])
        self.ai.assert_not_called()
        self.groq.assert_not_called()

    def test_unsupported_calls_ai_with_only_selected_source_and_analysis(self):
        code = "# other context\nother = 7\n\n" + SOURCE + "\ndef unrelated():\n    return 9\n"
        self.post(code)
        self.ai.assert_called_once()
        source, analysis = self.ai.call_args.args
        self.assertEqual(source, SOURCE.rstrip())
        self.assertNotIn("unrelated", source)
        self.assertEqual(analysis, main.analyze_function_node("total", ast.parse(SOURCE).body[0]))

    def test_candidate_metadata_and_full_code_preview(self):
        code = "other = 7\n\n" + SOURCE + "\ndef unrelated():\n    return 9\n"
        result = self.post(code)
        self.assertEqual(result["optimization_source"], "ai")
        self.assertEqual(result["verification_status"], "not_verified")
        self.assertTrue(result["ai_used"])
        self.assertFalse(result["pattern_applied"])
        self.assertEqual(result["optimized_function_code"], CANDIDATE)
        self.assertEqual(result["full_code"], main.replace_function_code(code, "total", CANDIDATE))
        self.assertIn("def unrelated()", result["full_code"])
        ast.parse(result["full_code"])

    def test_candidate_is_reanalyzed_without_trusted_override(self):
        self.ai.return_value["after"] = {"time_complexity": "AI claim"}
        with patch("main.analyze_function_node", wraps=main.analyze_function_node) as analyze:
            with patch("main.apply_after_override", side_effect=AssertionError("No trusted override for AI")):
                result = self.post()
        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(result["after"], main.analyze_function_node("total", ast.parse(CANDIDATE).body[0]))
        self.assertNotIn("AI claim", str(result))

    def test_declined_retains_original(self):
        self.ai.return_value = {"status": "declined", "can_optimize": False, "reason": "Insufficient input assumptions."}
        result = self.post()
        self.assertEqual(result["optimization_source"], "none")
        self.assertEqual(result["verification_status"], "not_applicable")
        self.assertTrue(result["ai_used"])
        self.assertEqual(result["reason"], self.ai.return_value["reason"])
        self.assertEqual(result["full_code"], SOURCE)
        self.assertEqual(result["after"], result["before"])

    def test_ai_error_is_nonfatal(self):
        self.ai.return_value = {"status": "error", "can_optimize": False, "error_code": "rate_limited", "reason": "AI candidate generation failed: rate_limited"}
        result = self.post()
        self.assertEqual(result["optimization_source"], "none")
        self.assertEqual(result["verification_status"], "ai_error")
        self.assertEqual(result["error_code"], "rate_limited")
        self.assertEqual(result["full_code"], SOURCE)
        self.assertNotIn("error", result)

    def test_unexpected_exception_does_not_leak(self):
        self.ai.side_effect = RuntimeError("fake-secret-sdk-details")
        result = self.post()
        self.assertEqual(result["verification_status"], "ai_error")
        self.assertNotIn("fake-secret", str(result))
        self.assertEqual(result["full_code"], SOURCE)

    def test_invalid_candidate_is_rejected_again_at_endpoint(self):
        for code in ("def total(values):\n    return (", "", "pass", "import os\n" + CANDIDATE,
                     CANDIDATE.replace("total", "changed"), CANDIDATE.replace("(values):", "(values, extra):")):
            with self.subTest(code=code):
                self.ai.return_value["optimized_function_code"] = code
                result = self.post()
                self.assertEqual(result["verification_status"], "ai_error")
                self.assertEqual(result["error_code"], "invalid_ai_candidate")
                self.assertEqual(result["full_code"], SOURCE)

    def test_invalid_ai_contract(self):
        for response in (None, {}):
            with self.subTest(response=response):
                self.ai.return_value = response
                self.assertEqual(self.post()["verification_status"], "ai_error")

    def test_decline_or_false_flag_is_not_applicable(self):
        for suggestion in ({"can_optimize": False}, {"status": "candidate", "can_optimize": False},
                           {"status": "declined"}, {"status": "no_optimization"}):
            with self.subTest(suggestion=suggestion):
                self.ai.return_value = suggestion
                result = self.post()
                self.assertEqual(result["optimization_source"], "none")
                self.assertEqual(result["verification_status"], "not_applicable")
                self.assertEqual(result["reason"], "No further optimization was found for this function.")
                self.assertEqual(result["full_code"], SOURCE)
                self.assertEqual(result["before"], result["after"])
                self.assertIsNone(result["error_code"])

    def test_identical_ast_is_not_a_candidate(self):
        for candidate in (SOURCE, SOURCE.replace("    ", "  ").replace("result = 0", "result=0  # same AST")):
            with self.subTest(candidate=candidate):
                self.ai.return_value["optimized_function_code"] = candidate
                result = self.post()
                self.assertEqual(result["optimization_source"], "none")
                self.assertEqual(result["verification_status"], "not_applicable")
                self.assertEqual(result["full_code"], SOURCE)
                self.assertEqual(result["changes"], [])
                self.assertEqual(result["reason"], "No further optimization was found for this function.")

    def test_different_ast_same_big_o_remains_candidate(self):
        self.ai.return_value["optimized_function_code"] = "def total(value):\n    return value * value"
        result = self.post("def total(value):\n    return value ** 2")
        self.assertEqual(result["optimization_source"], "ai")
        self.assertEqual(result["verification_status"], "not_verified")
        self.assertEqual(result["before"]["time_complexity"], result["after"]["time_complexity"])

    def test_append_early_return_also_calls_ai(self):
        source = "def total(values):\n    result = []\n    for item in values:\n        result.append(item)\n    return result"
        self.post(source)
        self.ai.assert_called_once()

    def test_simple_optimization_unchanged(self):
        result = self.post(path="/optimize/simple")
        self.assertEqual(set(result), {"function", "optimization_type", "optimized_function_code", "full_code", "changes", "before", "after"})
        self.assertEqual(result["optimization_type"], "simple")
        self.assertEqual(ast.dump(ast.parse(result["full_code"])), ast.dump(ast.parse(SOURCE)))
        self.ai.assert_not_called()

    def test_simulation_never_uses_ai(self):
        for optimization_type in ("simple", "replacement"):
            with self.subTest(optimization_type=optimization_type):
                result = self.post(path="/simulate", optimization_type=optimization_type)
                self.assertIn("points", result)
                self.assertNotIn("ai_used", result)
                self.assertEqual(ast.dump(ast.parse(result["full_code"])), ast.dump(ast.parse(SOURCE)))
        self.ai.assert_not_called()
        self.groq.assert_not_called()


if __name__ == "__main__":
    unittest.main()

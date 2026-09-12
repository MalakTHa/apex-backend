import ast
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import CodeInput, ReplacementEngine, app, optimize_replacement


class ReplacementPatternTests(unittest.TestCase):
    def check_pattern(self, source, expected):
        node = ast.parse(source).body[0]
        engine = ReplacementEngine(node.name)
        self.assertFalse(engine.pattern_applied)
        result = engine.optimize(node)
        self.assertEqual(engine.pattern_applied, expected)
        if not expected:
            self.assertEqual(ast.dump(ast.parse(result).body[0]), ast.dump(node))
        return engine, result

    def test_no_arguments_and_no_pattern(self):
        for source in ("def f():\n    return 1", "def f(x):\n    return x * x"):
            with self.subTest(source=source):
                self.check_pattern(source, False)

    def test_append_no_transformation_paths(self):
        for loop in (
            "for item in values:\n        result.append(item)",
            "for item in range(10, 0, -1):\n        result.append(item)",
            "for item in range(10, 0, -2):\n        result.append(item)",
            "for item in values:\n        if len(result) < 2:\n            result.append(item)",
        ):
            with self.subTest(loop=loop):
                self.check_pattern("def f(values):\n    result = []\n    " + loop + "\n    return result", False)

    def test_conditional_comprehension_applied(self):
        self.check_pattern("def f(values):\n    result = []\n    for item in values:\n        if item > 0:\n            result.append(item)\n    return result", True)

    def test_seen_set_applied(self):
        self.check_pattern("def f(values):\n    result = []\n    for item in values:\n        if item not in result:\n            result.append(item)\n    return result", True)

    def test_already_identical_trusted_template(self):
        source = "def f(values):\n    freq = {}\n    for value in values:\n        freq[value] += 1\n    return freq"
        _, result = self.check_pattern(source, True)
        # Compare identical trusted output even if the detector selected a branch.
        node = ast.parse(result).body[0]
        engine = ReplacementEngine("f")
        with patch.object(engine, "_optimize_trusted", return_value=result):
            engine.optimize(node)
        self.assertFalse(engine.pattern_applied)

    def test_flag_resets_on_reuse(self):
        engine = ReplacementEngine("f")
        with patch.object(engine, "_optimize_trusted", return_value="def f(x):\n    return x + 1"):
            engine.optimize(ast.parse("def f(x):\n    return x").body[0])
        self.assertTrue(engine.pattern_applied)
        engine.optimize(ast.parse("def f(x):\n    return x").body[0])
        self.assertFalse(engine.pattern_applied)

    def test_endpoint_retains_original_fields_on_decline(self):
        source = "def f(x):\n    return x * x"
        with patch("main.suggest_ai_optimization", return_value={"status": "declined", "can_optimize": False, "reason": "No proposal"}):
            result = optimize_replacement(CodeInput(code=source, function_name="f"))
        self.assertTrue({"function", "optimization_type", "optimized_function_code", "full_code", "changes", "before", "after", "reason"} <= set(result))
        self.assertEqual(result["full_code"], source)

    def test_existing_endpoints_without_ai(self):
        payload = {"code": "def f(x):\n    return x * x", "function_name": "f"}
        with patch("main.suggest_ai_optimization", side_effect=AssertionError("AI must remain isolated")) as ai_call:
            with TestClient(app) as client:
                for path in ("/analyze", "/optimize/simple", "/simulate"):
                    with self.subTest(path=path):
                        response = client.post(path, json=payload)
                        self.assertEqual(response.status_code, 200)
                        self.assertNotIn("error", response.json())
                        self.assertEqual(response.json()["function"], "f")
            ai_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()

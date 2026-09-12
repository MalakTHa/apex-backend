"""Explicitly run with unittest; excluded from default test_*.py discovery."""

import ast
import os
import unittest

from dotenv import load_dotenv

from ai_optimizer import ENV_PATH, suggest_ai_optimization
from main import ReplacementEngine, analyze_function_node


class ManualGroqIntegrationTest(unittest.TestCase):
    def test_live_candidate(self):
        load_dotenv(dotenv_path=ENV_PATH, override=False)
        if not os.getenv("GROQ_API_KEY", "").strip():
            self.skipTest("GROQ_API_KEY is not configured; no request made")
        # Synthetic example only. Neither the original nor the candidate is run.
        source = "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n"
        original = ast.parse(source).body[0]
        engine = ReplacementEngine(original.name)
        engine.optimize(original)
        self.assertFalse(engine.pattern_applied)
        result = suggest_ai_optimization(source, analyze_function_node(original.name, original))
        self.assertIn("can_optimize", result)
        self.assertIn(result["status"], ("candidate", "declined"), msg=result["error_code"])
        if result["can_optimize"]:
            self.assertTrue(result["optimized_function_code"].strip())
            tree = ast.parse(result["optimized_function_code"])
            self.assertEqual(len(tree.body), 1)
            proposed = tree.body[0]
            self.assertEqual(proposed.name, original.name)
            self.assertEqual(ast.dump(proposed.args), ast.dump(original.args))
        print("Groq connection succeeded; result status:", result["status"])


if __name__ == "__main__":
    unittest.main()

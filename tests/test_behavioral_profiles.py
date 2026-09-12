import ast
import unittest
from unittest.mock import Mock, patch
from verification_input_profile import resolve_input_profiles
from verification_inputs import generate_test_cases
from verification import verify_candidate
from test_verification_platform import BUBBLE


class BehavioralProfileTests(unittest.TestCase):
    def profile(self, source):
        return resolve_input_profiles(ast.parse(source).body[0], behavioral=True)[0]

    def test_unannotated_mutable_ordered_sequence(self):
        source = BUBBLE.replace(": list[int]", "")
        with patch("ai_optimizer.Groq", side_effect=AssertionError("No AI")):
            profile = self.profile(source)
            cases = generate_test_cases(ast.parse(source).body[0]).cases
        self.assertTrue(profile.supported)
        self.assertEqual(profile.source, "ast_inference")
        self.assertEqual(len(cases), 6)
        self.assertTrue(any(c.args[0] and type(c.args[0][0]) is int for c in cases))
        self.assertTrue(any(c.args[0] and type(c.args[0][0]) is str for c in cases))
        self.assertEqual(cases, generate_test_cases(ast.parse(source).body[0]).cases)
        self.assertIsNot(cases[0].args, cases[1].args)
        # Existing benchmark callers retain their original inference contract.
        self.assertIsNone(resolve_input_profiles(ast.parse(source).body[0])[0].inferred_type)

    def test_annotations_take_priority(self):
        self.assertEqual(self.profile(BUBBLE).inferred_type, "list[int]")
        self.assertEqual(self.profile(BUBBLE.replace("list[int]", "list[str]")).inferred_type, "list[str]")
        self.assertFalse(self.profile("def f(x: float): return x - 1").supported)

    def test_clear_scalar_and_element_operations(self):
        for source, kind in [
            ("def f(x): return x % 2", "int"),
            ("def f(x): return x / 2", "int"),
            ("def f(x): return x > 0", "int"),
            ("def f(x): return x.strip()", "str"),
            ("def f(x):\n for item in x:\n  item.lower()", "list[str]"),
            ("def f(x):\n for item in x:\n  item % 2", "list[int]"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(self.profile(source).inferred_type, kind)

    def test_insufficient_conflicting_and_rebound_remain_inconclusive(self):
        for source in ["def f(x): return x", "def f(x): return len(x)",
                       "def f(x):\n for item in x:\n  pass",
                       "def f(x): return x.strip() + (x - 1)",
                       "def f(x):\n for item in x:\n  item.lower()\n  item % 2",
                       "def f(x):\n x = 1\n return x + 2"]:
            with self.subTest(source=source):
                self.assertFalse(self.profile(source).supported)
                runtime = Mock()
                result = verify_candidate(source, source, "f", runtime)
                self.assertEqual(result.verification_status, "inconclusive")
                runtime.execute_function.assert_not_called()

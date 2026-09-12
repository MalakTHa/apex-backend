"""Deterministic built-in costs and their surrounding loop context."""
import ast
import unittest

from main import FunctionAnalyzer, estimate_complexity


def complexity(source):
    analyzer = FunctionAnalyzer()
    analyzer.visit(ast.parse(source).body[0])
    return estimate_complexity(analyzer)


class StaticBuiltinComplexityTests(unittest.TestCase):
    def test_standalone_calls(self):
        for expression, expected in [
            ("len(data)", "O(1)"),
            ("sorted(data)", "O(n log n)"),
            ("data.sort()", "O(n log n)"),
            ("sum(data)", "O(n)"),
            ("min(data)", "O(n)"),
            ("max(data)", "O(n)"),
            ("set(data)", "O(n)"),
            ("list(data)", "O(n)"),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(complexity(f"def f(data):\n    return {expression}"), expected)

    def test_calls_inside_loops(self):
        for expression, expected in [
            ("len(data)", "O(n)"),
            ("sorted(data)", "O(n\u00b2 log n)"),
            ("data.sort()", "O(n\u00b2 log n)"),
            *[(f"{name}(data)", "O(n\u00b2)") for name in ("sum", "min", "max", "set", "list")],
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(complexity(f"def f(data):\n    for item in data:\n        {expression}"), expected)

    def test_nested_loop_sort(self):
        self.assertEqual(complexity("def f(data):\n    for x in data:\n        for y in data:\n            sorted(data)"), "O(n\u00b3 log n)")

    def test_iterable_call_is_evaluated_before_loop(self):
        self.assertEqual(complexity("def f(data):\n    for x in sorted(data):\n        pass"), "O(n log n)")

    def test_sequential_costs_take_dominant_order(self):
        self.assertEqual(complexity("def f(data):\n    sorted(data)\n    for x in data:\n        for y in data:\n            pass"), "O(n\u00b2)")
        self.assertEqual(complexity("def f(data):\n    return sum(sorted(data))"), "O(n log n)")

    def test_existing_user_defined_and_recursive_calls(self):
        self.assertEqual(complexity("def f(data):\n    return custom(data)"), "O(1)")
        self.assertEqual(complexity("def f(data):\n    return f(data)"), "Recursive")

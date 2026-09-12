"""Fake observations only: no original or candidate Python code is executed."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import verification
import verification_inputs
from verification import (
    ComparisonPolicy, ExecutionResult, VerificationResult,
    compare_results, verify_candidate,
)
from verification_inputs import InputGenerationResult, generate_test_cases


ORIGINAL = "def transform(value: int):\n    return value\n"
CANDIDATE = "def transform(value: int):\n    return value + 0\n"


class FakeExecutor:
    """Return scripted observations; never interpret function_code."""
    def __init__(self, observations=()):
        self.observations = iter(observations)
        self.calls = []

    def execute_function(self, function_code, function_name, args, kwargs):
        self.calls.append((function_code, function_name, deepcopy(args), deepcopy(kwargs)))
        observation = next(self.observations, ExecutionResult("returned", 0, post_args=[], post_kwargs={}))
        if isinstance(observation, Exception):
            raise observation
        return observation


class InputGenerationTests(unittest.TestCase):
    def generate(self, parameters, body="return None"):
        return generate_test_cases(ast.parse(f"def transform({parameters}):\n    {body}").body[0])

    def test_supported_types_and_edge_cases(self):
        for annotation, expected_type in (("int", int), ("str", str), ("list[int]", list), ("list[str]", list)):
            with self.subTest(annotation=annotation):
                result = self.generate(f"value: {annotation}")
                self.assertEqual(result.status, "supported")
                self.assertEqual(len(result.cases), 6)
                self.assertTrue(all(type(case.args[0]) is expected_type for case in result.cases))
                if annotation.startswith("list"):
                    self.assertEqual(result.cases[0].args[0], [])
                    item_type = int if annotation == "list[int]" else str
                    self.assertTrue(all(type(item) is item_type for case in result.cases for item in case.args[0]))
        self.assertIn(-1, [case.args[0] for case in self.generate("value: int").cases])
        self.assertIn("", [case.args[0] for case in self.generate("value: str").cases])

    def test_two_integers_have_bounded_cartesian_coverage(self):
        result = self.generate("left: int, right: int")
        self.assertEqual(len(result.cases), 36)
        pairs = {tuple(case.args) for case in result.cases}
        self.assertEqual(len(pairs), 36)
        self.assertIn((0, -1), pairs)
        self.assertIn((-1, 0), pairs)

    def test_positional_only_and_keyword_only(self):
        result = self.generate("left: int, /, *, right: str")
        self.assertEqual(len(result.cases), 36)
        self.assertEqual(result.cases[0].args, [0])
        self.assertEqual(result.cases[0].kwargs, {"right": ""})

    def test_all_keyword_only(self):
        case = self.generate("*, value: list[str]").cases[0]
        self.assertEqual(case.args, [])
        self.assertEqual(case.kwargs, {"value": []})

    def test_forward_reference_annotations_are_parsed(self):
        self.assertEqual(self.generate("value: 'list[int]'").status, "supported")
        self.assertEqual(self.generate("value: 'unknown()'").status, "unsupported")

    def test_defaults_are_explicitly_supplied(self):
        cases = self.generate("value: int = 8").cases
        self.assertTrue(all(len(case.args) == 1 for case in cases))

    def test_unsupported_signatures(self):
        for parameters in ("", "value", "value=1", "value: float", "value: bool",
                           "value: dict[str, int]", "value: tuple[int]", "value: int | None",
                           "*values: int", "**values: int", "a: int, b: int, c: int", "a: int, a: int"):
            with self.subTest(parameters=parameters):
                result = self.generate(parameters)
                self.assertEqual(result.status, "unsupported")
                self.assertEqual(result.cases, ())

    def test_async_generators_and_decorators_unsupported(self):
        for source in ("async def transform(value: int):\n    return value",
                       "@decorator\ndef transform(value: int):\n    return value",
                       "def transform(value: int):\n    yield value"):
            with self.subTest(source=source):
                self.assertEqual(generate_test_cases(ast.parse(source).body[0]).status, "unsupported")

    def test_determinism_and_no_mutable_input_aliasing(self):
        first = self.generate("left: list[int], right: list[int]")
        second = self.generate("left: list[int], right: list[int]")
        self.assertEqual(first, second)
        first.cases[0].args[0].append(12345)
        self.assertEqual(first.cases[0].args[1], [])
        self.assertEqual(first.cases[1].args[0], [])
        self.assertEqual(second.cases[0].args[0], [])


class ComparisonTests(unittest.TestCase):
    def test_return_values_match(self):
        for value in (None, True, 42, 1.5, "text", [1, 2], {"b": [True], "a": None}):
            with self.subTest(value=value):
                self.assertEqual(compare_results(ExecutionResult("returned", value, post_args=[], post_kwargs={}), ExecutionResult("returned", deepcopy(value), post_args=[], post_kwargs={})).status, "passed")

    def test_types_and_list_order_are_not_coerced(self):
        for original, candidate in ((True, 1), (1, 1.0), ([1, 2], [2, 1]), ("1", 1)):
            with self.subTest(original=original, candidate=candidate):
                self.assertEqual(compare_results(ExecutionResult("returned", original, post_args=[], post_kwargs={}), ExecutionResult("returned", candidate, post_args=[], post_kwargs={})).status, "failed")

    def test_dictionary_key_order_is_not_compared(self):
        result = compare_results(ExecutionResult("returned", {"a": 1, "b": 2}, post_args=[], post_kwargs={}), ExecutionResult("returned", {"b": 2, "a": 1}, post_args=[], post_kwargs={}))
        self.assertEqual(result.status, "passed")

    def test_exception_versus_return_is_rejected_in_both_directions(self):
        returned = ExecutionResult("returned", 1, post_args=[], post_kwargs={})
        raised = ExecutionResult("raised", exception_type="builtins.ValueError", exception_message="bad", post_args=[], post_kwargs={})
        self.assertEqual(compare_results(returned, raised).status, "failed")
        self.assertEqual(compare_results(raised, returned).status, "failed")

    def test_equal_exception_requires_explicit_policy(self):
        raised = ExecutionResult("raised", exception_type="builtins.ValueError", exception_message="bad", post_args=[], post_kwargs={})
        self.assertEqual(compare_results(raised, raised).status, "inconclusive")
        self.assertEqual(compare_results(raised, raised, ComparisonPolicy(frozenset({"builtins.ValueError"}))).status, "passed")

    def test_exception_type_and_message_must_both_match(self):
        original = ExecutionResult("raised", exception_type="builtins.ValueError", exception_message="bad", post_args=[], post_kwargs={})
        for candidate in (ExecutionResult("raised", exception_type="builtins.TypeError", exception_message="bad", post_args=[], post_kwargs={}),
                          ExecutionResult("raised", exception_type="builtins.ValueError", exception_message="different", post_args=[], post_kwargs={})):
            self.assertEqual(compare_results(original, candidate).status, "failed")

    def test_timeout_is_inconclusive_even_on_both_sides(self):
        for other in (ExecutionResult("returned", 1, post_args=[], post_kwargs={}), ExecutionResult("timeout")):
            self.assertEqual(compare_results(ExecutionResult("timeout"), other).status, "inconclusive")

    def test_serialization_failure_is_distinct_from_timeout(self):
        comparison = compare_results(ExecutionResult("serialization_error"), ExecutionResult("returned", 1, post_args=[], post_kwargs={}))
        self.assertEqual(comparison.status, "inconclusive")
        self.assertEqual(comparison.kind, "serialization_failure")

    def test_unsupported_values_are_inconclusive(self):
        cycle = []
        cycle.append(cycle)
        for value in (object(), (1, 2), {1, 2}, {1: "value"}, float("nan"), float("inf"), cycle):
            with self.subTest(type_name=type(value).__name__):
                self.assertEqual(compare_results(ExecutionResult("returned", value, post_args=[], post_kwargs={}), ExecutionResult("returned", value, post_args=[], post_kwargs={})).status, "inconclusive")

    def test_worker_error_and_invalid_observations(self):
        for observation in (None, ExecutionResult("unknown"), ExecutionResult([]), ExecutionResult("raised", post_args=[], post_kwargs={}), ExecutionResult("error"),
                            ExecutionResult("returned", 0, exception_type="builtins.ValueError", post_args=[], post_kwargs={}), ExecutionResult("timeout", 1)):
            with self.subTest(observation=observation):
                self.assertEqual(compare_results(observation, ExecutionResult("returned", 0, post_args=[], post_kwargs={})).status, "error")


class VerificationEngineTests(unittest.TestCase):
    def verify(self, executor, **kwargs):
        return verify_candidate(ORIGINAL, CANDIDATE, "transform", executor, **kwargs)

    def test_all_observed_outputs_match(self):
        executor = FakeExecutor()
        result = self.verify(executor)
        self.assertEqual(result.verification_status, "verified")
        self.assertEqual((result.tests_total, result.tests_run, result.tests_passed, result.tests_failed), (6, 6, 6, 0))
        self.assertEqual(len(executor.calls), 12)
        for original_call, candidate_call in zip(executor.calls[::2], executor.calls[1::2]):
            self.assertEqual(original_call[0], ORIGINAL)
            self.assertEqual(candidate_call[0], CANDIDATE)
            self.assertEqual(original_call[1:], candidate_call[1:])
        self.assertEqual(result.mismatches, [])
        json.dumps(result.to_dict())

    def test_one_different_output_is_rejected(self):
        result = self.verify(FakeExecutor([ExecutionResult("returned", 1, post_args=[], post_kwargs={}), ExecutionResult("returned", 2, post_args=[], post_kwargs={})]))
        self.assertEqual(result.verification_status, "rejected")
        self.assertEqual((result.tests_passed, result.tests_failed), (5, 1))
        self.assertEqual(result.mismatches[0].case_id, "case_01")
        self.assertEqual(result.mismatches[0].kind, "return_value")

    def test_candidate_exception_is_rejected(self):
        result = self.verify(FakeExecutor([ExecutionResult("returned", 0, post_args=[], post_kwargs={}), ExecutionResult("raised", exception_type="builtins.ValueError", exception_message="bad", post_args=[], post_kwargs={})]))
        self.assertEqual(result.verification_status, "rejected")
        self.assertEqual(result.mismatches[0].kind, "exception_behavior")

    def test_timeout_prevents_verified(self):
        result = self.verify(FakeExecutor([ExecutionResult("timeout"), ExecutionResult("timeout")]))
        self.assertEqual(result.verification_status, "inconclusive")
        self.assertEqual((result.tests_inconclusive, result.tests_passed), (1, 5))

    def test_serialization_failure_prevents_verified(self):
        result = self.verify(FakeExecutor([ExecutionResult("returned", 0, post_args=[], post_kwargs={}), ExecutionResult("serialization_error")]))
        self.assertEqual(result.verification_status, "inconclusive")
        self.assertEqual(result.issues[0].kind, "serialization_failure")

    def test_unsupported_generation_does_not_call_executor(self):
        executor = FakeExecutor()
        source = "def transform(value):\n    return value"
        result = verify_candidate(source, source, "transform", executor)
        self.assertEqual(result.verification_status, "inconclusive")
        self.assertEqual(result.tests_total, 0)
        self.assertEqual(executor.calls, [])

    def test_empty_generated_cases_never_verify(self):
        executor = FakeExecutor()
        with patch("verification.generate_test_cases", return_value=InputGenerationResult("supported", "empty", ())):
            result = self.verify(executor)
        self.assertEqual(result.verification_status, "inconclusive")
        self.assertEqual(executor.calls, [])

    def test_input_generator_infrastructure_error_is_sanitized(self):
        executor = FakeExecutor()
        with patch("verification.generate_test_cases", side_effect=RuntimeError("private-generator-details")):
            result = self.verify(executor)
        self.assertEqual(result.verification_status, "error")
        self.assertEqual(result.tests_run, 0)
        self.assertEqual(executor.calls, [])
        self.assertNotIn("private-generator-details", json.dumps(result.to_dict()))

    def test_missing_executor_stays_not_verified(self):
        result = self.verify(None)
        self.assertEqual(result.verification_status, "inconclusive")
        self.assertEqual(result.tests_total, 6)
        self.assertEqual(result.tests_run, 0)
        self.assertEqual(VerificationResult().verification_status, "not_verified")

    def test_infrastructure_exception_is_sanitized_and_stops_suite(self):
        executor = FakeExecutor([RuntimeError("private-worker-details")])
        result = self.verify(executor)
        self.assertEqual(result.verification_status, "error")
        self.assertEqual((result.tests_total, result.tests_run, result.tests_errors), (6, 1, 1))
        self.assertEqual(len(executor.calls), 1)
        self.assertNotIn("private-worker-details", json.dumps(result.to_dict()))

    def test_explicit_infrastructure_error(self):
        result = self.verify(FakeExecutor([ExecutionResult("error"), ExecutionResult("returned", 0, post_args=[], post_kwargs={})]))
        self.assertEqual(result.verification_status, "error")
        self.assertEqual(result.tests_run, 1)

    def test_matching_unexpected_exceptions_do_not_verify(self):
        raised = ExecutionResult("raised", exception_type="builtins.ValueError", exception_message="bad", post_args=[], post_kwargs={})
        self.assertEqual(self.verify(FakeExecutor([raised] * 12)).verification_status, "inconclusive")

    def test_all_expected_exceptions_still_require_a_normal_return(self):
        raised = ExecutionResult("raised", exception_type="builtins.ValueError", exception_message="bad", post_args=[], post_kwargs={})
        policy = ComparisonPolicy(frozenset({"builtins.ValueError"}))
        result = self.verify(FakeExecutor([raised] * 12), policy=policy)
        self.assertEqual(result.tests_passed, 6)
        self.assertEqual(result.verification_status, "inconclusive")
        mixed = self.verify(FakeExecutor([raised, raised]), policy=policy)
        self.assertEqual(mixed.verification_status, "verified")

    def test_rejection_takes_priority_over_timeout(self):
        results = [ExecutionResult("returned", 0, post_args=[], post_kwargs={}), ExecutionResult("returned", 1, post_args=[], post_kwargs={}), ExecutionResult("timeout"), ExecutionResult("timeout")]
        result = self.verify(FakeExecutor(results))
        self.assertEqual(result.verification_status, "rejected")
        self.assertEqual((result.tests_failed, result.tests_inconclusive), (1, 1))

    def test_infrastructure_error_takes_priority_and_retains_mismatches(self):
        observations = [ExecutionResult("returned", 0, post_args=[], post_kwargs={}), ExecutionResult("returned", 1, post_args=[], post_kwargs={}), RuntimeError("worker failed")]
        result = self.verify(FakeExecutor(observations))
        self.assertEqual(result.verification_status, "error")
        self.assertEqual((result.tests_failed, result.tests_errors), (1, 1))
        self.assertEqual(len(result.mismatches), 1)

    def test_original_mutation_does_not_contaminate_candidate_inputs(self):
        class MutatingFake(FakeExecutor):
            def execute_function(self, code, name, args, kwargs):
                result = super().execute_function(code, name, args, kwargs)
                args[0].append(999)
                return verification.ExecutionResult("returned", result.return_value, post_args=deepcopy(args), post_kwargs=deepcopy(kwargs))

        executor = MutatingFake()
        source = "def transform(value: list[int]):\n    return 0"
        result = verify_candidate(source, source, "transform", executor)
        self.assertEqual(result.verification_status, "verified")
        for original_call, candidate_call in zip(executor.calls[::2], executor.calls[1::2]):
            self.assertEqual(original_call[2:], candidate_call[2:])
            self.assertNotIn(999, candidate_call[2][0])

    def test_invalid_or_mismatched_sources_do_not_execute(self):
        for candidate in ("not python!", "def renamed(value: int):\n    return value",
                          "def transform(other: int):\n    return other", "import os\n" + CANDIDATE):
            with self.subTest(candidate=candidate):
                executor = FakeExecutor()
                result = verify_candidate(ORIGINAL, candidate, "transform", executor)
                self.assertEqual(result.verification_status, "inconclusive")
                self.assertEqual(executor.calls, [])

    def test_verification_has_no_groq_or_fastapi_dependency_or_runtime_calls(self):
        with patch.dict(sys.modules, {"groq": None, "ai_optimizer": None, "fastapi": None}):
            self.assertEqual(self.verify(FakeExecutor()).verification_status, "verified")
        for module in (verification, verification_inputs):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, {"groq", "ai_optimizer", "fastapi", "main"})
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.assertNotIn(node.func.id, {"exec", "eval", "compile"})
                    if isinstance(node.func, ast.Attribute):
                        self.assertNotIn(node.func.attr, {"perf_counter", "timeit"})


if __name__ == "__main__":
    unittest.main()

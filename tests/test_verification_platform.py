"""Platform regression tests; no host execution of submitted Python."""
import ast
from copy import deepcopy
import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from verification_input_profile import resolve_input_profiles
from verification_inputs import generate_test_cases
from verification import ExecutionResult, VerificationResult, ComparisonPolicy, compare_results, verify_candidate
from docker_executor import DockerExecutor
from runtime.codec import dump_json, encode_value
from main import app

BUBBLE = '''def bubble_sort(items: list[int]):
    for i in range(len(items)):
        for j in range(len(items) - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return items
'''
SORTED = 'def bubble_sort(items: list[int]):\n    return sorted(items)\n'
INFERRED = 'def positive(items):\n    result = []\n    for item in items:\n        if item > 0:\n            result.append(item)\n    return result\n'
INFERRED_CANDIDATE = 'def positive(items):\n    return [item for item in items if item > 0]\n'

class InputProfileTests(unittest.TestCase):
    def profile(self, source):
        return resolve_input_profiles(ast.parse(source).body[0])[0]

    def test_annotation_has_priority(self):
        profile = self.profile('def f(items: list[int]): return items')
        self.assertEqual((profile.inferred_type, profile.source, profile.supported), ('list[int]', 'annotation', True))
        self.assertFalse(self.profile('def f(x: float): return x + 1').supported)

    def test_obvious_unannotated_collection(self):
        with patch('ai_optimizer.Groq', side_effect=AssertionError('No AI inference')):
            profile = self.profile(INFERRED)
        self.assertEqual((profile.inferred_type, profile.source), ('list[int]', 'ast_inference'))
        self.assertTrue(profile.evidence)
        self.assertEqual(len(generate_test_cases(ast.parse(INFERRED).body[0]).cases), 6)

    def test_scalar_and_string_evidence(self):
        self.assertEqual(self.profile('def f(x): return x + 1').inferred_type, 'int')
        self.assertEqual(self.profile('def f(x): return x.strip()').inferred_type, 'str')

    def test_ambiguity_and_conflicts_do_not_guess(self):
        for source in ['def f(x): return x', 'def f(x): return len(x)', 'def f(x):\n for item in x:\n  pass',
                       'def f(x): return x.strip() + (x + 1)', 'def f(x): return len(x) + (x + 1)', 'def f(x): return x * 2']:
            with self.subTest(source=source):
                self.assertFalse(self.profile(source).supported)

    def test_external_effects_do_not_execute(self):
        for body in ['return open("file").read()', 'return callback(x)', 'return global_items',
                     'import requests\n    return requests.get(x)', 'global cache\n    cache = x\n    return x',
                     'return x.custom()', 'return database.query(x)']:
            source = 'def f(x: int):\n    ' + body
            executor = Mock()
            result = verify_candidate(source, source, 'f', executor)
            self.assertEqual(result.verification_status, 'inconclusive')
            executor.execute_function.assert_not_called()

class MutationTests(unittest.TestCase):
    def observation(self, args, kwargs=None, value=None):
        return ExecutionResult('returned', value, post_args=args, post_kwargs=kwargs or {})

    def test_same_return_different_mutation_rejected(self):
        comparison = compare_results(self.observation([[1, 2]], value=[1, 2]), self.observation([[2, 1]], value=[1, 2]))
        self.assertEqual((comparison.status, comparison.kind), ('failed', 'input_mutation'))

    def test_same_mutation_and_keyword_state_match(self):
        left = self.observation([], {'items': [1, 2]}, 3)
        self.assertEqual(compare_results(left, deepcopy(left)).status, 'passed')
        self.assertEqual(compare_results(left, self.observation([], {'items': [2, 1]}, 3)).status, 'failed')

    def test_missing_or_unsupported_post_state_inconclusive(self):
        missing = ExecutionResult('returned', 1)
        self.assertEqual(compare_results(missing, missing).status, 'inconclusive')
        unsupported = self.observation([object()], value=1)
        self.assertEqual(compare_results(unsupported, unsupported).status, 'inconclusive')

    def test_exception_policy_retained_with_mutations(self):
        error = ExecutionResult('raised', exception_type='builtins.ValueError', exception_message='bad', post_args=[[1]], post_kwargs={})
        self.assertEqual(compare_results(error, error).status, 'inconclusive')
        policy = ComparisonPolicy(frozenset({'builtins.ValueError'}))
        self.assertEqual(compare_results(error, error, policy).status, 'passed')
        other = ExecutionResult('raised', exception_type='builtins.ValueError', exception_message='bad', post_args=[[2]], post_kwargs={})
        self.assertEqual(compare_results(error, other, policy).status, 'failed')

    def test_safe_transport_requires_both_post_fields(self):
        payload = {'status': 'returned', 'return_value': encode_value(1), 'post_args': encode_value([[1]]), 'post_kwargs': encode_value({})}
        self.assertEqual(DockerExecutor._observation(dump_json(payload)).post_args, [[1]])
        del payload['post_kwargs']
        self.assertEqual(DockerExecutor._observation(dump_json(payload)).status, 'error')

    def test_fresh_inputs_and_verified_evidence(self):
        observations = []
        class Runtime:
            def execute_function(self, source, name, args, kwargs):
                observations.append(deepcopy(args))
                args[0].append(99)
                return ExecutionResult('returned', 1, post_args=deepcopy(args), post_kwargs=deepcopy(kwargs))
        source = 'def f(items: list[int]): return items'
        result = verify_candidate(source, source, 'f', Runtime())
        self.assertEqual(result.verification_status, 'verified')
        self.assertEqual(result.mutation_checks_performed, 6)
        self.assertEqual(result.input_profile_source, 'annotation')
        self.assertEqual(result.comparison_scope, 'return_values_exceptions_and_input_mutations')
        self.assertEqual(observations[::2], observations[1::2])

class GatewayTests(unittest.TestCase):
    def test_trusted_candidate_does_not_require_behavioral_verification(self):
        with patch('main.verify_with_docker', side_effect=AssertionError('Trusted bypasses verification')) as verify, patch('ai_optimizer.Groq', side_effect=AssertionError('No Groq')), TestClient(app) as client:
            result = client.post('/optimize/replacement', json={'code': INFERRED, 'function_name': 'positive'}).json()
        self.assertEqual(result['optimization_source'], 'trusted_pattern')
        self.assertEqual(result['verification_status'], 'trusted_pattern')
        self.assertTrue(result['trusted_candidate_token'])
        verify.assert_not_called()

    def test_zero_cases_cannot_be_verified(self):
        with patch('main.verify_with_docker', return_value=VerificationResult(verification_status='verified')), TestClient(app) as client:
            result = client.post('/verify/candidate', json={'original_code': BUBBLE, 'candidate_function_code': SORTED, 'function_name': 'bubble_sort'}).json()
        self.assertEqual(result['verification_status'], 'inconclusive')
        self.assertNotIn('verified_full_code', result)

    def test_global_builtin_shadowing_is_inconclusive(self):
        source = 'sorted = custom_sort\n' + SORTED
        with patch('main.verify_with_docker') as runtime, TestClient(app) as client:
            result = client.post('/verify/candidate', json={'original_code': source, 'candidate_function_code': SORTED, 'function_name': 'bubble_sort'}).json()
            benchmark = client.post('/benchmark/candidate', json={'original_code': source, 'candidate_function_code': SORTED, 'function_name': 'bubble_sort'}).json()
        self.assertEqual(result['verification_status'], 'inconclusive')
        self.assertEqual(benchmark['benchmark_status'], 'inconclusive')
        runtime.assert_not_called()

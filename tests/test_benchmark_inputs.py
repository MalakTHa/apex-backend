"""Separate bounded performance inputs; no timing implementation changes."""
import ast
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from benchmark_inputs import generate_benchmark_cases, workload_sizes
from verification_inputs import generate_test_cases
from benchmark import BenchmarkConfig, benchmark_candidate
from test_benchmark import FakeRuntime
from test_verification_platform import BUBBLE
from main import app

class BenchmarkInputTests(unittest.TestCase):
    def test_large_list_workloads_are_separate_from_verification(self):
        node = ast.parse('def f(items: list[int]):\n    return sum(items)').body[0]
        small = generate_test_cases(node).cases
        large = generate_benchmark_cases(node).cases
        self.assertEqual(len(large), 6)
        self.assertLess(max(len(c.args[0]) for c in small), min(len(c.args[0]) for c in large))
        self.assertEqual(sorted({len(c.args[0]) for c in large}), [100, 1000, 5000])
        self.assertEqual(large, generate_benchmark_cases(node).cases)

    def test_complexity_budgets(self):
        self.assertEqual(workload_sizes('O(n²)'), (50, 150, 300))
        self.assertEqual(workload_sizes('O(n log n)'), (100, 1000, 3000))
        self.assertEqual(workload_sizes('O(2^n)'), (2, 4, 8))
        self.assertLess(max(workload_sizes('unknown')), max(workload_sizes('O(n)')))

    def test_comprehensions_and_recursion_have_safety_caps(self):
        for source, cap in [('def f(n: int): return sum(i+j for i in range(n) for j in range(n))', 300),
                            ('def f(n: int): return 1 if n < 2 else f(n-1)+f(n-2)', 8)]:
            cases = generate_benchmark_cases(ast.parse(source).body[0]).cases
            self.assertTrue(cases)
            self.assertLessEqual(max(c.args[0] for c in cases), cap)

    def test_bubble_unannotated_uses_bounded_nontrivial_lists(self):
        node = ast.parse(BUBBLE.replace(': list[int]', '')).body[0]
        sizes = sorted({len(c.args[0]) for c in generate_benchmark_cases(node).cases})
        self.assertEqual(sizes, [50, 150, 300])

    def test_trusted_does_not_execute_behavioral_checks(self):
        source = BUBBLE.replace(': list[int]', '')
        runtime = FakeRuntime(verification='error')
        with patch('benchmark.verify_candidate', side_effect=AssertionError('No trusted verification')):
            result = benchmark_candidate(source, source, 'bubble_sort', runtime, config=BenchmarkConfig(measurement_runs=3), optimization_source='trusted_pattern')
        self.assertEqual(result.benchmark_status, 'completed')
        self.assertFalse(any(c[0]=='execute' for c in runtime.calls))

    def test_trusted_token_binds_exact_sources(self):
        source = BUBBLE.replace(': list[int]', '')
        with TestClient(app) as client, patch('main.verify_with_docker', side_effect=AssertionError('No Verify')), patch('ai_optimizer.Groq', side_effect=AssertionError('No Groq')):
            result = client.post('/optimize/replacement', json={'code':source,'function_name':'bubble_sort'}).json()
            self.assertEqual(result['verification_status'], 'trusted_pattern')
            payload = {'original_code':source, 'candidate_function_code':result['optimized_function_code'], 'function_name':'bubble_sort',
                       'optimization_source':'trusted_pattern', 'trusted_candidate_token':result['trusted_candidate_token']}
            with patch('benchmark.benchmark_with_docker') as runtime:
                for key in ['original_code','candidate_function_code']:
                    response = client.post('/benchmark/candidate', json={**payload, key:payload[key]+'\n# edited'}).json()
                    self.assertEqual(response['benchmark_status'],'inconclusive')
                runtime.assert_not_called()

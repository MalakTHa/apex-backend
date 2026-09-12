"""Real Docker platform tests: source is never executed on the host."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from docker_executor import DockerExecutor
from main import app
from test_verification_platform import BUBBLE, SORTED, INFERRED, INFERRED_CANDIDATE

class PlatformDockerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DockerExecutor().availability().available:
            raise unittest.SkipTest('Docker unavailable')

    def verify(self, original, candidate, name):
        with patch('ai_optimizer.Groq', side_effect=AssertionError('No Groq')), TestClient(app) as client:
            return client.post('/verify/candidate', json={'original_code':original, 'candidate_function_code':candidate, 'function_name':name}).json()

    def test_bubble_in_place_vs_sorted_rejected(self):
        result = self.verify(BUBBLE, SORTED, 'bubble_sort')
        self.assertEqual(result['verification_status'], 'rejected', result)
        self.assertTrue(any(m['kind']=='input_mutation' for m in result['mismatches']))
        self.assertEqual(result['mutation_checks_performed'], 6)
        print('Docker bubble_sort vs sorted: rejected for observable input mutation.')

    def test_same_mutation_verified(self):
        candidate = 'def bubble_sort(items: list[int]):\n    items.sort()\n    return items\n'
        result = self.verify(BUBBLE, candidate, 'bubble_sort')
        self.assertEqual(result['verification_status'], 'verified', result)

    def test_unannotated_inference_and_trusted_gateway(self):
        result = self.verify(INFERRED, INFERRED_CANDIDATE, 'positive')
        self.assertEqual(result['verification_status'], 'verified', result)
        self.assertEqual(result['input_profile_source'], 'ast_inference')
        with patch('ai_optimizer.Groq', side_effect=AssertionError('No Groq')), TestClient(app) as client:
            trusted = client.post('/optimize/replacement', json={'code': INFERRED, 'function_name':'positive'}).json()
        self.assertEqual(trusted['optimization_source'], 'trusted_pattern')
        self.assertEqual(trusted['verification_status'], 'trusted_pattern', trusted)
        print('Docker unannotated positive(items): verified using conservative AST inference.')

    def test_unannotated_mutable_sequence_profiles(self):
        original = BUBBLE.replace(": list[int]", "")
        candidate = "def bubble_sort(items):\n    items.sort()\n    return items\n"
        result = self.verify(original, candidate, "bubble_sort")
        self.assertEqual(result["verification_status"], "verified", result)
        self.assertEqual(result["tests_run"], 6)
        self.assertEqual(result["mutation_checks_performed"], 6)
        result = self.verify(original, SORTED.replace(": list[int]", ""), "bubble_sort")
        self.assertEqual(result["verification_status"], "rejected", result)
        self.assertTrue(any(m["kind"] == "input_mutation" for m in result["mismatches"]))

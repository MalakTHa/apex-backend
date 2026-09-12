import ast
import hashlib
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main
from verification import VerificationResult, ExecutionResult, verify_candidate
from verification_cases import select_test_cases
from verification_inputs import generate_test_cases
from verification_service import interactive_case_limit

ORIGINAL = "def total(values: list[int]):\n    result = 0\n    for value in values:\n        result += value\n    return result\n"
CANDIDATE = "def total(values: list[int]):\n    return sum(values)\n"


class VerifyEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        patcher = patch("main.verify_with_docker", return_value=VerificationResult(
            verification_status="verified", tests_total=6, tests_available=6, tests_run=6, tests_passed=6))
        self.verify = patcher.start()
        self.addCleanup(patcher.stop)
        limits = patch("main.interactive_case_limit", return_value=12)
        limits.start()
        self.addCleanup(limits.stop)
        ai = patch("main.suggest_ai_optimization", side_effect=AssertionError("No AI during verification"))
        self.ai = ai.start()
        self.addCleanup(ai.stop)
        groq = patch("ai_optimizer.Groq", side_effect=AssertionError("No real Groq in tests"))
        self.groq = groq.start()
        self.addCleanup(groq.stop)

    def post(self, candidate=CANDIDATE, original=ORIGINAL, **extra):
        response = self.client.post("/verify/candidate", json={
            "original_code": original, "function_name": "total",
            "candidate_function_code": candidate, **extra,
        })
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_same_candidate_no_second_groq_and_exact_full_code(self):
        original = "CONSTANT = 3\n\n" + ORIGINAL + "\ndef other():\n    return 1\n"
        response = self.post(original=original)
        self.assertEqual(response["verification_status"], "verified")
        self.verify.assert_called_once_with(ORIGINAL.rstrip(), CANDIDATE, "total", case_limit=12)
        self.assertEqual(response["verified_full_code"], main.replace_function_code(original, "total", CANDIDATE))
        self.assertEqual(response["candidate_sha256"], hashlib.sha256(CANDIDATE.encode()).hexdigest())
        self.assertEqual(response["original_sha256"], hashlib.sha256(original.encode()).hexdigest())
        self.ai.assert_not_called()
        self.groq.assert_not_called()

    def test_analysis_is_recomputed_not_taken_from_client(self):
        response = self.post(before={"time_complexity": "invented"}, after={"time_complexity": "invented"})
        self.assertEqual(response["before"], main.analyze_function_node("total", ast.parse(ORIGINAL).body[0]))
        self.assertEqual(response["after"], main.analyze_function_node("total", ast.parse(CANDIDATE).body[0]))

    def test_nonverified_states_never_return_applicable_code(self):
        for status in ("rejected", "inconclusive", "error"):
            with self.subTest(status=status):
                self.verify.return_value = VerificationResult(verification_status=status, reason="private Docker command and path")
                response = self.post()
                self.assertEqual(response["verification_status"], status)
                self.assertNotIn("verified_full_code", response)
                self.assertNotIn("private Docker", str(response))
        self.assertEqual(response["reason"], "Verification runtime is currently unavailable.")

    def test_invalid_syntax_no_docker(self):
        self.assertEqual(self.post(candidate="def total(values: list[int]):\n    return (")["verification_status"], "rejected")
        self.verify.assert_not_called()

    def test_changed_name_no_docker(self):
        self.assertEqual(self.post(candidate=CANDIDATE.replace("total", "renamed"))["verification_status"], "rejected")
        self.verify.assert_not_called()

    def test_changed_parameters_and_api_no_docker(self):
        for candidate in (CANDIDATE.replace("values: list[int]", "data: list[int]"),
                          CANDIDATE.replace("values: list[int]", "values: list[int], other=0"),
                          CANDIDATE.replace("):", ") -> int:"), "@decorator\n" + CANDIDATE,
                          CANDIDATE.replace("values: list[int]", "*, values: list[int]")):
            with self.subTest(candidate=candidate):
                self.assertEqual(self.post(candidate=candidate)["verification_status"], "rejected")
        self.verify.assert_not_called()

    def test_multiple_or_async_candidate_functions_no_docker(self):
        for candidate in (CANDIDATE + "\ndef other():\n    pass", "async " + CANDIDATE):
            self.assertEqual(self.post(candidate=candidate)["verification_status"], "rejected")
        self.verify.assert_not_called()

    def test_unsupported_annotations_inconclusive_without_invocations(self):
        source = ORIGINAL.replace(": list[int]", "")
        candidate = CANDIDATE.replace(": list[int]", "")
        from verification_service import verify_with_docker
        with patch("main.verify_with_docker", wraps=verify_with_docker):
            with patch("verification_service.DockerExecutor") as executor:
                executor.return_value.pending_cleanup = ()
                response = self.post(candidate, source)
                self.assertEqual(response["verification_status"], "inconclusive")
                executor.return_value.execute_function.assert_not_called()

    def test_context_dependent_nested_function_is_inconclusive(self):
        nested = "def outer():\n" + "\n".join("    " + line for line in ORIGINAL.splitlines())
        self.assertEqual(self.post(original=nested)["verification_status"], "inconclusive")
        self.verify.assert_not_called()

    def test_exception_is_sanitized(self):
        self.verify.side_effect = RuntimeError("host-secret-details")
        response = self.post()
        self.assertEqual(response["verification_status"], "error")
        self.assertNotIn("host-secret", str(response))

    def test_optimization_does_not_automatically_verify(self):
        self.ai.side_effect = None
        self.ai.return_value = {"status": "candidate", "can_optimize": True,
                                "optimized_function_code": CANDIDATE, "reason": "Candidate"}
        result = self.client.post("/optimize/replacement", json={"code": ORIGINAL, "function_name": "total"}).json()
        self.assertEqual(result["verification_status"], "not_verified")
        self.verify.assert_not_called()


class InteractiveCaseLimitTests(unittest.TestCase):
    def cases(self):
        return generate_test_cases(ast.parse("def f(left: int, right: int):\n    return left + right").body[0]).cases

    def test_deterministic_limit_preserves_parameter_diversity(self):
        cases = self.cases()
        selected = select_test_cases(cases, 12)
        self.assertEqual(len(cases), 36)
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected, select_test_cases(cases, 12))
        for i in (0, 1):
            self.assertEqual({case.args[i] for case in selected}, {case.args[i] for case in cases})

    def test_limit_is_applied_to_executor_counts(self):
        from unittest.mock import Mock
        executor = Mock()
        executor.execute_function.return_value = ExecutionResult("returned", 1)
        source = "def f(left: int, right: int):\n    return left + right"
        result = verify_candidate(source, source, "f", executor, case_limit=12)
        self.assertEqual((result.tests_available, result.tests_total, result.tests_run), (36, 12, 12))
        self.assertEqual(executor.execute_function.call_count, 24)

    def test_env_default_override_and_invalid(self):
        with patch("verification_service.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(interactive_case_limit(), 12)
                os.environ["APEX_VERIFICATION_MAX_CASES"] = "4"
                self.assertEqual(interactive_case_limit(), 4)
                for bad in ("0", "-1", "many", "37"):
                    os.environ["APEX_VERIFICATION_MAX_CASES"] = bad
                    self.assertRaises(ValueError, interactive_case_limit)

    def test_invalid_limit_cannot_verify(self):
        for value in (0, -1, True, "12"):
            self.assertRaises(ValueError, select_test_cases, self.cases(), value)


if __name__ == "__main__":
    unittest.main()

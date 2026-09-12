"""Endpoint-to-Docker E2E. No Groq and no host execution of source."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from docker_executor import DockerExecutor
from main import app

ORIGINAL = "def total(values: list[int]):\n    result = 0\n    for value in values:\n        result += value\n    return result\n"


class VerifyDockerEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executor = DockerExecutor()
        available = executor.availability()
        cls.daemon_unavailable = available.error_code == "docker_unavailable"
        if not cls.daemon_unavailable:
            if not available.available:
                raise AssertionError("Docker daemon lacks the required isolation controls")
            if executor._image_id() is None:
                raise AssertionError("Prepare the versioned APEX runtime image before running E2E tests")

    def setUp(self):
        if self.daemon_unavailable:
            self.skipTest("Docker daemon is unavailable")
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def verify(self, expression):
        candidate = f"def total(values: list[int]):\n    return {expression}\n"
        with patch("ai_optimizer.Groq", side_effect=AssertionError("E2E must not call Groq")):
            response = self.client.post("/verify/candidate", json={
                "original_code": ORIGINAL, "function_name": "total", "candidate_function_code": candidate})
        self.assertEqual(response.status_code, 200)
        return response.json(), candidate

    def test_correct_candidate_verified(self):
        result, candidate = self.verify("sum(values)")
        self.assertEqual(result["verification_status"], "verified")
        self.assertEqual(result["tests_passed"], 6)
        self.assertEqual(result["verified_full_code"], candidate)
        print("Real Docker E2E: correct candidate -> verified (6/6).")

    def test_incorrect_candidate_rejected(self):
        result, _ = self.verify("sum(values) + 1")
        self.assertEqual(result["verification_status"], "rejected")
        self.assertEqual(result["tests_failed"], 6)
        self.assertNotIn("verified_full_code", result)
        print("Real Docker E2E: incorrect candidate -> rejected (6 failed).")


if __name__ == "__main__":
    unittest.main()

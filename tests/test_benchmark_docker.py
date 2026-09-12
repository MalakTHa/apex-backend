"""Real containers only. No live Groq, host execution, or host stopwatch."""
import json
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from benchmark import BENCHMARK_IMAGE
from docker_executor import DockerConfig, DockerExecutor
from main import app, _trusted_candidate_token

ORIGINAL = """def pair_total(value: int):
    n = 600 + abs(value)
    return sum(i + j for i in range(n) for j in range(n))
"""
CANDIDATE = """def pair_total(value: int):
    n = 600 + abs(value)
    return 2 * n * sum(range(n))
"""


class BenchmarkDockerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executor = DockerExecutor(DockerConfig(image=BENCHMARK_IMAGE))
        available = executor.availability()
        if available.error_code == "docker_unavailable":
            raise unittest.SkipTest("Docker daemon is unavailable")
        if not available.available or executor._image_id() is None:
            raise AssertionError("Prepare the 4A image and required Docker isolation controls")

    def setUp(self):
        self.executor = DockerExecutor(DockerConfig(image=BENCHMARK_IMAGE))

    def assert_clean(self):
        self.assertEqual(self.executor.pending_cleanup, ())
        if self.executor.last_container_name:
            reply = self.executor._call(["container", "ls", "--all", "--filter",
                "name=^/" + self.executor.last_container_name + "$", "--format", "{{.ID}}"])
            self.assertEqual(reply.returncode, 0)
            self.assertEqual(reply.stdout.strip(), b"")

    def test_real_endpoint_verified_pair_and_faster_median(self, optimization_source="ai"):
        with patch.dict("os.environ", {"APEX_BENCHMARK_WARMUP_RUNS": "1",
                        "APEX_BENCHMARK_MEASUREMENT_RUNS": "5", "APEX_BENCHMARK_MAX_CASES": "2"}), \
                patch("ai_optimizer.Groq", side_effect=AssertionError("No Groq")), \
                patch("docker_executor.DockerExecutor", return_value=self.executor), TestClient(app) as client:
            response = client.post("/benchmark/candidate", json={"original_code": ORIGINAL,
                "candidate_function_code": CANDIDATE, "function_name": "pair_total",
                "optimization_source": optimization_source,
                "trusted_candidate_token": _trusted_candidate_token(ORIGINAL, CANDIDATE, "pair_total")})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["verification_status"], "trusted_pattern" if optimization_source == "trusted_pattern" else "verified")
        self.assertEqual(result["benchmark_status"], "completed", result)
        self.assertEqual(result["cases_benchmarked"], 2)
        self.assertGreater(result["original"]["median_ns"], result["optimized"]["median_ns"])
        self.assertGreater(result["optimized"]["median_ns"], 0)
        print("Real Docker benchmark: " + json.dumps({key: result[key] for key in
              ("original", "optimized", "speedup", "warmup_runs", "measurement_runs", "cases_benchmarked")}))
        self.assert_clean()

    def test_trusted_pattern_endpoint_same_docker_runtime(self):
        with patch("main.ReplacementEngine", side_effect=AssertionError("No regeneration")):
            self.test_real_endpoint_verified_pair_and_faster_median("trusted_pattern")

    def test_timing_inside_worker_and_execute_contract_unchanged(self):
        code = """def f(value: int):
    import time
    time.sleep(0.002)
    return value
"""
        # Function loading is delayed deliberately outside the timed call. Defaults
        # are syntax accepted by this low-level worker, not a general endpoint input.
        slow_load = code.replace("value: int", "value: int = __import__('time').sleep(0.1)")
        result = self.executor.benchmark_function(slow_load, "f", [1], {}, warmup_runs=0, measurement_runs=3)
        self.assertEqual(result.status, "returned")
        self.assertEqual(len(result.return_value), 3)
        self.assertTrue(all(1_000_000 < value < 90_000_000 for value in result.return_value), result)
        executed = self.executor.execute_function(code, "f", [7], {})
        self.assertEqual((executed.status, executed.return_value), ("returned", 7))
        self.assert_clean()

    def test_fresh_nested_inputs_in_real_worker(self):
        code = """def f(values):
    if values != [{'x': [1]}]:
        raise ValueError('input reused')
    values[0]['x'].append(2)
    return 1
"""
        result = self.executor.benchmark_function(code, "f", [[{"x": [1]}]], {}, warmup_runs=3, measurement_runs=5)
        self.assertEqual(result.status, "returned")
        self.assertEqual(len(result.return_value), 5)
        self.assert_clean()

    def test_isolation_network_secret_readonly_and_privileges(self):
        code = """def f(value: int):
    import os
    import socket
    assert os.getuid() == 65534
    assert 'GROQ_API_KEY' not in os.environ
    with open('/proc/self/status') as stream:
        status = stream.read()
    assert 'NoNewPrivs:\\t1' in status
    assert 'Seccomp:\\t2' in status
    assert 'CapEff:\\t0000000000000000' in status
    try:
        with open('/tmp/apex-benchmark-write', 'w') as stream:
            stream.write('x')
    except OSError:
        pass
    else:
        raise AssertionError('writable filesystem')
    connection = socket.socket()
    connection.settimeout(0.05)
    try:
        connection.connect(('1.1.1.1', 53))
    except OSError:
        pass
    else:
        raise AssertionError('network available')
    finally:
        connection.close()
    return value
"""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-only-not-a-real-secret"}):
            result = self.executor.benchmark_function(code, "f", [1], {}, warmup_runs=0, measurement_runs=3)
        self.assertEqual(result.status, "returned", result)
        self.assert_clean()

    def test_timeout_cleanup(self):
        self.executor = DockerExecutor(DockerConfig(image=BENCHMARK_IMAGE, invocation_timeout=1))
        result = self.executor.benchmark_function("def f(x):\n    while True: pass", "f", [1], {}, measurement_runs=3)
        self.assertEqual(result.status, "timeout")
        self.assert_clean()

    def test_exception_excludes_all_samples(self):
        result = self.executor.benchmark_function("def f(x):\n    raise ValueError('example')", "f", [1], {}, measurement_runs=3)
        self.assertEqual(result.status, "raised")
        self.assertIsNone(result.return_value)
        self.assert_clean()


    def test_trusted_bubble_to_merge_real_performance(self):
        from test_verification_platform import BUBBLE
        source = BUBBLE.replace(': list[int]', '')
        with patch.dict('os.environ', {'APEX_BENCHMARK_WARMUP_RUNS':'1', 'APEX_BENCHMARK_MEASUREMENT_RUNS':'5', 'APEX_BENCHMARK_MAX_CASES':'6'}), patch('main.verify_with_docker', side_effect=AssertionError('No trusted verification')), patch('ai_optimizer.Groq', side_effect=AssertionError('No Groq')), TestClient(app) as client:
            optimized = client.post('/optimize/replacement', json={'code':source,'function_name':'bubble_sort'}).json()
            with patch('main.ReplacementEngine', side_effect=AssertionError('No regeneration')), patch('benchmark.verify_candidate', side_effect=AssertionError('No trusted verification')):
                result = client.post('/benchmark/candidate', json={'original_code':source,'candidate_function_code':optimized['optimized_function_code'],
                    'function_name':'bubble_sort','optimization_source':'trusted_pattern','trusted_candidate_token':optimized['trusted_candidate_token']}).json()
        self.assertEqual(result['benchmark_status'],'completed',result)
        self.assertEqual(result['verification_status'],'trusted_pattern')
        self.assertEqual(result['cases_benchmarked'],6)
        self.assertGreater(result['speedup'],1)
        print('Trusted bubble -> merge performance: '+json.dumps({key:result[key] for key in ('original','optimized','speedup','cases_benchmarked')}))


if __name__ == "__main__":
    unittest.main()

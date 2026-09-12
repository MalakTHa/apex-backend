import ast
from copy import deepcopy
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from benchmark import BenchmarkConfig, benchmark_candidate, speedup, summarize
from main import app
from runtime.codec import decode_value, encode_value
from runtime.measurement import measure
from verification import ExecutionResult

SOURCE = "def total(values: list[int]):\n    return sum(values)\n"
CANDIDATE = "def total(values: list[int]):\n    return sum(values) + 0\n"


class FakeRuntime:
    def __init__(self, original=None, optimized=None, status=None, verification=None):
        self.original = [6000, 3000, 9000] if original is None else original
        self.optimized = [3000, 1500, 4500] if optimized is None else optimized
        self.status = status
        self.verification = verification
        self.calls = []
        self.measurements = 0

    def execute_function(self, source, name, args, kwargs):
        self.calls.append(("execute", source, args, kwargs))
        if self.verification:
            return ExecutionResult(self.verification)
        return ExecutionResult("returned", 1, post_args=deepcopy(args), post_kwargs=deepcopy(kwargs))

    def benchmark_function(self, source, name, args, kwargs, **options):
        self.calls.append(("benchmark", source, args, kwargs))
        self.measurements += 1
        if self.status and self.measurements <= 2:
            return ExecutionResult(self.status)
        return ExecutionResult("returned", self.original if source == SOURCE else self.optimized)


class BenchmarkTests(unittest.TestCase):
    def run_benchmark(self, runtime=None, **config):
        return benchmark_candidate(SOURCE, CANDIDATE, "total", runtime or FakeRuntime(),
                                   config=BenchmarkConfig(measurement_runs=3, **config))

    def test_statistics(self):
        self.assertEqual(summarize([9, 1, 4, 2]), {"median_ns": 3, "mean_ns": 4, "min_ns": 1, "max_ns": 9})

    def test_faster_and_aggregate(self):
        result = self.run_benchmark()
        self.assertEqual(result.benchmark_status, "completed")
        self.assertEqual(result.original["median_ns"], 6000)
        self.assertEqual(result.speedup, 2)
        self.assertEqual(result.cases_benchmarked, 6)

    def test_slower(self):
        self.assertEqual(self.run_benchmark(FakeRuntime(optimized=[12000]*3)).speedup, .5)

    def test_invalid_samples(self):
        for values in ([], [0]*3, [-1]*3, [float("nan")]*3, [True]*3, [10], [31_000_000_000]*3):
            with self.subTest(values=values):
                result = self.run_benchmark(FakeRuntime(original=values))
                self.assertEqual(result.benchmark_status, "error")
                self.assertIsNone(result.speedup)

    def test_tiny_samples_no_speed_claim(self):
        result = self.run_benchmark(FakeRuntime(original=[1]*3, optimized=[2]*3))
        self.assertEqual(result.benchmark_status, "completed")
        self.assertIsNone(result.speedup)
        for value in (0, -1, float("inf"), float("nan"), True):
            self.assertIsNone(speedup(value, 10000))

    def test_fresh_nested_inputs_and_timing_boundaries(self):
        wire = encode_value([[{"nested": [1]}]])
        events, retained = [], []
        def fresh():
            events.append("decode")
            return decode_value(wire), {}
        def function(value):
            events.append("call")
            self.assertEqual(value, [{"nested": [1]}])
            retained.append(value)
            value[0]["nested"].append(2)
        ticks = iter([100, 110, 200, 220, 300, 330])
        def clock():
            events.append("clock")
            return next(ticks)
        self.assertEqual(measure(function, fresh, 2, 3, clock), [10, 20, 30])
        self.assertEqual(events, ["decode", "call"]*2 + ["decode", "clock", "call", "clock"]*3)
        self.assertEqual(len({id(value) for value in retained}), 5)

    def test_deterministic_limit_and_same_inputs(self):
        first, second = FakeRuntime(), FakeRuntime()
        a, b = self.run_benchmark(first, max_cases=3), self.run_benchmark(second, max_cases=3)
        self.assertEqual(a.case_results, b.case_results)
        self.assertEqual(a.cases_total, 3)
        calls = [call for call in first.calls if call[0] == "benchmark"]
        for index in range(0, len(calls), 2):
            self.assertEqual(calls[index][2:], calls[index+1][2:])
        self.assertEqual(len([call for call in first.calls if call[0] == "execute"]), 12)

    def test_timeout_excludes_pair(self):
        result = self.run_benchmark(FakeRuntime(status="timeout"))
        self.assertEqual(result.benchmark_status, "inconclusive")
        self.assertEqual(result.cases_benchmarked, 5)
        self.assertIsNone(result.speedup)

    def test_exception_excludes_pair_without_changing_verification(self):
        result = self.run_benchmark(FakeRuntime(status="raised"))
        self.assertEqual(result.verification_status, "verified")
        self.assertEqual(result.cases_benchmarked, 5)
        self.assertIsNone(result.case_results[0]["original_median_ns"])

    def test_infrastructure_error_sanitized(self):
        result = self.run_benchmark(FakeRuntime(status="error"))
        self.assertEqual(result.benchmark_status, "error")
        self.assertIsNone(result.speedup)

    def test_failed_verification_never_benchmarks(self):
        runtime = FakeRuntime(verification="timeout")
        result = self.run_benchmark(runtime)
        self.assertEqual(result.benchmark_status, "inconclusive")
        self.assertFalse(any(call[0] == "benchmark" for call in runtime.calls))

    def test_unsupported_inputs(self):
        runtime = FakeRuntime()
        result = benchmark_candidate("def f(x): return x", "def f(x): return x", "f", runtime)
        self.assertEqual(result.benchmark_status, "inconclusive")
        self.assertEqual(runtime.calls, [])

    def test_rejected_candidate_never_measured(self):
        runtime = FakeRuntime()
        def execution(source, *args):
            return ExecutionResult("returned", 1 if source == SOURCE else 2)
        runtime.execute_function = execution
        result = self.run_benchmark(runtime)
        self.assertEqual(result.verification_status, "rejected")
        self.assertIsNone(result.speedup)
        self.assertEqual(runtime.calls, [])

    def test_empty_cases_inconclusive(self):
        from verification_inputs import InputGenerationResult
        with patch("verification.generate_test_cases", return_value=InputGenerationResult("supported", "empty")):
            result = self.run_benchmark()
        self.assertEqual(result.benchmark_status, "inconclusive")
        self.assertEqual(result.cases_benchmarked, 0)

    def test_configuration_bounds(self):
        for options in ({"warmup_runs": -1}, {"measurement_runs": 1}, {"max_cases": 37}, {"max_cases": True}):
            self.assertRaises(ValueError, BenchmarkConfig, **options)
        with patch.dict("os.environ", {"APEX_BENCHMARK_WARMUP_RUNS": "4",
                        "APEX_BENCHMARK_MEASUREMENT_RUNS": "9", "APEX_BENCHMARK_MAX_CASES": "2"}):
            self.assertEqual(BenchmarkConfig.from_environment(), BenchmarkConfig(4, 9, 2))


class BenchmarkEndpointTests(unittest.TestCase):
    def request(self, **overrides):
        payload = dict(original_code=SOURCE, candidate_function_code=CANDIDATE, function_name="total")
        payload.update(overrides)
        with TestClient(app) as client:
            return client.post("/benchmark/candidate", json=payload)

    def test_forged_verified_field_does_not_grant_trust_or_call_groq(self):
        runtime = FakeRuntime(verification="timeout")
        with patch("docker_executor.DockerExecutor", return_value=runtime), patch("ai_optimizer.Groq") as groq:
            runtime.pending_cleanup = ()
            result = self.request(verification_status="verified").json()
        self.assertEqual(result["verification_status"], "inconclusive")
        self.assertFalse(any(call[0] == "benchmark" for call in runtime.calls))
        groq.assert_not_called()

    def test_invalid_sources_api_and_fingerprint_do_not_execute(self):
        for overrides in ({"candidate_function_code": "broken"}, {"candidate_function_code": "def other(values: list[int]): return 1"},
                          {"candidate_function_code": "def total(value: int): return 1"}, {"candidate_fingerprint": "0"*64}):
            with patch("benchmark.benchmark_with_docker") as service:
                self.assertEqual(self.request(**overrides).json()["benchmark_status"], "inconclusive")
                service.assert_not_called()

    def test_exact_sources_forwarded(self):
        with patch("benchmark.benchmark_with_docker", return_value=BenchmarkTests().run_benchmark()) as service:
            result = self.request().json()
        self.assertEqual(result["benchmark_status"], "completed")
        args = service.call_args.args
        self.assertEqual(args[0].rstrip(), SOURCE.rstrip())
        self.assertEqual(args[1:], (CANDIDATE, "total"))

    def test_unexpected_error_sanitized(self):
        with patch("benchmark.benchmark_with_docker", side_effect=RuntimeError("secret docker host")):
            result = self.request().json()
        self.assertEqual(result["benchmark_status"], "error")
        self.assertNotIn("secret", str(result))


class BenchmarkTransportTests(unittest.TestCase):
    def test_contract_and_pinned_image_same_for_verification_and_benchmark(self):
        import test_docker_executor as fixture
        from runtime.codec import dump_json, load_json, encode_value
        mocked = fixture.DockerExecutorTests()
        mocked.setUp()
        self.addCleanup(mocked.doCleanups)
        mocked.execute()
        mocked.reply = fixture.docker.CLIResult(stdout=dump_json({"status": "returned", "return_value": encode_value([2000]*3)}))
        result = mocked.executor.benchmark_function(fixture.SOURCE, "f", [1], {}, warmup_runs=2, measurement_runs=3)
        self.assertEqual(result.return_value, [2000]*3)
        requests = [load_json(options["input_data"]) for argv, options in mocked.calls if argv[0] == "start"]
        self.assertNotIn("operation", requests[0])
        self.assertEqual(requests[1]["operation"], "benchmark")
        self.assertEqual((requests[1]["warmup_runs"], requests[1]["measurement_runs"]), (2, 3))
        self.assertEqual(len([argv for argv, _ in mocked.calls if argv[:2] == ["image", "inspect"]]), 1)
        created = [argv for argv, _ in mocked.calls if argv[0] == "create"]
        self.assertTrue(all(fixture.IMAGE_ID in argv and "--read-only" in argv and "--network" in argv for argv in created))

    def test_invalid_repetitions_never_start_runtime(self):
        from docker_executor import DockerExecutor
        with patch("docker_executor._run_docker") as cli:
            result = DockerExecutor().benchmark_function(SOURCE, "total", [[]], {}, measurement_runs=1)
        self.assertEqual(result.status, "error")
        cli.assert_not_called()


if __name__ == "__main__":
    unittest.main()

class TrustedPatternBenchmarkTests(BenchmarkEndpointTests):
    def request(self, **overrides):
        from main import _trusted_candidate_token
        payload = {"optimization_source": "trusted_pattern", **overrides}
        payload.setdefault("trusted_candidate_token", _trusted_candidate_token(payload.get("original_code", SOURCE), payload.get("candidate_function_code", CANDIDATE), payload.get("function_name", "total")))
        return super().request(**payload)

    def test_forged_verified_field_does_not_grant_trust_or_call_groq(self):
        with patch("benchmark.benchmark_with_docker") as runtime:
            result = self.request(trusted_candidate_token="0"*64, verification_status="verified").json()
        self.assertEqual(result["benchmark_status"], "inconclusive")
        runtime.assert_not_called()

    def test_no_verification_prerequisite_or_regeneration(self):
        runtime = FakeRuntime()
        runtime.pending_cleanup = ()
        with patch.dict("os.environ", {"APEX_BENCHMARK_MEASUREMENT_RUNS": "3"}), patch("docker_executor.DockerExecutor", return_value=runtime), patch("main.ReplacementEngine", side_effect=AssertionError("No regeneration")), patch("main.verify_with_docker", side_effect=AssertionError("No verification endpoint")), patch("ai_optimizer.Groq", side_effect=AssertionError("No Groq")):
            result = self.request().json()
        self.assertEqual(result["benchmark_status"], "completed")
        self.assertEqual({call[1] for call in runtime.calls if call[0] == "benchmark"}, {SOURCE.rstrip(), CANDIDATE})

    def test_unknown_source_is_rejected(self):
        self.assertEqual(self.request(optimization_source="simple").status_code, 422)

    def test_runtime_failure_statuses(self):
        for status, expected in [("timeout", "inconclusive"), ("error", "error")]:
            runtime = FakeRuntime(status=status)
            runtime.pending_cleanup = ()
            with patch.dict("os.environ", {"APEX_BENCHMARK_MEASUREMENT_RUNS": "3"}), patch("docker_executor.DockerExecutor", return_value=runtime):
                self.assertEqual(self.request().json()["benchmark_status"], expected)

    def test_existing_trusted_result_is_forwarded_without_replacement(self):
        from benchmark import BenchmarkResult
        source = "def f(values):\n    result = []\n    for item in values:\n        if item > 0:\n            result.append(item)\n    return result"
        with TestClient(app) as client, patch("ai_optimizer.Groq", side_effect=AssertionError("No Groq")):
            optimized = client.post("/optimize/replacement", json={"code": source, "function_name": "f"}).json()
            self.assertTrue(optimized["pattern_applied"])
            self.assertEqual(optimized["optimization_source"], "trusted_pattern")
            with patch("main.ReplacementEngine", side_effect=AssertionError("No regeneration")), patch("benchmark.benchmark_with_docker", return_value=BenchmarkResult("f")) as runtime:
                client.post("/benchmark/candidate", json={"original_code": source, "function_name": "f", "optimization_source": "trusted_pattern", "candidate_function_code": optimized["optimized_function_code"], "trusted_candidate_token": optimized["trusted_candidate_token"]})
            self.assertEqual(runtime.call_args.args, (source, optimized["optimized_function_code"], "f"))

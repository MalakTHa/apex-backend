"""Finite-case benchmark orchestration. No host execution or host timing."""
import ast
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Protocol

from dotenv import load_dotenv
from verification import ExecutionResult, FunctionExecutor, verify_candidate
from verification_cases import select_test_cases
from benchmark_inputs import generate_benchmark_cases

BENCHMARK_IMAGE = "apex-verification:4a-python3.13.5"
MIN_RELIABLE_NS = 1000


class BenchmarkExecutor(FunctionExecutor, Protocol):
    def benchmark_function(self, function_code: str, function_name: str, args: list, kwargs: dict,
                           *, warmup_runs: int, measurement_runs: int) -> ExecutionResult:
        """returned carries a bounded list of integer nanosecond samples."""
        ...


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_runs: int = 3
    measurement_runs: int = 15
    max_cases: int = 6

    def __post_init__(self):
        for value, low, high in ((self.warmup_runs, 0, 20),
                                 (self.measurement_runs, 3, 100), (self.max_cases, 1, 36)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid benchmark configuration")

    @classmethod
    def from_environment(cls):
        load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
        return cls(*(int(os.getenv(key, default)) for key, default in (
            ("APEX_BENCHMARK_WARMUP_RUNS", "3"),
            ("APEX_BENCHMARK_MEASUREMENT_RUNS", "15"),
            ("APEX_BENCHMARK_MAX_CASES", "6"))))


@dataclass
class BenchmarkResult:
    function: str
    benchmark_status: str = "inconclusive"
    cases_total: int = 0
    cases_benchmarked: int = 0
    warmup_runs: int = 3
    measurement_runs: int = 15
    original: dict | None = None
    optimized: dict | None = None
    speedup: float | None = None
    case_results: list = field(default_factory=list)
    reason: str = "No suitable normal-return cases were measured."
    verification_status: str = "not_verified"
    candidate_fingerprint: str = ""
    original_fingerprint: str = ""
    aggregation: str = "median of per-case medians; mean/min/max over per-case medians"

    def to_dict(self):
        return asdict(self)


def summarize(samples, expected=None):
    if (type(samples) is not list or not samples or (expected is not None and len(samples) != expected)
            or any(type(value) is not int or not 0 < value <= 30_000_000_000 for value in samples)):
        raise ValueError("Invalid measurements")
    return {"median_ns": median(samples), "mean_ns": mean(samples),
            "min_ns": min(samples), "max_ns": max(samples)}


def speedup(original, optimized):
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and value >= MIN_RELIABLE_NS for value in (original, optimized)):
        return None
    return original / optimized


def input_summary(case):
    def describe(value):
        return {"type": type(value).__name__, **({"length": len(value)} if isinstance(value, (list, dict, str)) else {})}
    return {"args": [describe(value) for value in case.args],
            "kwargs": {key[:64]: describe(value) for key, value in case.kwargs.items()}}


def benchmark_candidate(original_code, candidate_code, function_name, executor: BenchmarkExecutor, *, config=None, optimization_source="ai"):
    """AI requires fresh verification; trusted patterns use their direct path.

    AI verification uses small cases; measurement uses separate bounded performance workloads.
    Exceptions/timeouts during measurement exclude the pair and make the job
    inconclusive. Infrastructure/invalid samples make it error. Partial aggregates
    are diagnostic only; overall speedup is emitted only for a completed suite.
    """
    config = config or BenchmarkConfig()
    result = BenchmarkResult(function_name, warmup_runs=config.warmup_runs,
                             measurement_runs=config.measurement_runs)
    result.candidate_fingerprint = hashlib.sha256(candidate_code.encode()).hexdigest()
    result.original_fingerprint = hashlib.sha256(original_code.encode()).hexdigest()
    try:
        if optimization_source == "ai":
            verification = verify_candidate(original_code, candidate_code, function_name, executor)
            result.verification_status = verification.verification_status
            if verification.verification_status != "verified":
                result.benchmark_status = "error" if verification.verification_status == "error" else "inconclusive"
                result.reason = "Fresh server-side behavioral verification did not succeed; benchmark was not started."
                return result
        elif optimization_source == "trusted_pattern":
            result.verification_status = "trusted_pattern"
        else:
            result.reason = "Unsupported optimization source."
            return result
        generation = generate_benchmark_cases(ast.parse(original_code).body[0])
        cases = select_test_cases(generation.cases, config.max_cases)
        result.cases_total = len(cases)
        original_medians, optimized_medians = [], []
        failed = False
        for index, case in enumerate(cases):
            row = {"case_index": index, "case_id": case.case_id, "input_summary": input_summary(case),
                   "original_median_ns": None, "optimized_median_ns": None, "speedup": None,
                   "status": "inconclusive"}
            result.case_results.append(row)
            sources = (original_code, candidate_code) if index % 2 == 0 else (candidate_code, original_code)
            observations = [executor.benchmark_function(source, function_name, deepcopy(case.args), deepcopy(case.kwargs),
                            warmup_runs=config.warmup_runs, measurement_runs=config.measurement_runs) for source in sources]
            if index % 2:
                observations.reverse()
            if any(item.status == "error" for item in observations):
                row["status"] = "error"
                result.benchmark_status = "error"
                result.reason = "Benchmark runtime is currently unavailable."
                return result
            if any(item.status != "returned" for item in observations):
                failed = True
                row["reason"] = "Timeout, exception or unsupported runtime observation; pair excluded."
                continue
            try:
                left, right = [summarize(item.return_value, config.measurement_runs) for item in observations]
            except ValueError:
                row["status"] = "error"
                result.benchmark_status = "error"
                result.reason = "The runtime returned invalid measurement samples."
                return result
            row.update(status="completed", original=left, optimized=right,
                       original_median_ns=left["median_ns"], optimized_median_ns=right["median_ns"],
                       speedup=speedup(left["median_ns"], right["median_ns"]))
            original_medians.append(left["median_ns"])
            optimized_medians.append(right["median_ns"])
            result.cases_benchmarked += 1
        if result.cases_benchmarked:
            # Medians can be half-integers for even repetition counts.
            def aggregate(values):
                return {"median_ns": median(values), "mean_ns": mean(values), "min_ns": min(values), "max_ns": max(values)}
            result.original, result.optimized = aggregate(original_medians), aggregate(optimized_medians)
            if not failed:
                result.benchmark_status = "completed"
                result.speedup = speedup(result.original["median_ns"], result.optimized["median_ns"])
                result.reason = "Measured function calls inside the isolated runtime; results depend on this machine and runtime."
            else:
                result.reason = "Some cases could not be measured normally; partial timings do not establish an overall speedup."
        return result
    except Exception:
        result.benchmark_status = "error"
        result.speedup = None
        result.reason = "Benchmark runtime is currently unavailable."
        return result


def benchmark_with_docker(original_code, candidate_code, function_name, *, optimization_source="ai"):
    from docker_executor import DockerConfig, DockerExecutor
    executor = DockerExecutor(DockerConfig(image=BENCHMARK_IMAGE))
    result = benchmark_candidate(original_code, candidate_code, function_name, executor,
                                 config=BenchmarkConfig.from_environment(), optimization_source=optimization_source)
    if executor.pending_cleanup and not executor.retry_cleanup():
        result.benchmark_status = "error"
        result.speedup = None
        result.reason = "Benchmark runtime cleanup could not be confirmed."
    return result

"""Behavioral comparison orchestration only; runtime adapters live separately.

No FastAPI/Groq dependency, execution, timing, or benchmarking lives here.
Callers may inject the Docker adapter; unit tests supply fake observations.
"""

import ast
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Literal, Protocol

from verification_inputs import generate_test_cases
from verification_input_profile import resolve_input_profiles, unsupported_effects
from verification_cases import select_test_cases

VerificationStatus = Literal["not_verified", "verified", "rejected", "inconclusive", "error"]
ExecutionStatus = Literal["returned", "raised", "timeout", "serialization_error", "error"]
ComparisonStatus = Literal["passed", "failed", "inconclusive", "error"]


@dataclass(frozen=True)
class ExecutionResult:
    """Observation from an isolated worker or test fake, with no performance metrics.

    return_value must be a losslessly decoded supported value (see compare_results).
    Exception types are fully qualified, e.g. builtins.ValueError. The executor
    must distinguish user exceptions from worker/transport/infrastructure errors.
    Timeout enforcement and serialization happen in that executor, not here.
    """
    status: ExecutionStatus
    return_value: object = None
    exception_type: str | None = None
    exception_message: str | None = None
    error_code: str | None = None
    post_args: list | None = None
    post_kwargs: dict | None = None


class FunctionExecutor(Protocol):
    def execute_function(
        self, function_code: str, function_name: str, args: list, kwargs: dict,
    ) -> ExecutionResult:
        """Run in isolation with a fresh environment; never in the API process.

        This protocol is not an isolation boundary. No concrete executor is
        provided here. Implementations own resource limits, lifecycle and IPC.
        """
        ...


@dataclass(frozen=True)
class ComparisonPolicy:
    # Empty by default: matching unexpected exceptions are inconclusive.
    expected_exception_types: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CaseComparison:
    status: ComparisonStatus
    kind: str
    reason: str
    case_id: str = ""


@dataclass
class VerificationResult:
    verification_status: VerificationStatus = "not_verified"
    tests_total: int = 0
    tests_available: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_inconclusive: int = 0
    tests_errors: int = 0
    reason: str = "No execution observations have been compared."
    mismatches: list[CaseComparison] = field(default_factory=list)
    issues: list[CaseComparison] = field(default_factory=list)
    input_profile_source: str | None = None
    input_profiles: list = field(default_factory=list)
    mutation_checks_performed: int = 0
    comparison_scope: str = "return_values_exceptions_and_input_mutations"

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical(value, depth=0, budget=None):
    """Type-sensitive comparison without invoking arbitrary object equality.

    Support None, bool, int, finite float, str, lists and string-keyed dicts.
    List order matters; dictionary key insertion order does not. Tuples, sets,
    custom objects, non-finite floats, cycles and oversized values are unsupported.
    """
    budget = [10000] if budget is None else budget
    budget[0] -= 1
    if depth > 32 or budget[0] < 0:
        raise ValueError("Unsupported value size or depth")
    value_type = type(value)
    if value_type in (type(None), bool, int, str):
        return (value_type.__name__, value)
    if value_type is float and math.isfinite(value):
        return ("float", value)
    if value_type is list:
        return ("list", tuple(_canonical(item, depth + 1, budget) for item in value))
    if value_type is dict and all(type(key) is str for key in value):
        return ("dict", tuple((key, _canonical(value[key], depth + 1, budget)) for key in sorted(value)))
    raise ValueError("Unsupported serialized value")



def _compare_mutations(original, candidate):
    if any(type(item.post_args) is not list or type(item.post_kwargs) is not dict for item in (original, candidate)):
        return CaseComparison("inconclusive", "missing_mutation_state", "The runtime did not provide reliable post-call input state.")
    try:
        if (_canonical(original.post_args), _canonical(original.post_kwargs)) != (_canonical(candidate.post_args), _canonical(candidate.post_kwargs)):
            return CaseComparison("failed", "input_mutation", "Observable input mutations differ.")
    except (ValueError, RecursionError):
        return CaseComparison("inconclusive", "input_mutation", "Post-call inputs are outside the supported comparison format.")
    return None


def compare_results(
    original: ExecutionResult, candidate: ExecutionResult,
    policy: ComparisonPolicy | None = None,
) -> CaseComparison:
    """Compare observations only; same exception is not automatically a pass."""
    policy = policy or ComparisonPolicy()
    allowed = {"returned", "raised", "timeout", "serialization_error", "error"}
    for observation in (original, candidate):
        if (
            not isinstance(observation, ExecutionResult)
            or type(observation.status) is not str or observation.status not in allowed
        ):
            return CaseComparison("error", "executor_contract", "Executor returned an invalid observation.")
        if observation.status == "raised" and (
            type(observation.exception_type) is not str or not observation.exception_type
            or type(observation.exception_message) is not str
        ):
            return CaseComparison("error", "executor_contract", "Exception metadata is missing or invalid.")
        if (
            observation.status != "raised"
            and (observation.exception_type is not None or observation.exception_message is not None)
        ) or (observation.status != "returned" and observation.return_value is not None):
            return CaseComparison("error", "executor_contract", "Observation fields contradict its execution status.")
    statuses = {original.status, candidate.status}
    if "error" in statuses:
        known_errors = {"docker_unavailable", "docker_isolation_unavailable", "docker_image_unavailable",
                        "docker_create_failed", "docker_execution_failed", "docker_cleanup_failed",
                        "invalid_worker_response", "invalid_request", "invalid_function_code",
                        "function_not_found", "worker_environment", "output_limit"}
        code = next((item.error_code for item in (original, candidate)
                     if item.status == "error" and type(item.error_code) is str and item.error_code in known_errors), None)
        reason = "Executor infrastructure failed." if code is None else f"Executor infrastructure failed: {code}."
        return CaseComparison("error", "infrastructure_error", reason)
    if "timeout" in statuses:
        return CaseComparison("inconclusive", "timeout", "At least one execution timed out; equivalence is unknown.")
    if "serialization_error" in statuses:
        return CaseComparison("inconclusive", "serialization_failure", "At least one result could not be serialized.")
    if original.status != candidate.status:
        return CaseComparison("failed", "exception_behavior", "One function returned while the other raised an exception.")
    if original.status == "raised":
        if (original.exception_type, original.exception_message) != (candidate.exception_type, candidate.exception_message):
            return CaseComparison("failed", "exception_behavior", "Exception types or messages differ.")
        if original.exception_type not in policy.expected_exception_types:
            return CaseComparison("inconclusive", "unexpected_exception", "Matching exceptions were not declared expected by the comparison policy.")
        mutation = _compare_mutations(original, candidate)
        if mutation:
            return mutation
        return CaseComparison("passed", "expected_exception", "Expected exception type and message match exactly.")
    try:
        original_value = _canonical(original.return_value)
        candidate_value = _canonical(candidate.return_value)
    except (ValueError, RecursionError):
        return CaseComparison("inconclusive", "serialization_failure", "Return values are outside the supported comparison format.")
    if original_value != candidate_value:
        return CaseComparison("failed", "return_value", "Return values differ in type, content, or list ordering.")
    mutation = _compare_mutations(original, candidate)
    if mutation:
        return mutation
    return CaseComparison("passed", "return_value", "Supported return values match.")


def verify_candidate(
    original_function_code: str, candidate_function_code: str, function_name: str,
    executor: FunctionExecutor | None = None, *, policy: ComparisonPolicy | None = None,
    case_limit: int | None = None,
) -> VerificationResult:
    """Compare a bounded suite using an injected executor, never Python execution.

    verified means only that this finite suite matched under the documented
    comparison scope, not universal correctness. At least one normal return
    must match. No executor or empty inputs can ever produce verified.
    """
    result = VerificationResult()
    try:
        nodes = []
        for source in (original_function_code, candidate_function_code):
            tree = ast.parse(source)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError("Expected one synchronous function")
            node = tree.body[0]
            if node.name != function_name:
                raise ValueError("Function name mismatch")
            nodes.append(node)
        original_node, candidate_node = nodes
        if ast.dump(original_node.args) != ast.dump(candidate_node.args):
            raise ValueError("Parameter API mismatch")
    except (SyntaxError, ValueError, TypeError, RecursionError):
        result.verification_status = "inconclusive"
        result.reason = "Sources must contain one matching synchronous function with an unchanged parameter API."
        return result
    try:
        profiles = resolve_input_profiles(original_node)
        result.input_profiles = [profile.to_dict() for profile in profiles]
        result.input_profile_source = "ast_inference" if any(p.source == "ast_inference" for p in profiles) else "annotation"
        generation = generate_test_cases(original_node)
        candidate_effect = unsupported_effects(candidate_node)
    except Exception:
        result.verification_status = "error"
        result.reason = "Test input generation infrastructure failed; no executions were requested."
        return result
    if generation.status != "supported" or candidate_effect:
        result.verification_status = "inconclusive"
        result.reason = generation.reason if generation.status != "supported" else candidate_effect
        return result
    result.tests_available = len(generation.cases)
    try:
        cases = select_test_cases(generation.cases, case_limit)
    except (ValueError, TypeError):
        result.verification_status = "error"
        result.reason = "Invalid verification case limit."
        return result
    result.tests_total = len(cases)
    if not cases:
        result.verification_status = "inconclusive"
        result.reason = "APEX could not generate reliable behavioral test cases for this function signature."
        return result
    if executor is None:
        result.verification_status = "inconclusive"
        result.reason = "No isolated executor has been configured; no tests were run."
        return result

    matched_returns = 0
    for case in cases:
        try:
            # Protect candidate inputs and later cases from original-side mutation.
            original = executor.execute_function(original_function_code, function_name, deepcopy(case.args), deepcopy(case.kwargs))
            candidate = executor.execute_function(candidate_function_code, function_name, deepcopy(case.args), deepcopy(case.kwargs))
            comparison = compare_results(original, candidate, policy)
            if (comparison.status == "passed" or comparison.kind == "input_mutation") and all(type(item.post_args) is list and type(item.post_kwargs) is dict for item in (original, candidate)):
                result.mutation_checks_performed += 1
        except Exception:
            # Do not expose raw worker exceptions or transport payloads.
            comparison = CaseComparison("error", "infrastructure_error", "Executor invocation or observation handling failed.")
        comparison = CaseComparison(comparison.status, comparison.kind, comparison.reason, case.case_id)
        result.tests_run += 1
        if comparison.status == "passed":
            result.tests_passed += 1
            matched_returns += comparison.kind == "return_value"
        elif comparison.status == "failed":
            result.tests_failed += 1
            result.mismatches.append(comparison)
        elif comparison.status == "inconclusive":
            result.tests_inconclusive += 1
            result.issues.append(comparison)
        else:
            result.tests_errors += 1
            result.issues.append(comparison)
            break  # Stop scheduling further work after infrastructure failure.

    # Infrastructure failure takes priority; known mismatches remain in the report.
    if result.tests_errors:
        result.verification_status = "error"
        result.reason = "Verification infrastructure failed; inspect recorded issues."
    elif result.tests_failed:
        result.verification_status = "rejected"
        result.reason = "At least one observed return value, exception behavior or input mutation differs."
    elif result.tests_inconclusive or result.tests_run != result.tests_total or not matched_returns:
        result.verification_status = "inconclusive"
        result.reason = "Evidence is incomplete or contains no matching normal returns."
    else:
        result.verification_status = "verified"
        result.reason = "All generated cases matched within the documented comparison scope."
    return result

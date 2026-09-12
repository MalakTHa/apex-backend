"""Docker behavioral verification service for AI candidates."""
import os
from pathlib import Path
from dotenv import load_dotenv
from docker_executor import DockerConfig, DockerExecutor
from verification import ComparisonPolicy, VerificationResult, verify_candidate


def interactive_case_limit() -> int:
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    limit = int(os.getenv("APEX_VERIFICATION_MAX_CASES", "12"))
    if not 1 <= limit <= 36:
        raise ValueError("APEX_VERIFICATION_MAX_CASES must be between 1 and 36")
    return limit


def verify_with_docker(original_function_code: str, candidate_function_code: str,
                       function_name: str, *, config: DockerConfig | None = None,
                       policy: ComparisonPolicy | None = None,
                       case_limit: int | None = None) -> VerificationResult:
    executor = DockerExecutor(config)
    result = verify_candidate(original_function_code, candidate_function_code, function_name,
                              executor, policy=policy, case_limit=case_limit)
    if executor.pending_cleanup:
        executor.retry_cleanup()
        if executor.pending_cleanup:
            result.verification_status = "error"
            # Generated container names only, never raw Docker exception text.
            result.reason = "Docker cleanup requires recovery for owned containers: " + ", ".join(executor.pending_cleanup)
    return result

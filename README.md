# APEX Backend

Backend service for the APEX (Algorithm Performance & Efficiency X-ray) platform.

The backend is responsible for analyzing Python source code, estimating algorithm complexity, generating optimization suggestions, applying safe code optimizations, and simulating algorithm performance.

## Features

- Python syntax validation
- Function detection
- Function-level analysis
- Time complexity estimation
- Space complexity estimation
- Algorithm pattern detection
- Performance warnings
- Optimization suggestions
- Simple Optimization
- Algorithm Replacement
- Safe optimization layer
- Performance simulation
- REST API using FastAPI

## Technologies

- Python
- FastAPI
- Uvicorn
- Pydantic
- AST (Abstract Syntax Tree)

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn
```

## Run the server

```bash
uvicorn main:app --reload
```

The API will be available at:

```
https://apex-backend-if96.onrender.com
```

Swagger documentation:

```
https://apex-backend-if96.onrender.com/docs
```

## API Endpoints

### Analyze Code

```
POST /analyze
```

Analyzes the selected Python function and returns:

- Time Complexity
- Space Complexity
- Algorithm Type
- Loop Information
- Warnings
- Suggestions

---

### Simple Optimization

```
POST /optimize/simple
```

Applies safe optimizations without changing the underlying algorithm.

---

### Algorithm Replacement

```
POST /optimize/replacement
```

Replaces inefficient implementations with more efficient algorithms or data structures when a trusted optimization pattern is detected.

---

### Simulation

```
POST /simulate
```

Generates performance comparison data between the original and optimized implementations.


## Notes

- Source code is analyzed statically using Python AST.
- User Python runs only through the explicit verification endpoint inside restricted Docker containers, never in the FastAPI process.
- AI Apply requires successful finite-case behavioral verification; trusted-pattern behavior is unchanged.

## AI-assisted optimization: phase 2 (unverified fallback previews)

`ai_optimizer.py` exposes `suggest_ai_optimization(function_code, analysis)`.
Pass only the selected function's source and its `analyze_function_node()` result.
`POST /optimize/replacement` calls this module only when
`ReplacementEngine.pattern_applied` is false. This optimization request never
executes generated functions. Simple optimization and `/simulate` do not use AI.

Replacement responses retain the original fields and add:

| Result | optimization_source | ai_used | verification_status | pattern_applied |
| --- | --- | --- | --- | --- |
| Trusted transformation | trusted_pattern | false | trusted_pattern | true |
| AI candidate | ai | true | not_verified | false |
| AI decline | none | true | not_applicable | false |
| AI error | none | true | ai_error | false |

AI candidates are parsed again and statically analyzed by APEX; trusted
`after_override` values are never used for them. `full_code` is a candidate copy
for preview and future verification, not an applied result. Errors and declines
retain the original full source and original analysis, with a reason and
`error_code` (null for a decline or candidate). `ai_used` indicates that the AI
module was consulted, even if a missing key prevented a network request.

The frontend labels AI results as "AI-Assisted Candidate / Not Verified Yet"
and blocks Apply Optimization until explicit verification succeeds. Run Simulation
remains blocked for AI, including after verification, with callback guards.
Trusted-pattern status labels the existing rule engine; it is not runtime proof.

Install the backend dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Set `GROQ_API_KEY` in the process environment, or create an untracked `backend/.env`
locally using `.env.example` as the template. Do not commit the key. The module
loads `.env` next to itself when called, independent of the working directory,
with `override=False`: existing environment variables take precedence.
`GROQ_MODEL` defaults to `openai/gpt-oss-120b`. Overrides must support strict
JSON Schema outputs; an unsupported model produces a sanitized API error.

The Groq request uses temperature 0, a 30-second timeout, no automatic retries,
and strict JSON Schema with only `can_optimize`, `optimized_function_code`, and
`reason`. See [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs).
The local result additionally contains `status` (`candidate`, `declined`, `error`)
and `error_code` (null on success/refusal). Errors include `missing_api_key`,
`rate_limited`, `timeout`, `api_unavailable`, `api_error`, `request_failed`,
`invalid_response`, `incomplete_response`, `missing_code`, `invalid_python`,
and function structure/name/parameter/API errors. Raw SDK errors are not returned
or logged by this module.

Candidates must parse as one top-level function and retain their original name,
parameters (including kinds, defaults, and annotations), return annotation,
decorators, and sync/async kind. These are structural checks only: they do not
prove output equivalence, preserve all side effects, or verify performance.
The AI does not supply complexity estimates or a verification status.

`ReplacementEngine.pattern_applied` starts false and is reset on each `optimize()`
call. It becomes true only if the trusted output's AST differs from the input AST,
ignoring source positions and formatting. Early returns and identical templates
remain false. The flag does not certify correctness of a trusted replacement.

### Tests (run from backend)

Tests use unittest and SDK mocks, plus real Docker when available; no live Groq request:

```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
```

One separate manual integration test sends only a synthetic example to Groq:

```bash
python -B -m unittest discover -s tests -p "manual_groq_integration.py" -v
```

The manual test skips if the key is missing. A valid refusal is a successful
connection result. If a candidate is returned, its AST, name, and parameters
are checked without running it. Default test discovery excludes this live test.

Frontend component safety tests (from `frontend`, no added dependencies):

```bash
node --test tests/ai-candidate.test.cjs
npm run build
```

## Phase 3A: verification engine foundation

`verification_inputs.py` generates deterministic test cases from a FunctionDef.
`verification.py` defines the executor protocol, observation/comparison models,
and `verify_candidate(original_function_code, candidate_function_code,
function_name, executor=None, *, policy=None, case_limit=None)`. Neither module imports FastAPI
or Groq. They do not execute, evaluate, or compile function source, and do not
measure time or memory. The separate phase 3B adapter below supplies Docker execution.

Phase 3C connects this engine through the separate `/verify/candidate` endpoint.
`/optimize/replacement` continues to label AI candidates `not_verified`; Apply
requires explicit verification and AI Simulation remains blocked.
The existing trusted transformations, simple optimization and `/simulate` are
unchanged. Verification tests supply scripted fake observations, not actual
execution of the original or candidate functions.

### Supported input generation

- One or two explicitly annotated parameters: `int`, `str`, `list[int]`,
  `list[str]`, or string annotations containing these exact forms.
- Positional-only, ordinary positional and keyword-only parameters are supported.
- Six small deterministic examples per parameter; two parameters use their full
  Cartesian product (36 cases maximum). Examples include zero, negatives, empty
  strings/lists, duplicates, unsorted lists and Unicode strings.
- Every argument is passed explicitly, including parameters with defaults.
  Each call receives a fresh deep copy of its generated args and kwargs.
- Missing/unknown annotations, other types, variadic parameters, zero or more
  than two parameters, decorators, generic, async and generator functions are
  unsupported. Parameter names/defaults are not used to guess input types.

These examples do not establish a function's intended input domain. They do not
cover omitted-default calls, large inputs, aliasing between arguments, closures,
global dependencies, all collection types, or all Python functions.

### Executor and comparison contract

`FunctionExecutor.execute_function(function_code, function_name, args, kwargs)`
returns `ExecutionResult` with `status` in `returned`, `raised`, `timeout`,
`serialization_error`, `error`, and fields `return_value`, `exception_type`,
`exception_message`. Status is the single source of truth for timeouts; there is
no redundant timed_out flag or timing metric. `error` denotes infrastructure,
not an exception raised by user code. Raw infrastructure exceptions are never
copied into the verification report.

The executor must execute each call in a fresh isolated environment and
return losslessly decoded observations. The protocol itself is not a sandbox
and does not enforce isolation; a caller must never plug in an in-process code
executor. Fake executors are used only in tests.

Comparison policy:

- Return values: type-sensitive comparison of `None`, bool, int, finite float,
  str, lists, and string-keyed dictionaries. `True`, `1` and `1.0` are distinct.
  List order matters; dictionary insertion order does not. Floats use exact
  numeric equality, without tolerance. Both values must fit the supported format.
- Unsupported/custom values, tuples, sets, non-finite floats, cycles, excessive
  depth (>32), or more than 10,000 comparison nodes produce an inconclusive
  serialization result. This in-memory comparison format is not yet a wire codec.
- Return versus raised exception, different exception types, or different
  exception messages constitute a mismatch.
- Matching exceptions are inconclusive by default. A caller can explicitly set
  `ComparisonPolicy(expected_exception_types=frozenset({...}))` with qualified
  type names such as `builtins.ValueError`. Only an allowed type with an exactly
  matching message counts as passed. Tracebacks and exception object attributes
  are outside the contract. Even all matching expected exceptions cannot verify
  a suite unless at least one normal return comparison also passes.
- A timeout on either side, even both sides, is inconclusive. Serialization
  failure on either side is inconclusive. Worker/IPC failure or an invalid
  executor observation is an infrastructure error.

Input copies prevent the original run from contaminating candidate inputs;
they do not test whether mutations themselves are equivalent. This phase's
comparison scope is **return values and exception type/message only**. It does
not verify input mutations, stdout, global state, file/network side effects,
object identity, nondeterminism, exception cause/context, or resource use.

### VerificationResult and aggregation

`VerificationResult` has `verification_status`, `tests_total`, `tests_run`,
`tests_passed`, `tests_failed`, `tests_inconclusive`, `tests_errors`, `reason`,
`mismatches`, `issues`, `tests_available`, and `comparison_scope`. `to_dict()` produces a JSON-ready
report. Each mismatch/issue records the case ID, status, kind and fixed reason,
without raw executor messages or output values.

`tests_available` counts all generated cases; `tests_total` counts selected cases,
not individual executor calls. With no case limit these counts agree. `tests_run`
counts attempted comparisons, including a case interrupted by infrastructure
failure. Passed + failed + inconclusive + errors equals tests_run; errors stop
further scheduling and may leave tests_run less than tests_total.

| Status | Condition |
| --- | --- |
| not_verified | No executor configured; no executions requested |
| verified | Nonempty suite, every case passed, and at least one normal return matched |
| rejected | At least one definitive output/exception mismatch and no infrastructure error |
| inconclusive | Unsupported sources/inputs, empty cases, incomplete observations, timeout, serialization problem, or only exception matches |
| error | Infrastructure or executor-contract failure |

Aggregation priority is error > rejected > inconclusive > verified; known
mismatches are retained even if a later infrastructure error occurs. Here,
`verified` means only that the generated finite suite matched within the stated
scope. It is not proof of equivalence for every input. Phase 3C permits Apply only
after the real Docker service succeeds for the exact current source and candidate.
Fake-based test results must not be presented as verification of actual AI code.

### Phase 3B requirements

The Docker adapter below addresses worker/container lifecycle,
OS-enforced isolation and permissions, limits for CPU/memory/processes/output,
timeout and termination policy, fresh per-call state, a bounded lossless IPC
codec, and clear distinction between user exceptions, serialization failures,
timeouts and worker failures. Test the isolation and failure cases separately
before using real source. Limits enforce containment; they are not benchmarks.
Side-effect coverage and unsupported dependencies need explicit policies.
Phase 3C adds endpoint/UI integration below. Benchmarks remain separate work.

## Phase 3B: Docker runtime adapter

New components:

- `docker_executor.py`: bounded Docker CLI transport, availability checks,
  invocation lifecycle, deadlines, cleanup, and protocol validation.
- `runtime/worker.py`: fixed worker that loads and invokes source **inside the
  container only**. Never run/import this worker to execute source on the host.
- `runtime/codec.py`: bounded tagged JSON values shared by host and worker.
- `runtime/Dockerfile` and `.dockerignore`: build context includes only the
  fixed worker/codec and Dockerfile. Project code, `.env`, and credentials are
  not copied into the image or mounted into invocation containers.
- `verification_service.verify_with_docker(...)`: internal service invoking
  `verify_candidate(..., DockerExecutor(...))`. Phase 3C calls it only on explicit verification.

### Image preparation (explicit, not automatic)

The image is `apex-verification:3b-python3.13.5`, based on the explicit official
`python:3.13.5-slim-bookworm` tag. It matches this project's Python 3.13.5 baseline
and needs no pip packages. `latest` and unversioned tags are rejected. The adapter
checks the runtime protocol label and resolves the local tag to a sha256 image
ID before creating the container. A patch-version tag is not a permanent registry
digest pin: rebuild/review the base deliberately when applying security updates.

From `backend`, with Docker Desktop running in **Linux containers** mode:

```bash
docker build --tag apex-verification:3b-python3.13.5 runtime
```

This build may download the base image. Verification never builds or pulls an
image automatically. A missing image returns `error_code=docker_image_unavailable`.

### Effective invocation controls

Defaults are centralized in `DockerConfig`:

| Control | Setting |
| --- | --- |
| Lifecycle | new UUID-named container per call, `create --rm`, then `start --attach --interactive` |
| Invocation deadline | 5 seconds (configurable up to 30), including Docker start/attach overhead |
| Docker control / cleanup deadline | 10 / 5 seconds per CLI operation |
| Network | `--network none`, no published ports |
| Filesystem | `--read-only`, no host/project/socket mounts, no writable tmpfs |
| Memory / memory+swap | `--memory 128m --memory-swap 128m` (no extra swap allowance) |
| CPU / process count | `--cpus 0.5 --pids-limit 32` |
| Identity | non-root `--user 65534:65534` |
| Privileges | `--cap-drop ALL --security-opt no-new-privileges=true` |
| Syscalls | Docker's default seccomp remains enabled; no unconfined profile |
| Other | `--init --ipc none --log-driver none`, nofile 64, core dumps disabled |

Availability fails closed if the daemon is unavailable, Windows-container mode
is selected, or required seccomp/memory/swap/PID/CPU-quota support is absent.
Docker-created device/proc mounts still exist; read-only root is not a promise
that every special filesystem is unwritable. `/tmp` is read-only in this setup.
Resource limits contain execution; no resource usage or benchmark is measured.

The host launches **only Docker CLI** using a subprocess argv list and
`shell=False`. User source and arguments never appear in the shell/argv. The
client gets a small OS/Docker environment allowlist, with no `GROQ_API_KEY`.
No application environment is forwarded to the container. Automatic Docker
proxy variables are explicitly cleared to avoid proxy credentials from client
configuration entering the runtime.

### IPC, serialization and exception handling

Request JSON goes over stdin and contains only `function_code`, `function_name`,
`args`, `kwargs`. Values in args/kwargs use tags. Worker stdout is exactly one
JSON response. Ordinary stdout/stderr and direct writes to descriptors 1/2 from
the function are redirected to `/dev/null`, not accumulated in RAM. The host
also enforces a byte cap while draining stdout/stderr, so malformed output or
an attempted protocol flood cannot grow an unbounded host buffer.

Limits: 64 KiB function source, 128 KiB request/result/stdout/stderr each, depth
32, 10,000 value nodes, strings 65,536 characters, integers 4,096 digits,
exception messages 1,024 characters. Strict JSON rejects duplicate keys,
non-finite constants, multiple objects and unknown response fields/statuses.

Tagged serialization preserves null/bool/int/finite-float/str/list/tuple and
string-keyed dicts, including nesting and tuple/list distinctions. Integer and
float tags use textual numeric representations to avoid JSON numeric coercion.
No pickle, custom object reconstruction, sets or arbitrary dictionary keys.
An unsupported value produces `serialization_error`. The existing 3A comparison
policy deliberately still treats tuple results as inconclusive; wire support
does not expand its comparison scope.

Worker statuses are `returned`, `raised`, `serialization_error`, `error`.
User exceptions report only qualified type and bounded message, without a
traceback. Invalid source and a missing function have fixed worker error codes.
The host alone reports `timeout`. `ExecutionResult.error_code` is optional and
allows fixed infrastructure codes (including `docker_unavailable`) to appear
in the verification report without raw CLI stderr, paths or SDK exceptions.

### Cleanup and failure behavior

Containers are created before attaching so their UUID names are known before
user code can run. On success, timeout, invalid IPC or failure, cleanup issues
`docker rm --force <owned-name>` and confirms absence with an exact-name filter.
Only names created by this adapter are targeted; no global pruning or cleanup
of unrelated containers is performed. Timeout kills the Docker client and then
force-removes the container; killing the client alone is not container cleanup.

If removal cannot be confirmed, the result is `error` with
`docker_cleanup_failed`, not a successful verification or ordinary timeout.
The adapter retains `pending_cleanup` and blocks new work until `retry_cleanup()`
succeeds. A create-request timeout retains the name for a possible late daemon
completion. The internal service makes another cleanup attempt and reports
remaining generated names for recovery if Docker remains unavailable.

No client can guarantee immediate removal while the daemon is unreachable, or
after the host/client process crashes. `--rm` is not a lifetime watchdog for a
still-running container. Pending names are in memory; durable recovery and an
external reaper/watchdog are necessary before exposing this to untrusted traffic.
These failures must not be presented as evidence that no containers remain.

### Tests and current limits

```bash
# All backend tests: mocks plus real Docker tests when available
python -B -m unittest discover -s tests -p "test_*.py" -v

# Real Docker isolation suite only
python -B -m unittest discover -s tests -p "test_docker_integration.py" -v
```

Real tests explicitly skip if the daemon/required isolation controls or prepared
runtime image are unavailable. Skips are not evidence that isolation passed.
Coverage includes return/exception/timeout, invalid source/function name,
serialization, print/direct-FD output, network denial, read-only `/tmp`, missing
secret environment, non-root/seccomp/capability state, bounded process creation,
fresh containers, cleanup, and the internal verification service. Groq is never
required. Existing Fake Executor tests remain unchanged.

On Windows, Docker Desktop must supply a running Linux VM/WSL2 engine; the Docker
CLI must be on PATH and the invoking account must have daemon access. Native
Windows containers are unsupported. Startup/attach overhead can consume the
invocation deadline, so it is configurable. A remote Docker context is trusted
infrastructure and receives the supplied source; use the intended local context.

Containers share their Linux kernel and are defense in depth, not a formally
verified hostile-code sandbox. The worker and user code share a Python interpreter;
IPC validation is not cryptographic attestation against code intentionally
tampering with worker internals. Also, mutation/filesystem/network side effects,
nondeterminism and universal equivalence remain outside the 3A comparison scope.

Before deploying verification publicly, review the real isolation suite on that
environment and provide authenticated/bounded job scheduling and durable cleanup.
The local endpoint integration below does not deploy the Docker runtime.

## Phase 3C: explicit candidate verification

`POST /verify/candidate` accepts only source and function selection as verification
inputs; client analysis is not trusted:

```json
{
  "original_code": "def total(values: list[int]):\n    result = 0\n    for value in values:\n        result += value\n    return result\n",
  "function_name": "total",
  "candidate_function_code": "def total(values: list[int]):\n    return sum(values)\n"
}
```

The endpoint extracts one unambiguous top-level original function, checks candidate
syntax/name/full parameter API, and recalculates `before` and `after` using APEX.
Invalid candidate syntax or API is rejected before Docker. Nested functions and
unsupported input annotations are inconclusive. Groq is never called here.
The existing service, Docker executor and comparison engine perform verification.

`APEX_VERIFICATION_MAX_CASES=12` is the default interactive cap (valid range 1..36).
`verification_cases.py` selects a deterministic subset, prioritizing coverage of
distinct values for each parameter, then spread across the generated sequence.
The original generator still produces all six or 36 cases. Direct engine calls
without a cap retain all cases. Each selected case invokes original and candidate
in separate fresh containers; 12 cases therefore mean 24 invocations.

Responses include `verification_status`, `tests_available`, `tests_total`,
`tests_run`, `tests_passed`, `tests_failed`, `tests_inconclusive`, `tests_errors`,
`comparison_scope`, sanitized `reason`, mismatch case IDs/kinds, static analyses,
and SHA-256 fingerprints of the exact original and candidate source strings.
Only `verified` includes `verified_full_code`, reconstructed from the submitted
full original and the exact candidate. Fingerprints identify versions; they are
neither verification nor authentication tokens.

| Status | UI behavior |
| --- | --- |
| not_verified | Verify Candidate available; Apply disabled |
| verified | Verified by APEX; Apply uses only verified_full_code |
| rejected | Verification Failed; Apply disabled |
| inconclusive | Verification Inconclusive; Apply disabled |
| error | Verification Runtime Error; Apply disabled |

The UI shows loading and blocks duplicate requests, displays counts and scope,
and discards replies if the source or selected optimization changed while waiting.
Apply also checks stored source/candidate snapshots against the current versions.
AI Simulation remains disabled in every status. The frontend uses the existing
Render API origin; this change does not publish that endpoint or provision Docker
on the remote server. Local endpoint tests use FastAPI TestClient.

Verification means: "Verified across the generated test cases within the supported
verification scope." It does not cover all possible inputs or side effects.
The request remains synchronous and container startup can take appreciable time;
the case cap is not a job queue or whole-request deadline. No benchmark, execution
time measurement, memory measurement, or AI-generated test cases are added.

`tests/test_verify_endpoint.py` covers validation, exact candidate forwarding,
sanitization, analysis, case selection and no automatic verification/Groq call.
`tests/test_verify_docker_e2e.py` tests the HTTP endpoint with real containers:
`sum(values)` passes and `sum(values) + 1` is rejected. These two tests skip only
when the daemon is unavailable; a missing image or isolation control fails them.
Frontend tests cover Verify, loading, statuses, exact verified-code Apply, stale
snapshots, finite-scope explanation and the continuing AI Simulation block.

## Phase 4A: isolated execution-time benchmark

`POST /benchmark/candidate` is independent of `/simulate` and the frontend.
It accepts `original_code`, `function_name`, `candidate_function_code`, and an
optional `candidate_fingerprint` (lowercase SHA-256 of the exact candidate string).
A mismatched fingerprint prevents execution. A fingerprint only identifies source;
a client-provided `verification_status` or earlier report cannot authorize timing.

There is no trusted server session/evidence store in this project. Therefore the
service **always runs fresh behavioral verification of the exact submitted pair**
before benchmarking. It verifies all generated cases (up to 36), then measures a
deterministic subset. This adds latency, but avoids inventing trust in frontend
state, persisted claims, or unsigned fingerprints. No Groq call or optimization is
performed. Existing `/verify/candidate` behavior and its interactive cap remain
unchanged. A future evidence cache would need server-owned source/policy/runtime
bindings, expiration and concurrency controls; none is required for this phase.

`benchmark.py` defines `BenchmarkExecutor`, `BenchmarkConfig`, `BenchmarkResult`,
statistics, case orchestration and `benchmark_with_docker`. The endpoint performs
structural source/API validation and delegates to the service. It does not time
code or estimate milliseconds from Big-O. Supported input generation is still
the limited annotated scalar/list generator described above, without AI inference.

### Runtime preparation and contract

Build the separate versioned image explicitly, from `backend`:

```bash
docker build --pull=false --tag apex-verification:4a-python3.13.5 runtime
```

The 3B image remains the default for existing verification calls; 4A uses the new
image for both its prerequisite verification and timings. The adapter pins the
first resolved image ID for its whole job. Both versions of the function use the
same immutable DockerConfig, image ID, generated values and repetition counts.
All existing network/filesystem/user/cgroup/environment/output/cleanup restrictions
remain in the single shared Docker transport path. No automatic build or pull is
performed by an endpoint. This local build does not deploy Docker to Render.

Legacy requests without `operation` still mean `execute`; explicit `execute` is
also accepted. `benchmark` adds `warmup_runs` and `measurement_runs` to the same
source/name/tagged-args/tagged-kwargs request. Its `returned` observation contains
a tagged list of integer nanosecond samples in `return_value`; other statuses keep
the existing exception/timeout/error contract. This operation-specific payload
does not alter verification's return-value contract. The host validates sample
count, integer types and bounds before using timings.

`runtime/measurement.py` is called by the fixed worker after source loading.
Every warmup and measured iteration decodes fresh args/kwargs from the original
tagged JSON, including nested collections. Decode happens before timing. Only
the function call sits between two `time.perf_counter_ns()` reads. Compilation,
function defaults/annotations, warmup, input decoding, result disposal, Docker
startup, Python startup and IPC are outside measured intervals. Unit tests use
injected harmless callables and a fake clock; user source never executes on host.

### Defaults, summaries and statuses

Environment settings load from `backend/.env` without overriding process values:

| Setting | Default | Accepted range |
| --- | --- | --- |
| APEX_BENCHMARK_WARMUP_RUNS | 3 | 0..20 |
| APEX_BENCHMARK_MEASUREMENT_RUNS | 15 | 3..100 |
| APEX_BENCHMARK_MAX_CASES | 6 | 1..36 |

Case selection reuses the deterministic diversity selector and never changes the
input generator. Original/candidate order alternates across cases to reduce order
bias. Each function/case gets a fresh container; repetitions share that container
and function object but get fresh inputs. Module state/caches and randomness are
not reset per repetition. Stateful/nondeterministic programs remain outside the
verification guarantees; fresh inputs are not a general side-effect equivalence
check. Cases that raise during timing contribute no samples to aggregates.

Per-case statistics summarize that case's measurement runs. The top-level median
is the **median of per-case medians**, giving each measured case equal weight.
Top-level mean/min/max also summarize those per-case medians, not pooled samples.
`speedup = original.median_ns / optimized.median_ns`. The ratio is null if either
median is below 1,000 ns, or the suite is incomplete. This one-microsecond floor
is a conservative display rule, not a statistical confidence guarantee. Zero,
negative, non-integer or out-of-bound samples are invalid and cause `error`.

| benchmark_status | Meaning |
| --- | --- |
| completed | Fresh verification succeeded and every selected case returned valid timings |
| inconclusive | Verification did not succeed, sources/inputs unsupported, no cases, timeout, exception or serialization failure |
| error | Configuration, Docker/cleanup, invalid samples or infrastructure failure |

Partial normal-return case statistics remain diagnostic; an incomplete job never
gets an overall speedup. Benchmark failures do not rewrite the behavioral decision.
Public reasons contain no raw Docker stderr, host paths, SDK errors or tracebacks.
Input summaries disclose types/collection lengths, with bounded parameter labels,
and never dump full values. Responses contain:

```text
benchmark_status, function, verification_status,
cases_total, cases_benchmarked, warmup_runs, measurement_runs,
original/optimized: {median_ns, mean_ns, min_ns, max_ns},
speedup, reason, aggregation, candidate_fingerprint, original_fingerprint,
case_results: [{case_index, case_id, input_summary, status,
                original_median_ns, optimized_median_ns, speedup,
                original/optimized: {median_ns, mean_ns, min_ns, max_ns}}]
```

Fingerprints in this report refer to the exact extracted original function string
and exact candidate string used by the service, not the whole original module.
`cases_total` is the selected timing count, not the number of verification cases
or repetitions. Each selected case normally contributes two sets of repetitions.

### Validation and limitations

`tests/test_benchmark.py` covers statistics, faster/slower candidates, invalid and
tiny samples, fresh nested inputs and timing boundaries, deterministic limits,
verification prerequisite, exceptions/timeouts, endpoint validation/sanitization,
no Groq, protocol compatibility and image pinning. `tests/test_benchmark_docker.py`
uses real containers for the endpoint's verified quadratic/linear example, timing
boundaries, fresh inputs, isolation, exceptions and timeout cleanup. Only an
unavailable daemon skips these real tests; missing image/isolation fails them.

Run all backend tests with the existing discovery command; run frontend tests and
build with the existing commands above. The real performance example verifies
both functions before measuring and checks a broad median ordering, not an exact
speedup. It prints actual nanosecond statistics for review.

Measurements are elapsed function-call time, including scheduling/cgroup throttling
that happens during the call, plus timer overhead. They depend on the current
machine, Docker/Linux VM, CPU load and runtime; they are not portable absolute
performance claims. Small generated inputs do not represent production workloads
or establish asymptotic scaling. Startup is excluded from timings but still costs
request latency. No queue, cancellation endpoint or whole-job deadline is added.
Container/worker trust limitations described in 3B still apply. No memory or CPU
profiling, frontend/Simulation changes, charts or remote deployment are included.

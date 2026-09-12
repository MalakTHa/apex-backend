"""Docker-only runtime adapter. Never runs user Python on the host.

All subprocess argv start with Docker CLI; shell interpolation is never used.
Docker Desktop must be running in Linux-container mode. Images are built explicitly,
not pulled or built as a side effect of verification.
"""
from dataclasses import dataclass
import json
import os
import re
import subprocess
import threading
import uuid

from runtime.codec import (MAX_CODE_BYTES, MAX_EXCEPTION_MESSAGE, MAX_WIRE_BYTES,
                           SerializationError, decode_value, dump_json, encode_value, load_json)
from verification import ExecutionResult

IMAGE = "apex-verification:3b-python3.13.5"
CONTAINER_PREFIX = "apex-verification-"
WORKER_ERRORS = {"invalid_request", "invalid_function_code", "function_not_found", "worker_environment"}


@dataclass(frozen=True)
class DockerConfig:
    image: str = IMAGE
    invocation_timeout: float = 5.0
    control_timeout: float = 10.0
    cleanup_timeout: float = 5.0
    memory_mb: int = 128
    cpus: float = 0.5
    pids_limit: int = 32
    nofile_limit: int = 64
    user: str = "65534:65534"

    def __post_init__(self):
        tag = self.image.rsplit("/", 1)[-1]
        if ":" not in tag or tag.rsplit(":", 1)[-1] in {"", "latest"}:
            raise ValueError("An explicit non-latest runtime image version is required")
        if not (0 < self.invocation_timeout <= 30 and 0 < self.control_timeout <= 30
                and 0 < self.cleanup_timeout <= 30):
            raise ValueError("Docker deadlines must be positive and at most 30 seconds")
        if not (32 <= self.memory_mb <= 512 and 0 < self.cpus <= 2 and 8 <= self.pids_limit <= 128):
            raise ValueError("Docker resource limits are outside the supported range")
        if self.user != "65534:65534" or self.nofile_limit != 64:
            raise ValueError("Runtime UID and descriptor limits are fixed for this phase")


@dataclass(frozen=True)
class DockerAvailability:
    available: bool
    error_code: str | None = None
    reason: str = "Docker is available."


@dataclass(frozen=True)
class CLIResult:
    returncode: int = 0
    stdout: bytes = b""
    timed_out: bool = False
    output_exceeded: bool = False


def _cli_environment():
    # Keep Docker/OS connection settings, not application API keys. These are
    # client-side settings only; none are supplied as container environment values.
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
               "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
               "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_TLS_VERIFY",
               "DOCKER_CERT_PATH", "SSH_AUTH_SOCK", "XDG_RUNTIME_DIR"}
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def _run_docker(arguments, *, input_data=b"", timeout=10.0) -> CLIResult:
    """Bound stdout/stderr during collection, not after communicate() allocates it."""
    process = subprocess.Popen(
        ["docker", *arguments], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, shell=False, env=_cli_environment(),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    ready = threading.Event()
    overflow = threading.Event()
    buffers = [bytearray(), bytearray()]

    def read_pipe(stream, buffer):
        try:
            while True:
                block = stream.read(4096)
                if not block:
                    break
                remaining = MAX_WIRE_BYTES - len(buffer)
                buffer.extend(block[:remaining])
                if len(block) > remaining:
                    overflow.set()
                    ready.set()
                    break
        finally:
            stream.close()

    def write_pipe():
        try:
            process.stdin.write(input_data)
            process.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def wait_process():
        process.wait()
        ready.set()

    readers = [threading.Thread(target=read_pipe, args=(stream, buffer), daemon=True)
               for stream, buffer in zip((process.stdout, process.stderr), buffers)]
    writer = threading.Thread(target=write_pipe, daemon=True)
    waiter = threading.Thread(target=wait_process, daemon=True)
    for thread in [*readers, writer, waiter]:
        thread.start()
    expired = not ready.wait(timeout)
    if (expired or overflow.is_set()) and process.poll() is None:
        process.kill()  # Kill only the Docker client; caller removes the named container.
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        raise OSError("Docker client did not terminate")
    for thread in [*readers, writer, waiter]:
        thread.join(timeout=1)
    if any(thread.is_alive() for thread in readers):
        return CLIResult(process.returncode, b"", output_exceeded=True)
    return CLIResult(process.returncode, bytes(buffers[0]), expired, overflow.is_set())


class DockerExecutor:
    """Use one adapter per verification job. No containers are shared across calls."""
    def __init__(self, config: DockerConfig | None = None):
        self.config = config or DockerConfig()
        self._pending_cleanup = set()
        self.last_container_name = None
        self._pinned_image_id = None

    @property
    def pending_cleanup(self):
        return tuple(sorted(self._pending_cleanup))

    def _call(self, arguments, *, input_data=b"", timeout=None):
        return _run_docker(arguments, input_data=input_data,
                           timeout=self.config.control_timeout if timeout is None else timeout)

    def availability(self) -> DockerAvailability:
        try:
            reply = self._call(["info", "--format", "{{json .}}"])
            if reply.returncode or reply.timed_out or reply.output_exceeded:
                raise OSError("Docker unavailable")
            info = json.loads(reply.stdout)
        except (OSError, ValueError):
            return DockerAvailability(False, "docker_unavailable", "Docker CLI or daemon is unavailable.")
        if type(info) is not dict:
            return DockerAvailability(False, "docker_unavailable", "Docker returned invalid daemon information.")
        if info.get("OSType") != "linux":
            return DockerAvailability(False, "docker_isolation_unavailable", "Linux containers are required.")
        options = info.get("SecurityOptions")
        seccomp = type(options) is list and any(type(option) is str and "name=seccomp" in option
                                               and "profile=unconfined" not in option for option in options)
        if not seccomp or not all(info.get(key) is True for key in ("MemoryLimit", "SwapLimit", "PidsLimit", "CpuCfsQuota")):
            return DockerAvailability(False, "docker_isolation_unavailable", "Required seccomp/cgroup resource controls are unavailable.")
        return DockerAvailability(True)

    def _image_id(self):
        reply = self._call(["image", "inspect", self.config.image, "--format", "{{json .}}"])
        if reply.returncode or reply.timed_out or reply.output_exceeded:
            return None
        try:
            image = json.loads(reply.stdout)
            config = image.get("Config", {})
            identifier = image.get("Id", "")
            if (image.get("Os") != "linux" or config.get("Labels", {}).get("org.apex.runtime.protocol") != "1"
                    or config.get("Volumes") or not re.fullmatch(r"sha256:[0-9a-f]{64}", identifier)):
                return None
            return identifier  # Pin this invocation to a content-addressed local image.
        except (ValueError, AttributeError, TypeError):
            return None

    def _create_arguments(self, name, image_id):
        cfg = self.config
        arguments = [
            "create", "--rm", "--name", name, "--label", "org.apex.verification=3b",
            "--interactive", "--pull", "never", "--network", "none", "--read-only",
            "--memory", f"{cfg.memory_mb}m", "--memory-swap", f"{cfg.memory_mb}m",
            "--cpus", str(cfg.cpus), "--pids-limit", str(cfg.pids_limit),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
            "--user", cfg.user, "--init", "--ipc", "none", "--log-driver", "none",
            "--ulimit", f"nofile={cfg.nofile_limit}:{cfg.nofile_limit}",
            "--ulimit", "core=0:0", "--no-healthcheck", "--stop-timeout", "1",
            "--workdir", "/opt/apex-runtime", "--entrypoint", "python",
        ]
        # Docker client config may automatically inject proxy variables. Override
        # them explicitly with empty values; never forward application environment.
        for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY",
                         "http_proxy", "https_proxy", "all_proxy", "no_proxy", "ftp_proxy"):
            arguments.extend(["--env", variable + "="])
        return [*arguments, image_id, "-I", "-B", "/opt/apex-runtime/worker.py"]

    def _cleanup(self, name):
        try:
            # Removal is restricted to this instance's UUID names, never a global prune.
            self._call(["rm", "--force", name], timeout=self.config.cleanup_timeout)
            check = self._call(["container", "ls", "--all", "--filter", f"name=^/{name}$", "--format", "{{.ID}}"],
                               timeout=self.config.cleanup_timeout)
            clean = not (check.returncode or check.timed_out or check.output_exceeded or check.stdout.strip())
        except OSError:
            clean = False
        if clean:
            self._pending_cleanup.discard(name)
        return clean

    def retry_cleanup(self):
        """Retry only containers owned by this adapter after daemon recovery."""
        outcomes = [self._cleanup(name) for name in tuple(self._pending_cleanup)]
        return all(outcomes)

    @staticmethod
    def _observation(data):
        try:
            response = load_json(data)
            if type(response) is not dict:
                raise SerializationError("Invalid result")
            state = {}
            if "post_args" in response or "post_kwargs" in response:
                args = decode_value(response.pop("post_args"))
                kwargs = decode_value(response.pop("post_kwargs"))
                if type(args) is not list or type(kwargs) is not dict:
                    raise SerializationError("Invalid post-call state")
                state = {"post_args": args, "post_kwargs": kwargs}
            status = response.get("status")
            if status == "returned" and set(response) == {"status", "return_value"}:
                return ExecutionResult("returned", decode_value(response["return_value"]), **state)
            if status == "raised" and set(response) == {"status", "exception_type", "exception_message"}:
                name, message = response["exception_type"], response["exception_message"]
                if (type(name) is not str or len(name) > 256 or not re.fullmatch(r"[\w.<>]+", name)
                        or type(message) is not str or len(message) > MAX_EXCEPTION_MESSAGE):
                    raise SerializationError("Invalid exception")
                return ExecutionResult("raised", exception_type=name, exception_message=message, **state)
            if status == "serialization_error" and set(response) == {"status"}:
                return ExecutionResult("serialization_error")
            if status == "error" and set(response) == {"status", "error_code"} and response["error_code"] in WORKER_ERRORS:
                return ExecutionResult("error", error_code=response["error_code"])
        except (ValueError, TypeError, KeyError, RecursionError):
            pass
        return ExecutionResult("error", error_code="invalid_worker_response")

    def benchmark_function(self, function_code, function_name, args, kwargs, *, warmup_runs=3, measurement_runs=15):
        if (type(warmup_runs) is not int or not 0 <= warmup_runs <= 20
                or type(measurement_runs) is not int or not 3 <= measurement_runs <= 100):
            return ExecutionResult("error", error_code="invalid_request")
        return self.execute_function(function_code, function_name, args, kwargs,
                                     _benchmark=(warmup_runs, measurement_runs))

    def execute_function(self, function_code, function_name, args, kwargs, *, _benchmark=None) -> ExecutionResult:
        available = self.availability()
        if not available.available:
            return ExecutionResult("error", error_code=available.error_code)
        if not self.retry_cleanup():
            return ExecutionResult("error", error_code="docker_cleanup_failed")
        try:
            if (type(function_code) is not str or len(function_code.encode("utf-8")) > MAX_CODE_BYTES
                    or type(function_name) is not str or not function_name.isidentifier()
                    or type(args) is not list or type(kwargs) is not dict):
                return ExecutionResult("error", error_code="invalid_request")
            request = {"function_code": function_code, "function_name": function_name,
                       "args": encode_value(args), "kwargs": encode_value(kwargs)}
            if _benchmark is not None:
                request.update(operation="benchmark", warmup_runs=_benchmark[0], measurement_runs=_benchmark[1])
            payload = dump_json(request)
        except (ValueError, TypeError, RecursionError, UnicodeError):
            return ExecutionResult("serialization_error")
        try:
            image_id = self._pinned_image_id or self._image_id()
            self._pinned_image_id = image_id
        except OSError:
            image_id = None
        if image_id is None:
            return ExecutionResult("error", error_code="docker_image_unavailable")

        name = CONTAINER_PREFIX + uuid.uuid4().hex
        self.last_container_name = name
        self._pending_cleanup.add(name)
        observation = ExecutionResult("error", error_code="docker_execution_failed")
        create_timed_out = False
        try:
            created = self._call(self._create_arguments(name, image_id))
            create_timed_out = created.timed_out
            if created.returncode or created.timed_out or created.output_exceeded:
                observation = ExecutionResult("error", error_code="docker_create_failed")
            else:
                reply = self._call(["start", "--attach", "--interactive", name], input_data=payload,
                                   timeout=self.config.invocation_timeout)
                if reply.timed_out:
                    observation = ExecutionResult("timeout")
                elif reply.output_exceeded:
                    observation = ExecutionResult("error", error_code="output_limit")
                elif reply.returncode:
                    observation = ExecutionResult("error", error_code="docker_execution_failed")
                else:
                    observation = self._observation(reply.stdout)
        except OSError:
            observation = ExecutionResult("error", error_code="docker_execution_failed")
        finally:
            cleaned = self._cleanup(name)
            if create_timed_out:
                # A timed-out daemon create request might finish late. Keep it in
                # the retry ledger even if the first removal found nothing.
                self._pending_cleanup.add(name)
            if not cleaned or create_timed_out:
                observation = ExecutionResult("error", error_code="docker_cleanup_failed")
        return observation

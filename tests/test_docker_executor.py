"""Offline Docker CLI mocks and codec tests; never invoke Python source on host."""
import ast
import io
import json
import os
import subprocess
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

import docker_executor as docker
from runtime.codec import SerializationError, decode_value, dump_json, encode_value, load_json, MAX_WIRE_BYTES
from verification import ExecutionResult, verify_candidate
from verification_service import verify_with_docker

INFO = {"OSType": "linux", "SecurityOptions": ["name=seccomp,profile=builtin"],
        "MemoryLimit": True, "SwapLimit": True, "PidsLimit": True, "CpuCfsQuota": True}
IMAGE_ID = "sha256:" + "a" * 64
IMAGE_INFO = {"Id": IMAGE_ID, "Os": "linux", "Config": {"Labels": {"org.apex.runtime.protocol": "1"}}}
SOURCE = "def f(value: int):\n    return value + 1\n"


class CodecTests(unittest.TestCase):
    def test_round_trip_preserves_types_and_tuple(self):
        for value in (None, True, 1, -2, 1.25, "\u0645\u0631\u062d\u0628\u0627", [1, True], (1, "x"), {"x": (1, ["a"])}):
            with self.subTest(value=value):
                restored = decode_value(load_json(dump_json(encode_value(value))))
                self.assertEqual(restored, value)
                self.assertIs(type(restored), type(value))
        self.assertNotEqual(encode_value([1]), encode_value((1,)))

    def test_unsupported_values_are_rejected(self):
        cycle = []
        cycle.append(cycle)
        for value in (object(), {1, 2}, {1: "x"}, float("inf"), float("nan"), cycle, "x" * 65537):
            with self.subTest(kind=type(value).__name__):
                self.assertRaises(SerializationError, encode_value, value)

    def test_invalid_wire_tags_and_duplicates(self):
        for node in ({"t": "int", "v": True}, {"t": "int", "v": "01"}, {"t": "float", "v": "nan"},
                     {"t": "pickle", "v": "payload"}, {"t": "null", "extra": 1},
                     {"t": "dict", "v": [["a", {"t": "null"}], ["a", {"t": "null"}]]}):
            self.assertRaises(SerializationError, decode_value, node)
        for data in (b'{"status":1,"status":2}', b'{"v":NaN}', b'{}\n{}', b'x' * (MAX_WIRE_BYTES + 1)):
            self.assertRaises(SerializationError, load_json, data)

    def test_payload_size_is_bounded(self):
        self.assertRaises(SerializationError, dump_json, {"large": "x" * MAX_WIRE_BYTES})


class DockerExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = docker.DockerExecutor()
        self.reply = docker.CLIResult(stdout=dump_json({"status": "returned", "return_value": encode_value(2)}))
        self.calls = []
        self.cleanup_present = False
        patcher = patch("docker_executor._run_docker", side_effect=self.cli)
        self.mock_cli = patcher.start()
        self.addCleanup(patcher.stop)

    def cli(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if argv[0] == "info":
            return docker.CLIResult(stdout=json.dumps(INFO).encode())
        if argv[:2] == ["image", "inspect"]:
            return docker.CLIResult(stdout=json.dumps(IMAGE_INFO).encode())
        if argv[0] == "create":
            return docker.CLIResult(stdout=b"container-id")
        if argv[0] == "start":
            return self.reply
        if argv[0] == "rm":
            return docker.CLIResult()
        if argv[:2] == ["container", "ls"]:
            return docker.CLIResult(stdout=b"still-present" if self.cleanup_present else b"")
        raise AssertionError(argv)

    def execute(self, code=SOURCE, name="f", args=None, kwargs=None):
        return self.executor.execute_function(code, name, [1] if args is None else args, {} if kwargs is None else kwargs)

    def test_normal_return_and_cleanup(self):
        self.assertEqual(self.execute(), ExecutionResult("returned", 2))
        name = self.executor.last_container_name
        self.assertTrue(name.startswith(docker.CONTAINER_PREFIX))
        self.assertIn((["rm", "--force", name], {"input_data": b"", "timeout": 5.0}), self.calls)
        self.assertEqual(self.executor.pending_cleanup, ())

    def test_new_container_for_every_invocation(self):
        self.execute()
        first = self.executor.last_container_name
        self.execute()
        self.assertNotEqual(first, self.executor.last_container_name)

    def test_isolation_flags_and_no_host_mounts_or_secrets(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "secret-test-value"}):
            self.execute()
        create = next(argv for argv, _ in self.calls if argv[0] == "create")
        expected = {"--network": "none", "--memory": "128m", "--memory-swap": "128m",
                    "--cpus": "0.5", "--pids-limit": "32", "--cap-drop": "ALL",
                    "--security-opt": "no-new-privileges=true", "--user": "65534:65534", "--pull": "never"}
        for key, value in expected.items():
            self.assertEqual(create[create.index(key) + 1], value)
        for flag in ("--rm", "--read-only", "--init"):
            self.assertIn(flag, create)
        for forbidden in ("--privileged", "--mount", "--volume", "-v", "--publish", "-p", "--publish-all"):
            self.assertNotIn(forbidden, create)
        self.assertNotIn("seccomp=unconfined", create)
        self.assertIn(IMAGE_ID, create)
        self.assertNotIn(SOURCE, create)
        self.assertNotIn("secret-test-value", str(self.calls))
        self.assertNotIn("GROQ_API_KEY", str(self.calls))

    def test_payload_only_on_stdin_and_no_shell_interpolation(self):
        code = "def f(value):\n    return '$HOME; $(not_a_command)'"
        self.execute(code)
        argv, options = next(call for call in self.calls if call[0][0] == "start")
        payload = load_json(options["input_data"])
        self.assertEqual(set(payload), {"function_code", "function_name", "args", "kwargs"})
        self.assertEqual(payload["function_code"], code)
        self.assertEqual(decode_value(payload["args"]), [1])
        self.assertNotIn(code, argv)
        self.assertEqual(options["timeout"], 5.0)

    def test_exception_observation(self):
        self.reply = docker.CLIResult(stdout=dump_json({"status": "raised", "exception_type": "builtins.ValueError", "exception_message": "bad"}))
        result = self.execute()
        self.assertEqual((result.status, result.exception_type, result.exception_message), ("raised", "builtins.ValueError", "bad"))

    def test_timeout_forces_cleanup(self):
        self.reply = docker.CLIResult(timed_out=True, returncode=-1)
        self.assertEqual(self.execute().status, "timeout")
        self.assertTrue(any(argv[0] == "rm" for argv, _ in self.calls))
        self.assertEqual(self.executor.pending_cleanup, ())

    def test_invalid_code_and_function_missing_worker_errors(self):
        for error_code in ("invalid_function_code", "function_not_found"):
            self.reply = docker.CLIResult(stdout=dump_json({"status": "error", "error_code": error_code}))
            result = self.execute()
            self.assertEqual((result.status, result.error_code), ("error", error_code))
            self.assertEqual(self.executor.pending_cleanup, ())

    def test_serialization_failure(self):
        self.reply = docker.CLIResult(stdout=b'{"status":"serialization_error"}')
        self.assertEqual(self.execute().status, "serialization_error")
        self.calls.clear()
        self.assertEqual(self.execute(args=[object()]).status, "serialization_error")
        self.assertFalse(any(argv[0] == "create" for argv, _ in self.calls))

    def test_docker_unavailable_is_infrastructure_error(self):
        for reply in (docker.CLIResult(returncode=1), docker.CLIResult(timed_out=True), docker.CLIResult(stdout=b"[]")):
            self.mock_cli.return_value = reply
            self.mock_cli.side_effect = None
            result = self.execute()
            self.assertEqual((result.status, result.error_code), ("error", "docker_unavailable"))
        self.mock_cli.side_effect = FileNotFoundError("private-host-path")
        self.assertEqual(self.execute().error_code, "docker_unavailable")

    def test_missing_isolation_controls_fail_closed(self):
        for change in ({"OSType": "windows"}, {"PidsLimit": False}, {"SwapLimit": False}, {"SecurityOptions": []}):
            self.mock_cli.side_effect = None
            self.mock_cli.return_value = docker.CLIResult(stdout=json.dumps({**INFO, **change}).encode())
            self.assertEqual(self.execute().error_code, "docker_isolation_unavailable")

    def test_image_must_be_prepared_and_labeled(self):
        with patch.object(self.executor, "_image_id", return_value=None):
            self.assertEqual(self.execute().error_code, "docker_image_unavailable")
        self.assertFalse(any(argv[0] == "create" for argv, _ in self.calls))

    def test_invalid_json_protocol_is_rejected_and_cleaned(self):
        for data in (b"print pollution\n{}", b"{}\n{}", b"null", b'{"status":"timeout"}',
                     b'{"status":"error","error_code":"raw-secret"}'):
            self.reply = docker.CLIResult(stdout=data)
            result = self.execute()
            self.assertEqual(result.error_code, "invalid_worker_response")
            self.assertNotIn("raw-secret", str(result))
            self.assertEqual(self.executor.pending_cleanup, ())

    def test_output_flood_is_error_and_cleaned(self):
        self.reply = docker.CLIResult(output_exceeded=True)
        self.assertEqual(self.execute().error_code, "output_limit")
        self.assertEqual(self.executor.pending_cleanup, ())

    def test_cleanup_failure_blocks_future_work_until_recovery(self):
        self.cleanup_present = True
        self.assertEqual(self.execute().error_code, "docker_cleanup_failed")
        self.assertEqual(len(self.executor.pending_cleanup), 1)
        creates = sum(argv[0] == "create" for argv, _ in self.calls)
        self.assertEqual(self.execute().error_code, "docker_cleanup_failed")
        self.assertEqual(sum(argv[0] == "create" for argv, _ in self.calls), creates)
        self.cleanup_present = False
        self.assertTrue(self.executor.retry_cleanup())
        self.assertEqual(self.executor.pending_cleanup, ())

    def test_create_timeout_retains_name_for_late_cleanup(self):
        base = self.cli
        self.mock_cli.side_effect = lambda argv, **kw: docker.CLIResult(timed_out=True) if argv[0] == "create" else base(argv, **kw)
        self.assertEqual(self.execute().error_code, "docker_cleanup_failed")
        self.assertEqual(len(self.executor.pending_cleanup), 1)

    def test_client_crash_still_cleans_container(self):
        base = self.cli
        def failing(argv, **kw):
            if argv[0] == "start":
                raise OSError("secret-client-details")
            return base(argv, **kw)
        self.mock_cli.side_effect = failing
        result = self.execute()
        self.assertEqual(result.error_code, "docker_execution_failed")
        self.assertNotIn("secret-client-details", str(result))
        self.assertEqual(self.executor.pending_cleanup, ())

    def test_real_adapter_connects_to_verification_engine_with_cli_mock(self):
        self.reply = docker.CLIResult(stdout=dump_json({"status": "returned", "return_value": encode_value(2),
            "post_args": encode_value([1]), "post_kwargs": encode_value({})}))
        result = verify_candidate(SOURCE, SOURCE, "f", self.executor)
        self.assertEqual(result.verification_status, "verified")
        self.assertEqual(result.tests_passed, 6)
        self.mock_cli.side_effect = FileNotFoundError()
        failed = verify_with_docker(SOURCE, SOURCE, "f")
        self.assertEqual(failed.verification_status, "error")
        self.assertIn("docker_unavailable", failed.issues[0].reason)

    def test_no_groq_api_or_host_python_execution_dependency(self):
        source = Path(docker.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile"})
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, {"groq", "main", "ai_optimizer"})


class DockerTransportTests(unittest.TestCase):
    class Process:
        def __init__(self, output=b"{}"):
            self.stdin, self.stdout, self.stderr = io.BytesIO(), io.BytesIO(output), io.BytesIO()
            self.returncode = 0
        def wait(self, timeout=None):
            return self.returncode
        def poll(self):
            return self.returncode
        def kill(self):
            self.returncode = -1

    def test_subprocess_is_docker_only_and_shell_false(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "must-not-inherit"}):
            with patch("docker_executor.subprocess.Popen", return_value=self.Process()) as spawn:
                result = docker._run_docker(["info"])
        self.assertEqual(result.stdout, b"{}")
        self.assertEqual(spawn.call_args.args[0], ["docker", "info"])
        self.assertIs(spawn.call_args.kwargs["shell"], False)
        self.assertNotIn("GROQ_API_KEY", spawn.call_args.kwargs["env"])

    def test_output_is_bounded_during_reading(self):
        with patch("docker_executor.subprocess.Popen", return_value=self.Process(b"x" * (MAX_WIRE_BYTES + 100))):
            result = docker._run_docker(["start", "container"])
        self.assertTrue(result.output_exceeded)
        self.assertLessEqual(len(result.stdout), MAX_WIRE_BYTES)

    def test_host_deadline_kills_only_docker_client(self):
        class HangingProcess(self.Process):
            def __init__(self):
                super().__init__()
                self.returncode = None
                self.done = threading.Event()
            def wait(self, timeout=None):
                if not self.done.wait(timeout):
                    raise subprocess.TimeoutExpired("docker", timeout)
                return self.returncode
            def kill(self):
                self.returncode = -9
                self.done.set()
        process = HangingProcess()
        with patch("docker_executor.subprocess.Popen", return_value=process):
            result = docker._run_docker(["start", "container"], timeout=0.01)
        self.assertTrue(result.timed_out)
        self.assertEqual(process.returncode, -9)

    def test_latest_image_and_invalid_limits_are_rejected(self):
        for config in ({"image": "python:latest"}, {"image": "python"}, {"invocation_timeout": 0}, {"memory_mb": 0}):
            with self.subTest(config=config):
                self.assertRaises(ValueError, docker.DockerConfig, **config)


if __name__ == "__main__":
    unittest.main()

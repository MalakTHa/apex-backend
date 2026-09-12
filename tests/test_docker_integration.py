"""Real Linux-container tests. Explicit skips when Docker/image is unavailable.

The strings below are sent to Docker stdin, never evaluated on the host.
No Groq calls, application secrets, host mounts, or application API changes.
"""
import os
import unittest
from unittest.mock import patch

from docker_executor import DockerConfig, DockerExecutor
from verification_service import verify_with_docker


class DockerIsolationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = DockerExecutor()
        availability = probe.availability()
        cls.skip_reason = None
        if not availability.available:
            cls.skip_reason = f"Docker integration skipped: {availability.error_code}: {availability.reason}"
        else:
            try:
                if probe._image_id() is None:
                    cls.skip_reason = "Docker runtime image missing; build backend/runtime first (see README)."
            except OSError:
                cls.skip_reason = "Docker became unavailable while checking runtime image."

    def setUp(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)
        self.executor = DockerExecutor(DockerConfig(invocation_timeout=10))
        self.addCleanup(self.executor.retry_cleanup)

    def execute(self, body, args=None, name="f", source=None):
        code = source if source is not None else "def f(value):\n" + "\n".join("    " + line for line in body.splitlines())
        return self.executor.execute_function(code, name, [1] if args is None else args, {})

    def assert_removed(self):
        self.assertEqual(self.executor.pending_cleanup, ())
        name = self.executor.last_container_name
        self.assertIsNotNone(name)
        reply = self.executor._call(["container", "ls", "--all", "--filter", f"name=^/{name}$", "--format", "{{.ID}}"])
        self.assertEqual(reply.returncode, 0)
        self.assertFalse(reply.stdout.strip(), "Invocation container must be removed")

    def test_return_and_cleanup_after_success(self):
        result = self.execute("return value + 1")
        self.assertEqual((result.status, result.return_value), ("returned", 2))
        self.assert_removed()

    def test_exception_without_traceback(self):
        result = self.execute("raise ValueError('example failure')")
        self.assertEqual(result.status, "raised")
        self.assertEqual(result.exception_type, "builtins.ValueError")
        self.assertEqual(result.exception_message, "example failure")
        self.assertNotIn("Traceback", str(result))
        self.assert_removed()

    def test_infinite_loop_times_out_and_container_is_removed(self):
        self.executor = DockerExecutor(DockerConfig(invocation_timeout=2))
        self.addCleanup(self.executor.retry_cleanup)
        result = self.execute("while True:\n    pass")
        self.assertEqual(result.status, "timeout")
        self.assert_removed()

    def test_invalid_code_and_cleanup_after_error(self):
        result = self.execute("", source="def f(value):\n    return (")
        self.assertEqual((result.status, result.error_code), ("error", "invalid_function_code"))
        self.assert_removed()

    def test_function_not_found(self):
        result = self.execute("return value", name="missing")
        self.assertEqual((result.status, result.error_code), ("error", "function_not_found"))
        self.assert_removed()

    def test_unsupported_serialization(self):
        result = self.execute("return {1, 2, 3}")
        self.assertEqual(result.status, "serialization_error")
        self.assert_removed()

    def test_tuple_and_nested_values_preserve_types(self):
        result = self.execute("return {'value': (True, value, [1.25, None, 'text'])}")
        self.assertEqual(result.status, "returned")
        self.assertEqual(result.return_value, {"value": (True, 1, [1.25, None, "text"])})

    def test_stdout_stderr_and_direct_descriptor_writes_do_not_pollute_ipc(self):
        result = self.execute("import os, sys\nprint('not JSON')\nprint('stderr', file=sys.stderr)\nos.write(1, b'raw stdout')\nos.write(2, b'raw stderr')\nreturn value")
        self.assertEqual((result.status, result.return_value), ("returned", 1))

    def test_network_access_is_blocked(self):
        result = self.execute("import socket\nsock = socket.socket()\nsock.settimeout(0.3)\ntry:\n    sock.connect(('1.1.1.1', 443))\n    return False\nexcept OSError:\n    return True\nfinally:\n    sock.close()")
        self.assertEqual((result.status, result.return_value), ("returned", True))

    def test_root_filesystem_is_read_only(self):
        result = self.execute("import errno\ntry:\n    with open('/tmp/apex-write-probe', 'w') as handle:\n        handle.write('probe')\n    return False\nexcept OSError as error:\n    return error.errno == errno.EROFS")
        self.assertEqual((result.status, result.return_value), ("returned", True))

    def test_secret_environment_is_not_inherited(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "synthetic-isolation-secret"}):
            result = self.execute("import os\nreturn 'GROQ_API_KEY' not in os.environ")
        self.assertEqual((result.status, result.return_value), ("returned", True))

    def test_non_root_no_privileges_and_seccomp(self):
        result = self.execute("import os\nwith open('/proc/self/status') as handle:\n    fields = dict(line.split(':', 1) for line in handle if ':' in line)\nreturn [os.getuid(), fields['NoNewPrivs'].strip(), fields['Seccomp'].strip(), fields['CapEff'].strip()]")
        self.assertEqual(result.status, "returned")
        self.assertEqual(result.return_value, [65534, "1", "2", "0000000000000000"])

    def test_no_project_or_docker_socket_mount(self):
        result = self.execute("import os\nreturn [os.path.exists('/var/run/docker.sock'), os.path.exists('/host'), os.path.exists('/opt/apex-runtime/main.py'), os.path.exists('/opt/apex-runtime/.env')]")
        self.assertEqual((result.status, result.return_value), ("returned", [False, False, False, False]))

    def test_process_limit_blocks_bounded_fork_attempt(self):
        result = self.execute("import os, signal\nchildren = []\ntry:\n    for index in range(128):\n        try:\n            pid = os.fork()\n        except OSError:\n            return {'blocked': True, 'created': len(children)}\n        if pid == 0:\n            while True:\n                signal.pause()\n        children.append(pid)\n    return {'blocked': False, 'created': len(children)}\nfinally:\n    for pid in children:\n        os.kill(pid, signal.SIGKILL)\n        os.waitpid(pid, 0)")
        self.assertEqual(result.status, "returned")
        self.assertTrue(result.return_value["blocked"])
        self.assertGreater(result.return_value["created"], 0)
        self.assertLess(result.return_value["created"], self.executor.config.pids_limit)
        self.assert_removed()

    def test_worker_process_exit_still_cleans_container(self):
        result = self.execute("import os\nos._exit(7)")
        self.assertEqual(result.status, "error")
        self.assert_removed()

    def test_excessive_return_value_is_serialization_failure(self):
        result = self.execute("return 'x' * 1000000")
        self.assertEqual(result.status, "serialization_error")

    def test_fresh_container_per_invocation(self):
        self.execute("return value")
        first = self.executor.last_container_name
        self.assert_removed()
        self.execute("return value")
        self.assertNotEqual(first, self.executor.last_container_name)
        self.assert_removed()

    def test_internal_verification_service_with_real_containers(self):
        original = "def f(value: int):\n    return value + 1"
        candidate = "def f(value: int):\n    return 1 + value"
        result = verify_with_docker(original, candidate, "f", config=DockerConfig(invocation_timeout=10))
        self.assertEqual(result.verification_status, "verified")
        self.assertEqual(result.tests_passed, 6)


if __name__ == "__main__":
    unittest.main()

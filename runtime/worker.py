"""Fixed worker entrypoint. User source is executed ONLY inside the runtime image.

Never import/call this worker to execute source on the host. stdout/stderr from
user code, including direct writes to file descriptors 1/2, are discarded.
The host treats all worker output as untrusted, bounded protocol data.
"""
import ast
import os
from pathlib import Path
import sys

# -I deliberately excludes the script directory; add only the immutable worker directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec import (MAX_CODE_BYTES, MAX_EXCEPTION_MESSAGE, MAX_WIRE_BYTES,
                   SerializationError, decode_value, dump_json, encode_value, load_json)
from measurement import measure


def _message(error):
    text = str(error)
    # No traceback, control sequences or unbounded messages in IPC.
    if len(text) > MAX_EXCEPTION_MESSAGE:
        raise SerializationError("Exception message exceeds limits")
    return "".join(character for character in text if character.isprintable() or character in "\n\t")


def run_request(payload):
    if type(payload) is not dict:
        return {"status": "error", "error_code": "invalid_request"}
    operation = payload.get("operation", "execute")
    fields = {"function_code", "function_name", "args", "kwargs"}
    if "operation" in payload:
        fields.add("operation")
    if operation == "benchmark":
        fields.update({"warmup_runs", "measurement_runs"})
    if operation not in {"execute", "benchmark"} or set(payload) != fields:
        return {"status": "error", "error_code": "invalid_request"}
    if operation == "benchmark" and not (
            type(payload["warmup_runs"]) is int and 0 <= payload["warmup_runs"] <= 20
            and type(payload["measurement_runs"]) is int and 3 <= payload["measurement_runs"] <= 100):
        return {"status": "error", "error_code": "invalid_request"}
    code, name = payload["function_code"], payload["function_name"]
    if type(code) is not str or type(name) is not str or not name.isidentifier() or len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return {"status": "error", "error_code": "invalid_request"}
    try:
        args, kwargs = decode_value(payload["args"]), decode_value(payload["kwargs"])
        if type(args) is not list or type(kwargs) is not dict:
            raise SerializationError("Invalid arguments")
    except SerializationError:
        return {"status": "error", "error_code": "invalid_request"}
    try:
        tree = ast.parse(code, filename="<apex-function>")
    except (SyntaxError, ValueError, RecursionError):
        return {"status": "error", "error_code": "invalid_function_code"}
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return {"status": "error", "error_code": "invalid_function_code"}
    if tree.body[0].name != name:
        return {"status": "error", "error_code": "function_not_found"}
    namespace = {"__name__": "__apex_candidate__"}
    try:
        # This is the ONLY user-source execution site; the Docker image owns it.
        exec(compile(tree, "<apex-function>", "exec"), namespace)
        if operation == "benchmark":
            value = measure(namespace[name],
                            lambda: (decode_value(payload["args"]), decode_value(payload["kwargs"])),
                            payload["warmup_runs"], payload["measurement_runs"])
        else:
            value = namespace[name](*args, **kwargs)
    except BaseException as error:
        try:
            error_type = type(error)
            qualified_name = f"{error_type.__module__}.{error_type.__qualname__}"
            if len(qualified_name) > 256:
                raise SerializationError("Exception type exceeds limit")
            state = {"post_args": encode_value(args), "post_kwargs": encode_value(kwargs)} if operation == "execute" else {}
            return {"status": "raised", "exception_type": qualified_name,
                    "exception_message": _message(error), **state}
        except BaseException:
            return {"status": "serialization_error"}
    try:
        state = {"post_args": encode_value(args), "post_kwargs": encode_value(kwargs)} if operation == "execute" else {}
        return {"status": "returned", "return_value": encode_value(value), **state}
    except (SerializationError, RecursionError):
        return {"status": "serialization_error"}


def main():
    # Defense against accidentally running this entrypoint on a developer host.
    if os.name != "posix" or not Path("/.dockerenv").exists() or os.getuid() == 0:
        sys.stdout.buffer.write(dump_json({"status": "error", "error_code": "worker_environment"}))
        return
    data = sys.stdin.buffer.read(MAX_WIRE_BYTES + 1)
    protocol_fd = os.dup(1)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    os.close(sink)
    try:
        try:
            response = run_request(load_json(data))
            encoded = dump_json(response)
        except BaseException:
            encoded = dump_json({"status": "serialization_error"})
        # Use a private descriptor so ordinary print/os.write(1/2) cannot pollute JSON.
        while encoded:
            encoded = encoded[os.write(protocol_fd, encoded):]
    finally:
        os.close(protocol_fd)


if __name__ == "__main__":
    main()

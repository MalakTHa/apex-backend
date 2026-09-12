import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError

import ai_optimizer as ai


SOURCE = "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n"
CANDIDATE = "def total(values):\n    return sum(values, 0)\n"


class AIOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GROQ_API_KEY": "unit-test-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dotenv = patch.object(ai, "load_dotenv")
        self.loader = self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.groq = patch.object(ai, "Groq")
        self.factory = self.groq.start()
        self.addCleanup(self.groq.stop)
        self.create = self.factory.return_value.__enter__.return_value.chat.completions.create

    def respond(self, code=CANDIDATE, can_optimize=True, **extra):
        content = json.dumps({
            "can_optimize": can_optimize,
            "optimized_function_code": code,
            "reason": "Candidate only", **extra,
        })
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=content)
        )])

    def suggest(self, source=SOURCE):
        return ai.suggest_ai_optimization(source, {"function": "total"})

    def test_missing_api_key(self):
        os.environ.pop("GROQ_API_KEY")
        self.assertEqual(self.suggest()["error_code"], "missing_api_key")
        self.factory.assert_not_called()

    def test_valid_structured_response(self):
        self.respond()
        result = self.suggest()
        self.assertEqual(result["status"], "candidate")
        self.assertTrue(result["can_optimize"])
        self.assertEqual(result["optimized_function_code"], CANDIDATE)
        original = ast.parse(SOURCE).body[0]
        proposed = ast.parse(result["optimized_function_code"]).body[0]
        self.assertEqual(proposed.name, original.name)
        self.assertEqual(ast.dump(proposed.args), ast.dump(original.args))
        request = self.create.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-oss-120b")
        self.assertEqual(request["temperature"], 0)
        schema = request["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["properties"]), {
            "can_optimize", "optimized_function_code", "reason"
        })
        self.assertEqual(json.loads(request["messages"][1]["content"])["function_code"], SOURCE)
        self.loader.assert_called_once_with(dotenv_path=ai.ENV_PATH, override=False)

    def test_model_override(self):
        os.environ["GROQ_MODEL"] = "configured-model"
        self.respond()
        self.suggest()
        self.assertEqual(self.create.call_args.kwargs["model"], "configured-model")

    def test_dotenv_loading_and_environment_precedence(self):
        self.loader.side_effect = load_dotenv
        self.respond()
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("GROQ_API_KEY=fake-local-key\nGROQ_MODEL=local-model\n", encoding="utf-8")
            with patch.object(ai, "ENV_PATH", env_path):
                self.suggest()
                self.assertEqual(self.factory.call_args.kwargs["api_key"], "unit-test-key")
                self.assertEqual(self.create.call_args.kwargs["model"], "local-model")
                os.environ.pop("GROQ_API_KEY")
                self.suggest()
                self.assertEqual(self.factory.call_args.kwargs["api_key"], "fake-local-key")

    def test_real_sdk_with_mock_http_transport(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])
            self.assertNotIn("tools", payload)
            return httpx.Response(200, json={
                "id": "mock-completion", "object": "chat.completion",
                "created": 0, "model": ai.DEFAULT_MODEL,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps({
                        "can_optimize": True, "optimized_function_code": CANDIDATE,
                        "reason": "Candidate only",
                    }),
                }}],
            })

        self.factory.side_effect = lambda **kwargs: Groq(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        self.assertEqual(self.suggest()["status"], "candidate")

    def test_declined(self):
        self.respond(code="", can_optimize=False)
        result = self.suggest()
        self.assertEqual(result["status"], "declined")
        self.assertFalse(result["can_optimize"])
        self.assertEqual(result["optimized_function_code"], "")

    def test_invalid_python(self):
        self.respond("def total(values):\n    return (")
        self.assertEqual(self.suggest()["error_code"], "invalid_python")

    def test_changed_name(self):
        self.respond(CANDIDATE.replace("total", "renamed"))
        self.assertEqual(self.suggest()["error_code"], "function_name_changed")

    def test_changed_parameters(self):
        for parameters in ("items", "values, extra", "*, values", "values=()", "values: list", "values, /", "*values", "**values"):
            with self.subTest(parameters=parameters):
                self.respond(f"def total({parameters}):\n    return 0")
                self.assertEqual(self.suggest()["error_code"], "function_parameters_changed")

    def test_parameter_order(self):
        self.respond("def total(b, a):\n    return a + b")
        self.assertEqual(self.suggest("def total(a, b):\n    return a + b")["error_code"], "function_parameters_changed")

    def test_changed_function_api(self):
        for code in ("async " + CANDIDATE, "@decorator\n" + CANDIDATE,
                     "def total(values) -> int:\n    return 0"):
            with self.subTest(code=code):
                self.respond(code)
                self.assertEqual(self.suggest()["error_code"], "function_api_changed")

    def test_missing_code(self):
        self.respond("  ")
        self.assertEqual(self.suggest()["error_code"], "missing_code")

    def test_extra_top_level_code(self):
        self.respond("import os\n" + CANDIDATE)
        self.assertEqual(self.suggest()["error_code"], "invalid_function_structure")

    def test_invalid_json_and_schema(self):
        for content in ("not JSON", "```json\n{}\n```", "{}", "null", "", None,
                        '{"can_optimize":"true","optimized_function_code":"","reason":""}'):
            with self.subTest(content=content):
                self.respond()
                self.create.return_value.choices[0].message.content = content
                self.assertEqual(self.suggest()["error_code"], "invalid_response")
        self.respond(time_complexity="O(1)")
        self.assertEqual(self.suggest()["error_code"], "invalid_response")

    def test_empty_choices_and_truncation(self):
        self.create.return_value = SimpleNamespace(choices=[])
        self.assertEqual(self.suggest()["error_code"], "invalid_response")
        self.respond()
        self.create.return_value.choices[0].finish_reason = "length"
        self.assertEqual(self.suggest()["error_code"], "incomplete_response")

    def test_invalid_input_does_not_call_api(self):
        self.assertEqual(self.suggest("print('not a function')")["error_code"], "invalid_input")
        self.create.assert_not_called()

    def test_api_errors_are_sanitized(self):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        secret = "unit-test-key"
        errors = [
            (APIConnectionError(message=secret, request=request), "api_unavailable"),
            (APITimeoutError(request=request), "timeout"),
            (RateLimitError(secret, response=httpx.Response(429, request=request), body={}), "rate_limited"),
            (APIStatusError(secret, response=httpx.Response(500, request=request), body={}), "api_error"),
            (RuntimeError(secret), "request_failed"),
        ]
        for exception, expected in errors:
            with self.subTest(expected=expected):
                self.create.side_effect = exception
                result = self.suggest()
                self.assertEqual(result["error_code"], expected)
                self.assertNotIn(secret, json.dumps(result))


if __name__ == "__main__":
    unittest.main()

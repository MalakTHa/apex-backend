"""Deterministic, bounded input generation. No code or annotations are executed."""

import ast
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Literal
from verification_input_profile import resolve_input_profiles, unsupported_effects


@dataclass(frozen=True)
class TestCase:
    case_id: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class InputGenerationResult:
    status: Literal["supported", "unsupported"]
    reason: str
    cases: tuple[TestCase, ...] = ()



def _values(type_name: str) -> list:
    # At most six examples per parameter, hence at most 36 cases per function.
    return {
        "list[int] | list[str]": [[], [1, 1], [3, -1, 2], [""], ["a", "a"], ["b", "A", "a"]],
        "int": [0, 1, -1, 2, -2, 10],
        "str": ["", "a", "aa", "a b", "Ab!", "\u0645\u0631\u062d\u0628\u0627"],
        "list[int]": [[], [0], [1, 1], [3, 1, 2], [-2, 0, 2], [1, 2, 1]],
        "list[str]": [[], [""], ["a", "a"], ["b", "a"], ["a b", ""], ["\u0645\u0631\u062d\u0628\u0627", "A"]],
    }[type_name]


def generate_test_cases(function_node: ast.FunctionDef) -> InputGenerationResult:
    """Support 1–2 explicitly annotated scalar/list parameters, including kw-only.

    Profiles use explicit annotations first, then conservative body evidence.
    No inference from parameter names, defaults or AI output.
    Defaults are always supplied explicitly; omission behavior is out of scope.
    Each case owns its mutable inputs; there are no cross-parameter aliases.
    """
    if not isinstance(function_node, ast.FunctionDef):
        return InputGenerationResult("unsupported", "Only synchronous functions are supported.")
    if function_node.decorator_list or getattr(function_node, "type_params", []):
        return InputGenerationResult("unsupported", "Decorated and generic functions are unsupported.")
    if any(isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)) for node in ast.walk(function_node)):
        return InputGenerationResult("unsupported", "Generator and asynchronous behavior is unsupported.")
    arguments = function_node.args
    if arguments.vararg or arguments.kwarg:
        return InputGenerationResult("unsupported", "Variadic parameters are unsupported.")
    positional = arguments.posonlyargs + arguments.args
    parameters = positional + arguments.kwonlyargs
    if not 1 <= len(parameters) <= 2:
        return InputGenerationResult("unsupported", "Exactly one or two parameters are required.")
    names = [parameter.arg for parameter in parameters]
    if len(set(names)) != len(names):
        return InputGenerationResult("unsupported", "Duplicate parameter names are unsupported.")
    effect = unsupported_effects(function_node)
    if effect:
        return InputGenerationResult("unsupported", effect)
    profiles = resolve_input_profiles(function_node, behavioral=True)
    types = [profile.inferred_type for profile in profiles]
    if any(type_name is None for type_name in types):
        return InputGenerationResult("unsupported", "APEX could not generate reliable behavioral test cases for this function signature.")
    cases = []
    for index, values in enumerate(product(*[_values(type_name) for type_name in types]), start=1):
        values = deepcopy(values)
        cases.append(TestCase(
            case_id=f"case_{index:02d}",
            args=list(values[:len(positional)]),
            kwargs={parameter.arg: value for parameter, value in zip(arguments.kwonlyargs, values[len(positional):])},
        ))
    return InputGenerationResult("supported", "Deterministic examples for the declared parameter types.", tuple(cases))

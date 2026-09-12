"""Bounded tagged JSON values shared by the host and the fixed container worker.

No pickle, object constructors, arbitrary equality, or code evaluation.
Tuple tags are preserved; the phase 3A comparator still treats tuples as unsupported.
"""
import json
import math
import re

MAX_WIRE_BYTES = 128 * 1024
MAX_CODE_BYTES = 64 * 1024
MAX_DEPTH = 32
MAX_NODES = 10000
MAX_STRING_LENGTH = 65536
MAX_INT_DIGITS = 4096
MAX_EXCEPTION_MESSAGE = 1024


class SerializationError(ValueError):
    pass


def _step(depth, budget):
    budget[0] -= 1
    if depth > MAX_DEPTH or budget[0] < 0:
        raise SerializationError("Value exceeds structural limits")


def encode_value(value, depth=0, budget=None):
    budget = [MAX_NODES] if budget is None else budget
    _step(depth, budget)
    kind = type(value)
    if value is None:
        return {"t": "null"}
    if kind is bool:
        return {"t": "bool", "v": value}
    if kind is int:
        try:
            number = str(value)
        except ValueError as error:
            raise SerializationError("Integer exceeds limits") from error
        if len(number.lstrip("-")) > MAX_INT_DIGITS:
            raise SerializationError("Integer exceeds limits")
        return {"t": "int", "v": number}
    if kind is float and math.isfinite(value):
        return {"t": "float", "v": repr(value)}
    if kind is str and len(value) <= MAX_STRING_LENGTH:
        return {"t": "str", "v": value}
    if kind in (list, tuple):
        return {"t": kind.__name__, "v": [encode_value(item, depth + 1, budget) for item in value]}
    if kind is dict and all(type(key) is str and len(key) <= MAX_STRING_LENGTH for key in value):
        return {"t": "dict", "v": [[key, encode_value(item, depth + 1, budget)] for key, item in value.items()]}
    raise SerializationError("Unsupported value")


def decode_value(node, depth=0, budget=None):
    budget = [MAX_NODES] if budget is None else budget
    _step(depth, budget)
    if type(node) is not dict or type(node.get("t")) is not str:
        raise SerializationError("Invalid value tag")
    tag = node["t"]
    if tag == "null" and set(node) == {"t"}:
        return None
    if set(node) != {"t", "v"}:
        raise SerializationError("Invalid value fields")
    value = node["v"]
    if tag == "bool" and type(value) is bool:
        return value
    if tag == "str" and type(value) is str and len(value) <= MAX_STRING_LENGTH:
        return value
    if tag == "int" and type(value) is str and len(value.lstrip("-")) <= MAX_INT_DIGITS:
        if re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
            return int(value)
    if tag == "float" and type(value) is str and len(value) <= 32:
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except ValueError:
            pass
    if tag in {"list", "tuple"} and type(value) is list:
        items = [decode_value(item, depth + 1, budget) for item in value]
        return tuple(items) if tag == "tuple" else items
    if tag == "dict" and type(value) is list:
        result = {}
        for pair in value:
            if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str
                    or len(pair[0]) > MAX_STRING_LENGTH or pair[0] in result):
                raise SerializationError("Invalid dictionary entry")
            result[pair[0]] = decode_value(pair[1], depth + 1, budget)
        return result
    raise SerializationError("Invalid tagged value")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError("Duplicate JSON key")
        result[key] = value
    return result


def load_json(data):
    if type(data) is not bytes or len(data) > MAX_WIRE_BYTES:
        raise SerializationError("Payload exceeds limit")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(SerializationError("Non-finite JSON")))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise SerializationError("Invalid bounded JSON") from error


def dump_json(value):
    try:
        data = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as error:
        raise SerializationError("Invalid JSON value") from error
    if len(data) > MAX_WIRE_BYTES:
        raise SerializationError("Payload exceeds limit")
    return data

"""Conservative static profiles: no execution or AI inference."""
import ast
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class InputProfile:
    parameter: str
    inferred_type: str | None
    source: str
    confidence: str
    evidence: tuple[str, ...]
    supported: bool

    def to_dict(self):
        return asdict(self)

def annotation_type(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except (SyntaxError, ValueError, RecursionError):
            return None
    if isinstance(node, ast.Name) and node.id in {"int", "str"}:
        return node.id
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "list" and isinstance(node.slice, ast.Name)
            and node.slice.id in {"int", "str"}):
        return "list[" + node.slice.id + "]"
    return None

SAFE_CALLS = {"len", "range", "sum", "min", "max", "abs", "sorted", "enumerate", "zip",
              "list", "dict", "set", "tuple", "int", "str", "bool", "all", "any", "reversed",
              "ValueError", "TypeError", "IndexError", "ZeroDivisionError"}
SAFE_METHODS = {"append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse", "copy",
                "count", "index", "get", "items", "keys", "values", "add", "discard",
                "lower", "upper", "strip", "lstrip", "rstrip", "split", "join", "replace",
                "startswith", "endswith", "find", "isdigit", "isalpha"}

def unsupported_effects(node):
    """Unknown dependencies cannot establish supported effect coverage."""
    parameters = {p.arg for p in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
    local = parameters | {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for child in ast.walk(node):
        if isinstance(child, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith,
                              ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef)) or (isinstance(child, ast.FunctionDef) and child is not node):
            return "External dependencies, callbacks and nonlocal side effects are outside the supported verification scope."
        if isinstance(child, ast.Call):
            fn = child.func
            if isinstance(fn, ast.Name):
                if fn.id not in SAFE_CALLS or fn.id in local:
                    return "Unknown calls or callbacks are outside the supported verification scope."
            elif isinstance(fn, ast.Attribute):
                if fn.attr not in SAFE_METHODS or not isinstance(fn.value, ast.Name) or fn.value.id not in local:
                    return "External methods and custom objects are outside the supported verification scope."
            else:
                return "Dynamic calls are outside the supported verification scope."
        if isinstance(child, ast.Attribute) and child.attr not in SAFE_METHODS:
            return "Custom object attributes are outside the supported verification scope."
    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id not in local | SAFE_CALLS:
                return "Global dependencies are outside the supported verification scope."
    return None

def resolve_input_profiles(node):
    profiles = []
    for parameter in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        if parameter.annotation is not None:
            kind = annotation_type(parameter.annotation)
            profiles.append(InputProfile(parameter.arg, kind, "annotation", "explicit", ("Explicit parameter annotation",), kind is not None))
            continue
        name = parameter.arg
        types, evidence = set(), []
        def is_param(value):
            return isinstance(value, ast.Name) and value.id == name
        def numeric(value):
            return isinstance(value, ast.Constant) and type(value.value) is int
        elements = {n.target.id for n in ast.walk(node) if isinstance(n, ast.For) and is_param(n.iter) and isinstance(n.target, ast.Name)}
        def element(value):
            return (isinstance(value, ast.Subscript) and is_param(value.value) and not isinstance(value.slice, ast.Slice)) or (isinstance(value, ast.Name) and value.id in elements)
        for child in ast.walk(node):
            if isinstance(child, ast.Compare) or (isinstance(child, ast.BinOp) and isinstance(child.op, (ast.Add, ast.Sub, ast.FloorDiv, ast.Mod))):
                values = [child.left, child.right] if isinstance(child, ast.BinOp) else [child.left, *child.comparators]
                if any(numeric(v) for v in values):
                    if any(is_param(v) for v in values):
                        types.add("int"); evidence.append("Parameter used with an integer literal")
                    if any(element(v) for v in values):
                        types.add("list[int]"); evidence.append("Indexed or iterated element used with an integer literal")
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and is_param(child.func.value):
                if child.func.attr in {"lower", "upper", "strip", "lstrip", "rstrip", "split", "startswith", "endswith", "isdigit", "isalpha"}:
                    types.add("str"); evidence.append("String-specific method: " + child.func.attr)
        collection_use = any((isinstance(n, ast.Subscript) and is_param(n.value))
                             or (isinstance(n, ast.For) and is_param(n.iter))
                             or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" and any(is_param(a) for a in n.args))
                             for n in ast.walk(node))
        if collection_use and "int" in types:
            types.add("conflicting_collection")
        rebound = any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store) for n in ast.walk(node))
        kind = next(iter(types)) if len(types) == 1 and not rebound else None
        profiles.append(InputProfile(name, kind, "ast_inference", "conservative" if kind else "insufficient", tuple(sorted(set(evidence))) or ("No unambiguous type evidence",), kind is not None))
    return profiles

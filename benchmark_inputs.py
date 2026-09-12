"""Bounded performance workloads, separate from behavioral verification cases."""
import ast
from verification_input_profile import resolve_input_profiles
from verification_inputs import InputGenerationResult, TestCase
from runtime.codec import encode_value, dump_json, SerializationError

def workload_sizes(complexity):
    text = (complexity or '').lower().replace(' ', '')
    if any(t in text for t in ('2^', '2**', 'exponential', 'n!', 'factorial')):
        return (2, 4, 8)
    if any(t in text for t in ('n^4', 'n⁴', 'n^3', 'n³')):
        return (5, 10, 20)
    if any(t in text for t in ('n^2', 'n²', 'n*m')):
        return (50, 150, 300)
    if 'log' in text:
        return (100, 1000, 3000)
    if text in {'o(1)', 'o(n)'}:
        return (100, 1000, 5000)
    return (10, 30, 100)

def generate_benchmark_cases(function_node):
    from main import analyze_function_node
    if not isinstance(function_node, ast.FunctionDef):
        return InputGenerationResult('unsupported', 'Only standalone synchronous functions have benchmark workloads.')
    if function_node.args.vararg or function_node.args.kwarg or function_node.decorator_list:
        return InputGenerationResult('unsupported', 'No supported performance workload for this API.')
    positional = function_node.args.posonlyargs + function_node.args.args
    parameters = positional + function_node.args.kwonlyargs
    if not 1 <= len(parameters) <= 2:
        return InputGenerationResult('unsupported', 'No supported performance workload for this API.')
    profiles = resolve_input_profiles(function_node)
    kinds = []
    for parameter, profile in zip(parameters, profiles):
        kind = profile.inferred_type
        if kind is None and parameter.annotation is None:
            # Representative workload selection is not a claim of behavioral proof.
            name = parameter.arg
            def named(n):
                return isinstance(n, ast.Name) and n.id == name
            if any((isinstance(n, ast.Subscript) and named(n.value)) or (isinstance(n, ast.For) and named(n.iter))
                   or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'len' and any(named(a) for a in n.args)) for n in ast.walk(function_node)):
                kind = 'list[int]'
            elif any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'range' and any(named(a) for a in n.args) for n in ast.walk(function_node)):
                kind = 'int'
        if kind not in {'int', 'str', 'list[int]', 'list[str]'}:
            return InputGenerationResult('unsupported', 'No supported performance workload for this API.')
        kinds.append(kind)
    complexity = analyze_function_node(function_node.name, function_node).get('time_complexity')
    sizes = workload_sizes(complexity)
    # Workload safety only: the analyzer intentionally remains unchanged.
    # Comprehension loops can be absent from its published loop count.
    def loop_depth(node):
        weight = 1 if isinstance(node, (ast.For, ast.While)) else len(node.generators) if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) else 0
        return weight + max((loop_depth(child) for child in ast.iter_child_nodes(node)), default=0)
    depth = loop_depth(function_node)
    if depth >= 3:
        sizes = tuple(min(n, cap) for n, cap in zip(sizes, (5, 10, 20)))
    elif depth >= 2:
        sizes = tuple(min(n, cap) for n, cap in zip(sizes, (50, 150, 300)))
    recursive = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == function_node.name for n in ast.walk(function_node))
    variable_exponent = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and not isinstance(n.right, ast.Constant) for n in ast.walk(function_node))
    if recursive or variable_exponent:
        sizes = tuple(min(n, cap) for n, cap in zip(sizes, (2, 4, 8)))
    cases = []
    for size in sizes:
        for shape in ('descending', 'mixed'):
            values = []
            for kind in kinds:
                numbers = list(range(size, 0, -1)) if shape == 'descending' else [(i * 37) % max(size, 1) for i in range(size)]
                if kind == 'int': value = size
                elif kind == 'str': value = ('ab ' * ((size + 2)//3))[:size]
                elif kind == 'list[int]': value = numbers
                else: value = [str(n) for n in numbers]
                values.append(value)
            args = values[:len(positional)]
            kwargs = {p.arg: v for p, v in zip(parameters[len(positional):], values[len(positional):])}
            try:
                dump_json({'function_code': ast.unparse(function_node), 'function_name': function_node.name,
                           'args': encode_value(args), 'kwargs': encode_value(kwargs),
                           'operation': 'benchmark', 'warmup_runs': 20, 'measurement_runs': 100})
            except (SerializationError, ValueError, RecursionError):
                continue
            cases.append(TestCase(f'performance_{size}_{shape}', args, kwargs))
    return InputGenerationResult('supported' if cases else 'unsupported', 'Bounded complexity-aware performance workloads.', tuple(cases))

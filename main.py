import ast
from platform import node
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math #جديد
import textwrap

app = FastAPI()#هنا انشأت متغير او اوبجكت من كلاس فاست اي بي اي 

app.add_middleware(
    CORSMiddleware,
    #يعتمد CORSMiddleware على قيمة Origin المرسلة مع الطلب من المتصفح، حيث يقوم بمقارنتها مع القيم المسموح بها في allow_origins، فإذا كانت متطابقة يسمح بالاتصال، وإذا لم تكن كذلك يقوم المتصفح بمنع الاستجابة.
    #allow_origins=["*"],
    allow_origins=["*"],
    allow_credentials=True,#باستخدامها بعدين لما اعمل تسجيل دخول وحفظ جلسات 
    allow_methods=["*"],#هذا يسمح بكل انوع الطلبات وقد اغيره لاني استخدم بس pos
    allow_headers=["*"],
)

# قاموس لتعقيد العمليات المدمجة في بايثون
BUILTIN_COMPLEXITIES = {
    "len": "O(1)",
    "append": "O(1)",
    "pop": "O(1)", # O(1) for last element, O(n) otherwise
    "insert": "O(n)",
    "remove": "O(n)",
    "extend": "O(n)",
    "copy": "O(n)",
    "reverse": "O(n)",
    "count": "O(n)",
    "index": "O(n)",
    "sorted": "O(n log n)",
    "sort": "O(n log n)",
    "min": "O(n)",
    "max": "O(n)",
    "sum": "O(n)",
    "join": "O(n)",
    "split": "O(n)",
}

# انشاء ملاس يرث من البيسمودل بحيث يحول البيانات المرسلة كجيسون الى بيانات من نوع اوبجكت 
class CodeInput(BaseModel):#هنا الي خارج القوس يرث من الي داخل 
    code: str
    function_name: str
#ملاحضة اذا شفت متغير قد يكون كلاس يمرر في اقواس كلاس اخر عند انشاءه هذا يعني اني اعمل الكلاس الثاني يرث من الكلاس الذي داخل الاقواس

class FunctionAnalyzer(ast.NodeVisitor):#يعني هنا الكلاس FunctionAnalyzer يرث من  ast.NodeVisitor
    def __init__(self):

        self.list_membership_inside_loop = False
        self.has_divide_and_conquer = False
        self.loops = 0
        self.nested_loops = 0
        self.current_depth = 0
        self.max_depth = 0
        self.recursive = False
        self.self_recursive_calls = 0
        self.function_name = ""

        self.has_nested_loops = False
        self.has_swap = False
        self.has_append = False
        self.has_sort_call = False
        self.has_list_membership = False
        self.has_string_concat = False
        self.has_dict_usage = False
        self.has_set_usage = False
        
        # New fields for better accuracy
        self.local_functions = {} # Stores complexity of nested functions
        self.variable_types = {} # NEW: Tracks variable types (set, dict, etc.)
        self.has_logarithmic_loop = False # NEW: Tracks if a loop divides data (Binary Search)
        self.max_effective_depth = 0
        self.current_call_depth = 0

    def visit_FunctionDef(self, node):
        if not self.function_name:
            self.function_name = node.name
            self.generic_visit(node)
        else:
            # This is a nested function
            nested_analyzer = FunctionAnalyzer()
            nested_analyzer.visit(node)
            # Store the full analyzer results to propagate later in visit_Call
            self.local_functions[node.name] = nested_analyzer
            # Do NOT call generic_visit(node) here to avoid double-counting the body

    def visit_For(self, node):
        self.loops += 1
        self.current_depth += 1

        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth

        if self.current_depth > 1:
            self.nested_loops += 1
            self.has_nested_loops = True

        self.generic_visit(node)
        self.current_depth -= 1

    def visit_While(self, node):
        self.loops += 1
        self.current_depth += 1

        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth

        if self.current_depth > 1:
            self.nested_loops += 1
            self.has_nested_loops = True

        # Check if this while loop looks like a Binary Search (dividing the range)
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.FloorDiv, ast.RShift)):
                if isinstance(child.right, ast.Constant) and child.right.value == 2:
                    self.has_logarithmic_loop = True

        self.generic_visit(node)
        self.current_depth -= 1

    def visit_Call(self, node):
             
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == self.function_name:#هذا الشرط لجل اكتشاف الاستدعاء الذاتي داخل الدالة نفسها
                self.recursive = True
                self.self_recursive_calls += 1
                
                for arg in node.args:
                    if isinstance(arg, ast.Subscript):#هنا  لجل اتاكد ان في الباراميتر للدالة الداخلية اقواس مربعة والتي قد تكون سلايس او لا
                        if isinstance(arg.slice, ast.Slice):#لجل اتاكد ان الاقواس المربعة فيها سلايس لجل يتحقق شرط ال ديفايد اند كونكر
                            self.has_divide_and_conquer = True
            
            # Check for nested function calls
            if func_name in self.local_functions:
                child = self.local_functions[func_name]
                # Propagate depth
                added_depth = child.max_depth
                effective_depth = self.current_depth + added_depth
                if effective_depth > self.max_depth:
                    self.max_depth = effective_depth
                
                # Propagate flags
                if child.recursive:
                    self.recursive = True
                    self.self_recursive_calls += child.self_recursive_calls
                if child.has_nested_loops: self.has_nested_loops = True
                if child.has_swap: self.has_swap = True
                if child.has_append: self.has_append = True
                if child.has_sort_call: self.has_sort_call = True
                if child.has_list_membership: self.has_list_membership = True
                if child.has_string_concat: self.has_string_concat = True
                if child.has_dict_usage: self.has_dict_usage = True
                if child.has_set_usage: self.has_set_usage = True
                if getattr(child, 'has_divide_and_conquer', False):
                    self.has_divide_and_conquer = True

            if func_name in {"sorted", "set", "dict"}:
                if func_name == "sorted":
                    self.has_sort_call = True
                if func_name == "set":
                    self.has_set_usage = True
                if func_name == "dict":
                    self.has_dict_usage = True
            
            # Check for built-in functions that add complexity
            if func_name in BUILTIN_COMPLEXITIES:
                comp = BUILTIN_COMPLEXITIES[func_name]
                added_depth = 0
                if comp == "O(n)": added_depth = 1
                elif comp == "O(n log n)": added_depth = 1 # Simplified
                
                effective_depth = self.current_depth + added_depth
                if effective_depth > self.max_depth:
                    self.max_depth = effective_depth

        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name == "append":
                self.has_append = True
            if attr_name == "sort":
                self.has_sort_call = True
            if attr_name == "add":
                self.has_set_usage = True
            
            # Check for built-in attributes that add complexity (like list.count)
            if attr_name in BUILTIN_COMPLEXITIES:
                comp = BUILTIN_COMPLEXITIES[attr_name]
                added_depth = 0
                if comp == "O(n)": added_depth = 1
                effective_depth = self.current_depth + added_depth
                if effective_depth > self.max_depth:
                    self.max_depth = effective_depth

        self.generic_visit(node)

    def visit_Subscript(self, node):
        # Check for slicing like arr[a:b] which is O(n)
        if isinstance(node.slice, ast.Slice):
            effective_depth = self.current_depth + 1
            if effective_depth > self.max_depth:
                self.max_depth = effective_depth
        
        self.generic_visit(node)

    def _is_string_concat(self, node):
        if not node.targets or not isinstance(node.value, ast.BinOp) or not isinstance(node.value.op, ast.Add):
            return False
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return False
        
        var_name = target.id
        is_target_str = self.variable_types.get(var_name) == "str"
        
        # Check if we are adding to the same variable: s = s + ... or s = ... + s
        mentions_target = False
        if isinstance(node.value.left, ast.Name) and node.value.left.id == var_name:
            mentions_target = True
        elif isinstance(node.value.right, ast.Name) and node.value.right.id == var_name:
            mentions_target = True

        if mentions_target:
            # If the variable is known to be a string, any addition to it is likely concat
            if is_target_str:
                return True
            # Otherwise, check if any of the sides is a string literal, f-string, or str() call
            for side in [node.value.left, node.value.right]:
                if isinstance(side, ast.Constant) and isinstance(side.value, str):
                    return True
                if isinstance(side, (ast.JoinedStr, ast.Call)):
                    if isinstance(side, ast.JoinedStr): return True
                    if isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "str":
                        return True
        return False

    def visit_Assign(self, node):
        if node.targets:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    # Track if variable is a dict
                    if isinstance(node.value, ast.Dict):
                        self.variable_types[var_name] = "dict"
                        self.has_dict_usage = True
                    # Track if variable is a set
                    elif isinstance(node.value, ast.Set):
                        self.variable_types[var_name] = "set"
                        self.has_set_usage = True
                    # Track if variable is a string (literal or f-string)
                    elif (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) or isinstance(node.value, ast.JoinedStr):
                        self.variable_types[var_name] = "str"
                    # Track if variable is created via call
                    elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        if node.value.func.id == "dict":
                            self.variable_types[var_name] = "dict"
                            self.has_dict_usage = True
                        elif node.value.func.id == "set":
                            self.variable_types[var_name] = "set"
                            self.has_set_usage = True
                        elif node.value.func.id == "str":
                            self.variable_types[var_name] = "str"

            # Handle tuple swapping
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                self.has_swap = True

            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                if self._is_string_concat(node):
                    self.has_string_concat = True

        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            # s += ...
            if self.variable_types.get(var_name) == "str":
                self.has_string_concat = True
            elif isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    self.has_string_concat = True
                elif isinstance(node.value, ast.JoinedStr):
                    self.has_string_concat = True
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "str":
                    self.has_string_concat = True

        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            if isinstance(op, (ast.In, ast.NotIn)):
            #   هنا نحدد نوع container
                if isinstance(node.comparators[0], ast.Name):#comparators[0] هو الطرف اليمين من المقارنة.
                    container_name = node.comparators[0].id

                # إذا كان المتغير معروفاً كـ dict أو set من خلال تتبعنا للـ Assignments
                    is_fast_container = self.variable_types.get(container_name) in {"dict", "set"}
                    
                    # إذا لم يكن معروفاً، نترك الأسماء القديمة كاحتياط (freq, seen)
                    if not is_fast_container and container_name not in {"freq", "seen"}:
                        self.has_list_membership = True
                else:
                    self.has_list_membership = True

        self.generic_visit(node)


def format_polynomial_complexity(depth):
    if depth <= 0:
        return "O(1)"
    if depth == 1:
        return "O(n)"
    if depth == 2:
        return "O(n²)"
    if depth == 3:
        return "O(n³)"
    if depth == 4:
        return "O(n⁴)"
    return f"O(n^{depth})"


def estimate_complexity(analyzer):# الان الباراميتر analyzer من نوع FunctionAnalyzer
    # 🔴 Hidden quadratic (string concatenation inside loop)
    if analyzer.has_string_concat and analyzer.loops > 0 and analyzer.max_depth == 1:
        return "O(n²)"

    # 🔴 Recursion
    if analyzer.recursive:
        # Memoization detection: Recursion + Dictionary usage usually implies O(n)
        if analyzer.has_dict_usage:
            return "O(n) [Memoized]"
            
        if getattr(analyzer, "has_divide_and_conquer", False):
            return "O(n log n)"

        if getattr(analyzer, "self_recursive_calls", 0) >= 2:
            return "O(2^n)"

        return "Recursive"

    # 🔴 Nested loops (dynamic handling)
    if analyzer.max_depth >= 2:
        # Sorting pattern (swap-based) فقط إذا عمق = 2
        if analyzer.has_swap and analyzer.max_depth == 2:
            return "O(n²)"

        # Membership بين قائمتين (n * m)
        if analyzer.has_list_membership and analyzer.max_depth == 2:
            return "O(n * m)"

        # باقي الحالات حسب العمق الحقيقي (n^k)
        return format_polynomial_complexity(analyzer.max_depth)

    if analyzer.list_membership_inside_loop and analyzer.loops > 0:
        return "O(n²)"

    # 🔴 Single loop
    if analyzer.loops == 1:
        if getattr(analyzer, "has_logarithmic_loop", False):
            return "O(log n)"
        return "O(n)"
    
    # 🔴 Logarithmic handling for multiple/nested loops
    if getattr(analyzer, "has_logarithmic_loop", False):
        if analyzer.max_depth == 2:
            return "O(n log n)"

    return "O(1)"


def estimate_space_complexity(analyzer):
    if analyzer.recursive:
        return "O(n)"

    if analyzer.has_append or analyzer.has_dict_usage or analyzer.has_set_usage:
        return "O(n)"

    return "O(1)"


def detect_loop_type(analyzer):
    if analyzer.max_depth >= 2:
        return "Nested"

    if analyzer.loops == 1:
        return "Single"

    return "None"


def detect_algorithm(analyzer):
    if analyzer.has_swap and analyzer.max_depth >= 2:
        return "Manual Swap-Based Sorting"

    if analyzer.recursive:
        return "Recursive Algorithm"

    if analyzer.has_dict_usage:
        return "Dictionary-Based Processing"

    if analyzer.has_set_usage:
        return "Set-Based Processing"

    if analyzer.has_sort_call:
        return "Sorting"

    if analyzer.max_depth >= 2:
        return "Nested Loop Pattern"

    if analyzer.loops == 1:
        if getattr(analyzer, "has_logarithmic_loop", False):
            return "Binary Search / Logarithmic Scan"
        return "Linear Scan"

    return "Unknown"


def get_warnings(analyzer):
    warnings = []

    if analyzer.max_depth >= 2:
        warnings.append("High time complexity detected (likely nested loops or expensive calls inside loops).")

    if analyzer.recursive:
        warnings.append(
            "Recursion detected. Repeated recursive calls may be expensive."
        )

    if analyzer.has_list_membership and analyzer.loops > 0:
        warnings.append("Membership checks inside loops may be inefficient with lists (O(n) inside loop).")

    if analyzer.has_string_concat and analyzer.loops > 0:
        warnings.append(
            "String concatenation inside loops may cause repeated memory allocation (O(n^2) behavior)."
        )

    if not warnings:
        warnings.append("No critical performance warnings detected.")

    return warnings


def get_suggestions(analyzer):
    suggestions = []

    if analyzer.has_swap and analyzer.max_depth >= 2:
        suggestions.append(
            "Replace manual swap-based sorting with a divide-and-conquer sorting algorithm."
        )

    if analyzer.max_depth >= 2:
        suggestions.append(
            "Consider replacing nested scans with set or dictionary based lookup."
        )

    if analyzer.has_list_membership and analyzer.loops > 0:
        suggestions.append("Precompute a set for faster membership checks.")

    if analyzer.has_string_concat and analyzer.loops > 0:
        suggestions.append(
            "Use join-based string construction instead of repeated concatenation."
        )

    if analyzer.recursive:
        suggestions.append(
            "Consider memoization if recursive calls repeat the same subproblems."
        )

    if not suggestions:
        suggestions.append("No major optimization opportunity detected.")

    return suggestions


def analyze_function_node(function_name, node):
    analyzer = FunctionAnalyzer()
    analyzer.visit(node)

    return {
        "function": function_name,
        "time_complexity": estimate_complexity(analyzer),
        "space_complexity": estimate_space_complexity(analyzer),
        "loops": analyzer.loops,
        "nested_loops": analyzer.nested_loops,
        "loop_type": detect_loop_type(analyzer),
        "max_depth": analyzer.max_depth,
        "recursion": analyzer.recursive,
        "algorithm": detect_algorithm(analyzer),
        "warnings": get_warnings(analyzer),
        "suggestions": get_suggestions(analyzer),
    }

def check_syntax(code):
    try:
        ast.parse(code)
        return None
    except SyntaxError as error:
        return f"Syntax Error: {error.msg} (line {error.lineno})"
    
def extract_selected_function_source(code, function_name):
    """Extracts the original source code of a function using AST range detection."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    node = find_function_node(tree, function_name)
    if not node:
        return None, None

    lines = code.splitlines()
    # node.lineno and node.end_lineno are 1-based
    function_lines = lines[node.lineno - 1 : node.end_lineno]
    function_code = "\n".join(function_lines)

    # Dedent to make it parseable as a standalone function
    return textwrap.dedent(function_code), node.lineno
       
def find_function_node(tree, function_name):
    """Finds a function node by name anywhere in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    return None

def extract_function_code(code, function_name):
    """Extracts function code using AST unparse for a clean version."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    node = find_function_node(tree, function_name)
    if node:
        return ast.unparse(node)

    return ""


def replace_function_code(full_code, function_name, new_function_code):
    """Replaces a function's source code while preserving its original indentation."""
    try:
        tree = ast.parse(full_code)
    except SyntaxError:
        return full_code

    node = find_function_node(tree, function_name)
    if not node:
        return full_code

    lines = full_code.splitlines()
    
    # Get indentation of the original function's first line
    first_line = lines[node.lineno - 1]
    indentation = first_line[: len(first_line) - len(first_line.lstrip())]
    
    # Indent the new code to match the original
    indented_new_code = textwrap.indent(new_function_code, indentation)
    
    # Replace the lines from node.lineno to node.end_lineno
    new_lines = lines[: node.lineno - 1] + [indented_new_code] + lines[node.end_lineno :]
    return "\n".join(new_lines)


@app.post("/analyze")
def analyze_code(input: CodeInput):
    function_code, start_line = extract_selected_function_source(
        input.code,
        input.function_name
    )

    if not function_code:
        return {"error": "Function not found"}

    try:
         #تعديل للhighLight
        tree = ast.parse(function_code)
    except SyntaxError as error:
        real_line = start_line + error.lineno - 1
        return {
        "error": f"Syntax Error: {error.msg} (line {real_line})",
        "error_line": real_line
    }
     #هنا في تكرك وذلك اني لن احتاج الى البحث في جسم الشجرة لاني اعلم الكود فعليا ويمكنني الاسناد طوالي كذا يعني node=tree.body[0]
    node = tree.body[0]
    if isinstance(node, ast.FunctionDef) and node.name == input.function_name:
        return analyze_function_node(input.function_name, node)
        

    return {"error": "Function not found"}


class SimpleOptimizationTransformer(ast.NodeTransformer):
    #NodeTransformer يمر على الشجرة ويستطيع تعديل العقد وإرجاع نسخة معدلة
    def __init__(self, target_function):
        self.target_function = target_function
        self.changes = []

    def visit_FunctionDef(self, node):
        # لنود هنا تمثل شجرة الكود الكامل لذلك نرسل له النود الهدف عند انشاء اوبجكت من هذا الكلاس السبب في انه ياخذ الكود كله هو انه بعدل فيه ويرجه نفس ماهو باستثناء الدالة التي تم تحسينها 
        #هذه الدالة التشغيلية لهذا الكلاس 
        if node.name != self.target_function:
            return node

        node = self.generic_visit(node)

        if self._looks_like_swap_sort(node):
            node = self._add_early_exit(node)
            self.changes.append(
                "Added early-exit flag to avoid unnecessary loop passes."
            )
            self.changes.append(
                "Preserved the original algorithm structure while improving best-case behavior."
            )
            return node

        if self._add_search_break(node):
            self.changes.append(
                "Added 'break' to exit the search loop immediately once the target is found."
            )
            self.changes.append(
                "Improved average-case performance by avoiding unnecessary iterations."
            )
            return node

        self.changes.append(
            "No safe simple optimization was detected for this function yet."
        )
        return node

    def _looks_like_swap_sort(self, node):
        return self._max_loop_depth(node) >= 2 and self._has_swap(node)

    def _max_loop_depth(self, node):
        max_depth = 0#هذا متغير على مستوى الدالة اي انه اذا خرج من الدلة يتصفر على عكس المتغيرات التي في اول كلاس كانت على مستوى  الكلاس 

        def walk(current_node, depth):
            #هذا يسمح للدالة الداخلية walk أن تعدّل المتغير الخارجي:
            nonlocal max_depth

            if isinstance(current_node, (ast.For, ast.While)):
                depth += 1
                max_depth = max(max_depth, depth)

            for child in ast.iter_child_nodes(current_node):
                walk(child, depth)

        walk(node, 0)
        return max_depth

    def _has_swap(self, node):
        #يمر على كل العقد داخل الدالة، ليس فقط الأبناء المباشرين.
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                if child.targets and isinstance(child.targets[0], ast.Tuple):
                    #if child.targets هذه بترجع ليست من التارجت لكن المغزا هنا مش ترجع ترو او فولس يل هل القيمة تعتبر فارغة أم لا؟
                    if isinstance(child.value, ast.Tuple):
                        return True

        return False

    def _add_early_exit(self, node):
        for statement in node.body:
            if isinstance(statement, ast.For) and self._contains_inner_swap_loop(
                statement
            ):
                statement.body.insert(
                    0,
                    ast.Assign(
                        targets=[ast.Name(id="swapped", ctx=ast.Store())],
                        value=ast.Constant(value=False),
                    ),
                )
                
                #هذه الدالة الثانية ستدخل داخل الحلقة وتبحث عن مكان الـ swap الحقيقي، وبعده تضيف:swapped = True
                self._insert_swapped_true(statement)

                statement.body.append(
                    ast.If(
                        test=ast.UnaryOp(#UnaryOp هذا يعني ان  الشرط الذي باضيفه بيكون احادي الاتجاه يعني مافيه طرفين مثل if i in x بل جهة واحدة مثل if true
                            op=ast.Not(), operand=ast.Name(id="swapped", ctx=ast.Load())
                        ),
                        body=[ast.Break()],
                        orelse=[],
                    )
                )

        return node

    def _contains_inner_swap_loop(self, node):#غالبا هي الدوارة الخارجية لانه لن ينفذ هذه الدالة الااذا كانت من نوع for 
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                if child is not node and self._has_swap(child):
                    #child is not nodeيعني لا تحسب الحلقة الخارجية نفسها. نريد حلقة داخلية فقط.
                    #وذلك لان دالة ال walk لانجيب بس ابناء النود بل الابناء المباشرين والغير مباشير اي حتى الاحفاد وكذلك تجيب النود نفسها 
                    return True

        return False

    def _insert_swapped_true(self, node):#هذه الدالة تبحث عن الswap الحقيقي في الدالة وتعمل تحته انا النتغير الذي انشاناه swapped يساوي true
        for child in ast.walk(node):
            if isinstance(child, ast.If):#لانه في الدالة الاصل قبل مايتم عمل التبديل بيكون فيه شرط ضروري 
                updated_body = []

                for item in child.body:
                    updated_body.append(item)

                    if isinstance(item, ast.Assign):
                        if item.targets and isinstance(item.targets[0], ast.Tuple):
                            if isinstance(item.value, ast.Tuple):
                                updated_body.append(
                                    ast.Assign(
                                        targets=[
                                            ast.Name(id="swapped", ctx=ast.Store())
                                        ],
                                        value=ast.Constant(value=True),
                                    )
                                )

                child.body = updated_body

    def _add_search_break(self, node):
        optimized = False
        # نبحث عن الحلقات التي تحتوي على شرط (if) وبداخلها إسناد لقيمة (Assign)
        # ولكن لا تحتوي على break أو return
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.For, ast.While)):
                for item in stmt.body:
                    if isinstance(item, ast.If):
                        # نتحقق أن جسم الـ if يحتوي على عملية إسناد بسيطة
                        has_assignment = any(isinstance(x, ast.Assign) for x in item.body)
                        # نتحقق أنه لا يوجد مخرج مسبق (break أو return)
                        has_exit = any(isinstance(x, (ast.Break, ast.Return)) for x in item.body)
                        # نتحقق أن العملية ليست إضافة لقائمة (append) لأننا قد نريد جمع كل العناصر
                        is_collecting = any(
                            isinstance(x, ast.Expr) and 
                            isinstance(x.value, ast.Call) and 
                            isinstance(x.value.func, ast.Attribute) and 
                            x.value.func.attr == "append" 
                            for x in item.body
                        )
                        
                        if has_assignment and not has_exit and not is_collecting:
                            item.body.append(ast.Break())
                            optimized = True
        return optimized


@app.post("/optimize/simple")
def optimize_simple(input: CodeInput):
    syntax_error = check_syntax(input.code)

    if syntax_error:
        return syntax_error

    tree = ast.parse(input.code)

    node = find_function_node(tree, input.function_name)

    if not node:
        return {"error": "Function not found"}

    before_analysis = analyze_function_node(input.function_name, node)

    transformer = SimpleOptimizationTransformer(input.function_name)
    optimized_tree = transformer.visit(tree)
    
    #هذه الدالة وضيفتها تعمل تنسيق وتكملة للبيانات الناقصة مثلا عدد ارقام الاسطر اذا كان في اسطر جديدة في الدالة المحسنة
    ast.fix_missing_locations(optimized_tree)

    full_code = ast.unparse(optimized_tree)
    optimized_function_code = extract_function_code(full_code, input.function_name)

    after_tree = ast.parse(full_code)
    after_node = find_function_node(after_tree, input.function_name)
    after_analysis = None
    
    if after_node:
        after_analysis = analyze_function_node(input.function_name, after_node)

    return {
        "function": input.function_name,
        "optimization_type": "simple",
        "optimized_function_code": optimized_function_code,
        "full_code": full_code,
        "changes": transformer.changes,
        "before": before_analysis,
        "after": after_analysis,
    }


class ReplacementPatternDetector(ast.NodeVisitor):
    def __init__(self):
        self.function_name = ""

        self.current_depth = 0
        self.max_depth = 0
        self.has_nested_loops = False

        self.has_index_compare = False
        self.returns_true_on_compare = False

        self.has_swap = False
        self.has_append = False
        self.has_membership = False
        self.has_string_concat = False
        self.recursive_call_count = 0

        self.has_dict_assignment = False
        self.has_counter_increment = False
        self.has_seen_set_pattern = False
        self.has_duplicate_return = False
        self.has_equality_compare = False
        self.has_not_in_compare = False
        
        self.has_slicing = False # NEW: To detect sorting patterns
        self.dict_names = set() # اضفته جديد

    def visit_FunctionDef(self, node):
        self.function_name = node.name
        self.generic_visit(node)

    def visit_For(self, node):
        self.current_depth += 1

        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth

        if self.current_depth > 1:
            self.has_nested_loops = True

        self.generic_visit(node)
        self.current_depth -= 1

    def visit_While(self, node):
        self.current_depth += 1

        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth

        if self.current_depth > 1:
            self.has_nested_loops = True

        self.generic_visit(node)
        self.current_depth -= 1

    def visit_Assign(self, node):
        if node.targets:
            target = node.targets[0]

            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                self.has_swap = True

            if isinstance(node.value, ast.Dict):
                self.has_dict_assignment = True

                if isinstance(target, ast.Name):
                    self.dict_names.add(target.id)

            if isinstance(node.value, ast.Set):
                self.has_seen_set_pattern = True

            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                if isinstance(node.value.left, ast.Constant) and isinstance(
        node.value.left.value, str
    ):
                    self.has_string_concat = True

                elif isinstance(node.value.right, ast.Constant) and isinstance(
        node.value.right.value, str
    ):
                    self.has_string_concat = True

                elif isinstance(node.value.right, ast.Call):
                    if isinstance(node.value.right.func, ast.Name):
                        if node.value.right.func.id == "str":
                            self.has_string_concat = True

        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Add):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id == "str":
                        self.has_string_concat = True

            if isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    self.has_string_concat = True

            if isinstance(node.target, ast.Subscript):
                if isinstance(node.target.value, ast.Name):
                    if getattr(self, "dict_names", set()):
                        if node.target.value.id in self.dict_names:
                            self.has_counter_increment = True

        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Slice):
            self.has_slicing = True
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.function_name:
            self.recursive_call_count += 1

        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "append":
                self.has_append = True

            if node.func.attr == "add":
                self.has_seen_set_pattern = True

        self.generic_visit(node)

    def visit_Compare(self, node):
        if any(isinstance(op, ast.NotIn) for op in node.ops):
            self.has_not_in_compare = True

        if any(isinstance(op, ast.Eq) for op in node.ops):
            self.has_index_compare = True
            self.has_equality_compare = True

        if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            self.has_membership = True

        self.generic_visit(node)

    def visit_Return(self, node):
        if isinstance(node.value, ast.Constant):# node.value هذه يقصد بها المتغير الي يكون بعد ال return 
            if node.value.value is True:# node.value.value هنا يقصد بها قيمة العنصر بعد ال return هل هي فعليا true او false
                self.has_duplicate_return = True
                
                if self.has_equality_compare:
                    self.returns_true_on_compare = True

        self.generic_visit(node)


class ReplacementEngine:
    def __init__(self, function_name):
        self.function_name = function_name
        self.changes = []#قائمة الرسائل التي ستظهر في الواجهة تحت Changes Made.
        self.reason = ""#سبب اختيار التحسين.
        self.after_override = None#نستخدمه عندما نريد تعديل نتيجة التحليل بعد التحسين يدويًا، مثل وضع O(n) أو اسم الخوارزمية.

    def optimize(self, function_node):
        detector = ReplacementPatternDetector()
        detector.visit(function_node)

        args = [arg.arg for arg in function_node.args.args]

        if not args:
            self.reason = "No input arguments detected."
            self.changes.append(
            "No safe algorithm replacement was applied because the function has no input arguments."
        )
            return ast.unparse(function_node)

        if detector.has_append and detector.has_not_in_compare:
            return self._remove_duplicates_to_seen_set(function_node, args[0])

        if (
        detector.has_nested_loops
        and detector.max_depth >= 3
        and detector.has_equality_compare
    ):
            return self._triple_sum_to_set(function_node, args[0])

        if detector.has_swap and detector.has_nested_loops:
            return self._manual_sort_to_merge_sort(function_node, args[0])

        if self._is_pair_sum_pattern(function_node) and len(args) >= 2:
            return self._pair_sum_to_seen_set(function_node, args[0], args[1])

        if (
        detector.has_nested_loops
        and detector.has_index_compare
        and detector.has_duplicate_return
    ):
            return self._duplicate_nested_to_set(function_node, args[0])

        if (
        detector.has_nested_loops
        and detector.has_equality_compare
        and len(args) >= 2
    ):
            return self._nested_comparison_to_set_lookup(
            function_node, args[0], args[1]
        )

        if detector.has_counter_increment and detector.has_dict_assignment:
            return self._manual_frequency_to_dictionary_count(function_node, args[0])

        if detector.has_duplicate_return and detector.has_membership:
            return self._duplicate_detection_to_seen_set(function_node, args[0])
        
        if detector.recursive_call_count >= 2 and not detector.has_slicing:
            return self._repeated_recursion_to_memoization(function_node, args[0])

        if detector.has_string_concat and not detector.has_nested_loops:
            return self._string_concat_to_join(function_node, args[0])

        if detector.has_append:
            return self._append_loop_to_list_comprehension(function_node, args[0])

        self.reason = "No trusted replacement pattern matched."
        self.changes.append(
        "No safe algorithm replacement was applied because no trusted pattern matched this function."
    )
        return ast.unparse(function_node)
    
    
    def _remove_duplicates_to_seen_set(self, node, data_arg):
        self.reason = "Detected duplicate removal using list membership inside a loop."
        self.changes.extend(
            [
                "Replaced list-based membership checking with set-based tracking.",
                "Preserved the original output order.",
                "Reduced time complexity from O(n²) to O(n).",
            ]
        )
        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "algorithm": "Order-Preserving Duplicate Removal",
            "loop_type": "Single pass with set lookup",
        }
        return f"""def {node.name}({data_arg}):
    seen = set()
    result = []
    
    for item in {data_arg}:
        if item not in seen:
            seen.add(item)
            result.append(item)
            
    return result"""

    def _triple_sum_to_set(self, node, data_arg):
        self.reason = (
            "Detected triple nested loop with arithmetic condition (a + b == c)."
        )
        self.changes.extend(
            [
                "Reduced one nested loop using a set lookup.",
                "Converted O(n³) complexity to O(n²).",
                "Used hashing for fast membership checking.",
            ]
        )
        self.after_override = {
            "time_complexity": "O(n²)",
            "space_complexity": "O(n)",
            "algorithm": "Set-based pair sum detection",
            "loop_type": "Double nested loop",
        }
        return f"""def {node.name}({data_arg}):
    values = set({data_arg})
    count = 0
    for a in {data_arg}:
        for b in {data_arg}:
            if (a + b) in values:
                count += 1
    return count"""

    def _manual_sort_to_merge_sort(self, node, data_arg):
        self.reason = "Detected manual swap-based sorting using nested loops."
        self.changes.extend(
            [
                "Replaced manual swap-based sorting with Merge Sort.",
                "Used divide-and-conquer splitting and merging.",
                "Reduced expected time complexity from O(n²) to O(n log n).",
                "Generated an explicit algorithm structure instead of using a built-in sorting shortcut.",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "algorithm": "Merge Sort",
            "loop_type": "Divide-and-Conquer",
        }

        return f"""def {node.name}({data_arg}):
    def merge_sort(items):
        if len(items) <= 1:
            return items

        middle = len(items) // 2
        left_half = merge_sort(items[:middle])
        right_half = merge_sort(items[middle:])

        return merge(left_half, right_half)

    def merge(left, right):
        merged = []
        left_index = 0
        right_index = 0

        while left_index < len(left) and right_index < len(right):
            if left[left_index] <= right[right_index]:
                merged.append(left[left_index])
                left_index += 1
            else:
                merged.append(right[right_index])
                right_index += 1

        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged

    return merge_sort({data_arg})"""

    def _duplicate_nested_to_set(self, node, data_arg):
        self.reason = (
            "Detected duplicate detection using nested loops and equality comparison."
        )

        self.changes.extend(
            [
                "Replaced nested loop duplicate detection with set-based tracking.",
                "Reduced time complexity from O(n²) to O(n).",
                "Used a set for constant-time membership checks.",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "algorithm": "Set-Based Duplicate Detection",
            "loop_type": "Single pass",
        }
        return f"""def {node.name}({data_arg}):
    seen = set()
    
    for item in {data_arg}:
        if item in seen:
            return True
            
        seen.add(item)
        
    return False"""

    def _is_pair_sum_pattern(self, node):
        #نمط pair sum يعني دالة تبحث عن زوج عناصر مجموعهما يساوي target.
        for statement in ast.walk(node):
            if isinstance(statement, ast.If):
                if self._is_sum_equals_target(statement.test):# statement.test استخراج الشرط الموجود داخل جملة if
                    if self._has_append_pair(statement):
                        return True
        return False

    def _is_sum_equals_target(self, test):
        if not isinstance(test, ast.Compare):
            return False
        if len(test.ops) != 1:
            return False
        if not isinstance(test.ops[0], ast.Eq):
            return False
        if len(test.comparators) != 1:
            return False
        left = test.left # الذي قبل عملية المساواة 
        right = test.comparators[0] #الذي بعد عملية المساواة 
        return self._is_addition_of_two_subscripts(
            left
        ) or self._is_addition_of_two_subscripts(right)

    def _is_addition_of_two_subscripts(self, node):
        if not isinstance(node, ast.BinOp):#BinOp يعني نوع عملية من العملياتلاالتي تحتاج عنصرين اثنين مثل الجمع وغيرها 
            return False
        if not isinstance(node.op, ast.Add):
            return False
        return isinstance(node.left, ast.Subscript) and isinstance(
            node.right, ast.Subscript
        )

    def _has_append_pair(self, if_node):
        for child in ast.walk(if_node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr == "append":
                        return True

        return False

    def _pair_sum_to_seen_set(self, node, data_arg, target_arg):
        self.reason = "Detected pair-sum search using nested loops."
        self.changes.extend(
            [
                "Replaced nested pair comparison with set-based complement lookup.",
                "Stored visited numbers in a set.",
                "Checked target - item using average O(1) membership lookup.",
                "Reduced time complexity from O(n²) to O(n).",
            ]
        )
        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "algorithm": "Set-Based Pair Sum",
            "loop_type": "Single pass with complement lookup",
        }

        return f"""def {node.name}({data_arg}, {target_arg}):
    seen = set()
    result = []
    for item in {data_arg}:
        needed = {target_arg} - item
        if needed in seen:
            result.append((needed, item))
        seen.add(item)
    return result"""


    
    def _nested_comparison_to_set_lookup(self, node, first_arg, second_arg):
        self.reason = (
            "Detected nested equality comparison between two input collections."
        )
        self.changes.extend(
            [
                "Replaced repeated pairwise comparison with set-based lookup.",
                "Reduced repeated scanning by using average O(1) membership checks.",
                "Reduced expected time complexity from O(n*m) to O(n + m).",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n + m)",
            "space_complexity": "O(m)",
            "algorithm": "Set Lookup",
            "loop_type": "Single pass with set membership",
        }

        return f"""def {node.name}({first_arg}, {second_arg}):
    lookup = set({second_arg})
    result = []

    for item in {first_arg}:
        if item in lookup:
            result.append(item)

    return result"""

    def _manual_frequency_to_dictionary_count(self, node, data_arg):
        self.reason = "Detected manual frequency counting using dictionary updates."
        self.changes.extend(
            [
                "Rebuilt the counting logic into a clean dictionary counting strategy.",
                "Removed unnecessary branching around repeated key updates.",
                "Preserved O(n) behavior with clearer frequency aggregation.",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(k)",
            "algorithm": "Dictionary Frequency Counting",
            "loop_type": "Single pass",
        }

        return f"""def {node.name}({data_arg}):
    frequencies = {{}}

    for item in {data_arg}:
        frequencies[item] = frequencies.get(item, 0) + 1

    return frequencies"""

    def _duplicate_detection_to_seen_set(self, node, data_arg):
        self.reason = "Detected duplicate detection pattern using membership checks."
        self.changes.extend(
            [
                "Replaced repeated duplicate detection logic with a seen set.",
                "Used average O(1) membership checks.",
                "Reduced duplicate search behavior to a single pass.",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "algorithm": "Set-Based Duplicate Detection",
            "loop_type": "Single pass",
        }

        return f"""def {node.name}({data_arg}):
    seen = set()

    for item in {data_arg}:
        if item in seen:
            return True

        seen.add(item)

    return False"""

    def _string_concat_to_join(self, node, data_arg):
        self.reason = "Detected repeated string concatenation."
        self.changes.extend(
            [
                "Replaced repeated string concatenation with join-based construction.",
                "Reduced repeated string allocation inside loops.",
                "Improved scalability for larger input sizes.",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "algorithm": "Join-Based String Construction",
            "loop_type": "Generator expression",
        }

        return f"""def {node.name}({data_arg}):
    return "".join(str(item) for item in {data_arg})"""

    def _repeated_recursion_to_memoization(self, node, data_arg):
        self.reason = "Detected repeated recursive calls."
        self.changes.extend(
            [
                "Replaced repeated recursion with memoized dynamic programming.",
                "Stored previously computed results to avoid repeated subproblems.",
                "Reduced repeated recursive branching into cached computation.",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "algorithm": "Memoized Dynamic Programming",
            "loop_type": "Recursive with cache",
        }

        return f"""def {node.name}({data_arg}):
    memo = {{}}

    def solve(value):
        if value in memo:
            return memo[value]

        if value <= 1:
            memo[value] = value
            return value

        memo[value] = solve(value - 1) + solve(value - 2)
        return memo[value]

    return solve({data_arg})"""

    def _append_loop_to_list_comprehension(self, node, data_arg):
        loop_target = "item"
        condition = None

        for statement in node.body:
            if isinstance(statement, ast.For):
                if isinstance(statement.iter, ast.Call):
                    if isinstance(statement.iter.func, ast.Name):
                        if statement.iter.func.id == "range":
                            args = statement.iter.args# هنا بياخذ ال args التي في range دالة ال

                            if len(args) == 3:
                                step = args[2]
                                if isinstance(step, ast.UnaryOp):
                                    self.changes.append(
                                        "No safe algorithm replacement was detected for this function yet."
                                    )
                                    return ast.unparse(node)

                                if isinstance(step, ast.Constant) and step.value == -1:
                                    self.changes.append(
                                        "No safe algorithm replacement was detected for this function yet."
                                    )
                                    return ast.unparse(node)
                if isinstance(statement.target, ast.Name):
                    loop_target = statement.target.id
                for inner_statement in statement.body:#  هنا بيدخل الى الابناء عند الفور التي تم الحصول عليها يعني اولا مر على كل نود في النود الاصله ثم حصل فور ثم بيمر على ابناءها 
                    if isinstance(inner_statement, ast.If):
                        condition = ast.unparse(inner_statement.test)
                        # تجنب تحويل الشروط التي تعتمد على القائمة result أثناء بنائها،
                        # لأن list comprehension قد يغيّر السلوك الأصلي للدالة.
                        if condition and "result" in condition:
                            self.changes.append(
                                "No safe algorithm replacement was detected for this function yet."
                            )
                            return ast.unparse(node)

        self.reason = "Detected list construction using append inside a loop."
        if condition:
            self.changes.extend(
                [
                    "Converted append-based list construction into a conditional list comprehension.",
                    "Preserved the original filtering condition.",
                    "Maintained the same output behavior while improving readability.",
                ]
            )
            self.after_override = {
                "time_complexity": "O(n)",
                "space_complexity": "O(n)",
                "algorithm": "Conditional List Comprehension",
                "loop_type": "Single pass",
            }
            return f"""def {node.name}({data_arg}):
        return [{loop_target} for {loop_target} in {data_arg} if {condition}]"""
        
        # اذا تحققق الشرط السابق لن يرى الرسالة وسيحسنه
        self.changes.append(
            "No safe algorithm replacement was applied because no trusted pattern matched this function."
        )
        return ast.unparse(node)

    def _membership_to_precomputed_set(self, node, data_arg):# هذه الدالة لم استخدمها فضلا عن كونها بدائية لايبدو انها تتاكد من شي وتغير معنى الدوال 
        self.reason = "Detected membership-heavy logic."
        self.changes.extend(
            [
                "Introduced a precomputed set to improve membership lookup.",
                "Reduced repeated list membership checks to average O(1).",
                "Prepared the function for faster lookup-based processing.",
            ]
        )

        self.after_override = {
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "algorithm": "Precomputed Set Lookup",
            "loop_type": "Single pass",
        }

        return f"""def {node.name}({data_arg}):
    lookup = set({data_arg})
    return lookup"""


def apply_after_override(after_analysis, override):
    if not override:
        return after_analysis

    updated = dict(after_analysis)# هنا بالرغم انه فعلا  دكشنري الا انه ضروري اعمل لها نسخه لجل لايغير في الموقع نفسه 
    updated["time_complexity"] = override.get(
        "time_complexity", updated["time_complexity"]
    )
    updated["space_complexity"] = override.get(
        "space_complexity", updated["space_complexity"]
    )
    updated["algorithm"] = override.get("algorithm", updated["algorithm"])
    updated["loop_type"] = override.get("loop_type", updated["loop_type"])
    return updated


@app.post("/optimize/replacement")
def optimize_replacement(input: CodeInput):
    syntax_error = check_syntax(input.code)

    if syntax_error:
        return syntax_error

    tree = ast.parse(input.code)

    target_node = find_function_node(tree, input.function_name)

    if target_node is None:
        return {"error": "Function not found"}

    before_analysis = analyze_function_node(input.function_name, target_node)

    engine = ReplacementEngine(input.function_name)
    optimized_function_code = engine.optimize(target_node)

    full_code = replace_function_code(
        input.code,
        input.function_name,
        optimized_function_code,
    )

    try:
        after_tree = ast.parse(full_code)
    except SyntaxError:
        return {
            "error": "Generated replacement code is not valid Python.",
            "changes": engine.changes,
        }

    after_node = find_function_node(after_tree, input.function_name)
    after_analysis = None
    if after_node:
        after_analysis = analyze_function_node(input.function_name, after_node)

    after_analysis = apply_after_override(after_analysis, engine.after_override)

    return {
        "function": input.function_name,
        "optimization_type": "replacement",
        "optimized_function_code": optimized_function_code,
        "full_code": full_code,
        "changes": engine.changes,
        "before": before_analysis,
        "after": after_analysis,
        "reason": engine.reason,
    }
    
#جديد
class SimulationInput(BaseModel):
    code: str
    function_name: str
    optimization_type: str = "replacement"
    start_n: int = 10
    end_n: int = 100
    step: int = 10


def normalize_simulation_range(start_n, end_n, step):
    start_n = max(1, int(start_n))
    end_n = max(start_n, int(end_n))
    step = max(1, int(step))

    if ((end_n - start_n) // step) > 50:
        step = max(step, (end_n - start_n) // 50)

    return start_n, end_n, step


def estimate_operations(complexity, n):
    if not complexity:
        return n

    normalized = complexity.replace(" ", "").lower()

    if "2^n" in normalized:
        return min(2 ** min(n, 20), 2 ** 20)

    if "n³" in normalized or "n^3" in normalized:
        return n ** 3

    if "n⁴" in normalized or "n^4" in normalized:
        return n ** 4

    if "n²" in normalized or "n^2" in normalized or "n*m" in normalized:
        return n ** 2

    if "nlogn" in normalized or "nlog" in normalized:
        return n * max(1, math.log2(n))

    if "o(n)" in normalized or normalized == "recursive":
        return n

    return 1


def build_simulation_points(before_analysis, after_analysis, start_n, end_n, step):
    points = []

    for n in range(start_n, end_n + 1, step):
        original_ops = estimate_operations(
            before_analysis.get("time_complexity"),
            n,
        )

        optimized_ops = estimate_operations(
            after_analysis.get("time_complexity"),
            n,
        )

        points.append(
            {
                "n": n,
                "original": round(original_ops / 100, 2),
                "optimized": round(optimized_ops / 100, 2),
            }
        )

    return points


# Removed find_top_level_function as it is replaced by find_function_node


def build_simple_simulation_result(input, tree, target_node):
    before_analysis = analyze_function_node(
        input.function_name,
        target_node,
    )

    transformer = SimpleOptimizationTransformer(input.function_name)

    optimized_tree = transformer.visit(tree)

    ast.fix_missing_locations(optimized_tree)

    full_code = ast.unparse(optimized_tree)

    optimized_function_code = extract_function_code(
        full_code,
        input.function_name,
    )

    after_tree = ast.parse(full_code)

    after_node = find_function_node(
        after_tree,
        input.function_name,
    )

    after_analysis = analyze_function_node(
        input.function_name,
        after_node,
    )

    return {
        "optimized_function_code": optimized_function_code,
        "full_code": full_code,
        "changes": transformer.changes,
        "before": before_analysis,
        "after": after_analysis,
    }


def build_replacement_simulation_result(input, target_node):
    before_analysis = analyze_function_node(
        input.function_name,
        target_node,
    )

    engine = ReplacementEngine(input.function_name)

    optimized_function_code = engine.optimize(target_node)

    full_code = replace_function_code(
        input.code,
        input.function_name,
        optimized_function_code,
    )

    after_tree = ast.parse(full_code)

    after_node = find_function_node(
        after_tree,
        input.function_name,
    )

    after_analysis = analyze_function_node(
        input.function_name,
        after_node,
    )

    after_analysis = apply_after_override(
        after_analysis,
        engine.after_override,
    )

    return {
        "optimized_function_code": optimized_function_code,
        "full_code": full_code,
        "changes": engine.changes,
        "before": before_analysis,
        "after": after_analysis,
        "reason": engine.reason,
    }


@app.post("/simulate")
def simulate_code(input: SimulationInput):
    syntax_error = check_syntax(input.code)

    if syntax_error:
        return {"error": syntax_error}

    tree = ast.parse(input.code)

    target_node = find_function_node(
        tree,
        input.function_name,
    )

    if target_node is None:
        return {"error": "Function not found"}

    optimization_type = input.optimization_type

    if optimization_type not in {"simple", "replacement"}:
        optimization_type = "replacement"

    if optimization_type == "simple":
        result = build_simple_simulation_result(
            input,
            tree,
            target_node,
        )
    else:
        result = build_replacement_simulation_result(
            input,
            target_node,
        )

    start_n, end_n, step = normalize_simulation_range(
        input.start_n,
        input.end_n,
        input.step,
    )

    points = build_simulation_points(
        result["before"],
        result["after"],
        start_n,
        end_n,
        step,
    )

    last_point = points[-1] if points else {
        "original": 0,
        "optimized": 0,
    }

    if last_point["optimized"] > 0:
        speedup = round(
            last_point["original"] / last_point["optimized"],
            2,
        )
    else:
        speedup = 1

    return {
        "function": input.function_name,
        "optimization_type": optimization_type,
        "range": {
            "start_n": start_n,
            "end_n": end_n,
            "step": step,
        },
        "points": points,
        "speedup": speedup,
        **result,
    }

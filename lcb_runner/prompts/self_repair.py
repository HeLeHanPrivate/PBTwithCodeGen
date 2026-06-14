import json
import ast
import re

from lcb_runner.prompts.anthropic_compat import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = (
        "You are a helpful programming assistant and an expert Python programmer. "
        "You are repairing Python code for an automated evaluator. Preserve the expected "
        "interface or stdin/stdout protocol exactly and return only corrected Python code in one fenced block. "
        "Your response must start with ```python and must contain no explanation, analysis, or prose outside the code block."
    )

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct a error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters."

    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entired fixed program within code delimiters only for once., for example: 
```python 
# YOUR CODE HERE
```"""

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        "The user gives a error program, and the Assistant corrects it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )

    FORMATTING_REPEAT = f"First reason about the code providing a textual explanation of what is wrong with the code and then generate a fixed of the program enclosed code delimiters."

    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


# def truncate_io(io):
#     if len(str(io)) > 200:
#         io = str(io)[:200] + "...."
#     return io


# def get_check_prompt(question: str, result, metadata):
#     # v1
#     ## assumes i/o examples are already truncated!
#     ## less pressure on storing 10 MB json because on a single large input-output pair
#     # result_by_test_case = result
#     # assert len(metadata) == 1, f"metadata = {metadata}"
#     # metadata = metadata[0]
#     metadata = json.loads(metadata)
#     if "error_code" not in metadata:
#         return ""
#     if metadata["error_code"] == -1:
#         # time limit exceeded
#         message = f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"
#     elif metadata["error_code"] == -2:
#         # wrong answer
#         message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata['inputs']}\nGenerated Output: {metadata['output']}\nExpected: {metadata['expected']}"
#     elif metadata["error_code"] == -3:
#         # time limit exceeded
#         message = f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}"
#         pass
#     elif metadata["error_code"] == -4:
#         # runtime error
#         if 'inputs' in metadata.keys():
#             message = f"The above code is incorrect and got a runtime error.\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n{metadata['error']}"
#         else:
#             message = f"The above code is incorrect and got a runtime error.\nError_Message: {metadata['error_message']}"
#     elif metadata["error_code"] == -5:
#         # TestRunnerError
#         message = f"The above code is incorrect and got a runtime error.\n{metadata['error']}\nError_Message: {metadata['error_message']}"
#     else:
#         raise NotImplementedError(
#             f"metadata['error_code'] = {metadata['error_code']} not implemented || {metadata=}"
#         )
#     return message


def _looks_like_assertion_failure(metadata):
    error_text = " ".join(
        str(metadata.get(key, ""))
        for key in ("error", "error_message")
    )
    return "AssertionError" in error_text or "AssertError" in error_text


def _looks_like_precise_decimal_mismatch(metadata):
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ("output", "expected", "error_message")
    )
    return bool(re.search(r"\d+\.\d{10,}", text))


def _format_shortest_input_note(metadata):
    if "inputs" not in metadata:
        return ""
    return (
        "Shortest known failing input/call:\n"
        f"{metadata.get('inputs', 'N/A')}\n"
    )


def _short_feedback_text(value, limit=700):
    text = str(value or "N/A")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _parse_metadata_obj(metadata):
    if isinstance(metadata, dict):
        return dict(metadata)
    try:
        parsed = json.loads(metadata)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _primary_solution_method(tree: ast.Module) -> tuple[str, ast.FunctionDef] | None:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef)
            and not item.name.startswith("_")
            and item.name != "__init__"
        ]
        if methods:
            return node.name, methods[0]
    return None


def _primary_solution_method_from_source(source: str):
    if not source.strip():
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        class_method = _primary_solution_method(tree)
        if class_method is not None:
            class_name, method = class_method
            try:
                signature = ast.unparse(method.args)
            except Exception:
                signature = ", ".join(arg.arg for arg in method.args.args)
            return class_name, method.name, signature
    class_match = re.search(r"(?m)^class\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*$", source)
    if not class_match:
        return None
    class_name = class_match.group(1)
    rest = source[class_match.end():]
    method_match = re.search(
        r"(?m)^[ \t]+def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?\s*:",
        rest,
    )
    if not method_match:
        return None
    method_name = method_match.group(1)
    signature = method_match.group(2).strip()
    return class_name, method_name, signature


def _function_interface_note(code: str, metadata=None) -> str:
    parsed_metadata = _parse_metadata_obj(metadata)
    starter_code = str(parsed_metadata.get("starter_code") or "")
    interface_source = starter_code if starter_code.strip() else code
    class_method = _primary_solution_method_from_source(interface_source)
    if class_method is not None:
        class_name, method_name, signature = class_method
        return (
            "Interface contract for the evaluator:\n"
            f"- This is a LeetCode-style class solution. Preserve `class {class_name}`.\n"
            f"- Keep exactly this callable method name/signature inside the class: def {method_name}({signature})\n"
            "- Treat this starter class/method as authoritative even if the current submission is empty, malformed, or defines a different target.\n"
            "- The evaluator instantiates the class and calls this method; do not replace it with stdin/stdout code.\n"
            "- If the current candidate uses solve(), main(), sys.stdin, json input adapters, or prints output, convert it back to this class method API.\n"
            "- Return a complete Python snippet containing the class, any needed imports/helpers, and the corrected method body indented inside the class.\n"
            "- If the starter signature uses type names such as List, Dict, Optional, Tuple, Set, or Deque, include the needed imports from typing/collections unless the code avoids those annotations.\n"
            "- Do not return only the method body, a top-level function, a main() wrapper, pseudocode, or a partial patch.\n"
            "- Preserve argument order and return type semantics implied by the problem statement and public examples.\n\n"
        )
    try:
        tree = ast.parse(interface_source)
    except SyntaxError:
        return (
            "Interface contract:\n"
            "- Preserve the original public API from the given code.\n"
            "- Preserve the original input/output style unless the problem statement clearly requires a different one.\n\n"
        )
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        empty_note = ""
        if not str(interface_source).strip():
            empty_note = "- There is no usable existing implementation; solve the problem from scratch from the statement.\n"
        return (
            "Interface contract for the evaluator:\n"
            + empty_note
            + "- This is an executable contest-style program, not a function-only task.\n"
            "- Preserve the stdin/stdout protocol required by the problem statement.\n"
            "- Return a complete Python program that reads input, computes all test cases, and writes exactly the required output.\n"
            "- Do not replace the program with only a helper function, a partial patch, pseudocode, or hard-coded sample handling.\n"
            "- Keep any useful solve()/main() structure, but make sure the submitted code actually calls it when executed.\n\n"
        )
    lines = []
    for fn in functions[:2]:
        try:
            signature = ast.unparse(fn.args)
        except Exception:
            signature = ", ".join(arg.arg for arg in fn.args.args)
        lines.append(f"- Keep exactly this callable name/signature: def {fn.name}({signature})")
    return (
        "Interface contract for the evaluator:\n"
        + "\n".join(lines)
        + "\n- The evaluator calls the function by this exact name; do not rename it.\n"
        "- Do not replace the function with main(), stdin/stdout parsing, command-line code, or a differently named helper as the public API.\n"
        "- Return a complete Python snippet containing the corrected public function, starting with the exact `def ...:` line.\n"
        "- Do not return only an indented function body, partial patch, pseudocode, or a fragment without the public function definition.\n"
        "- Include all required return statements inside the function.\n"
        "- You may keep small helper functions/imports only if the required public function name remains defined.\n\n"
    )


def _repair_rules(code: str, metadata=None) -> str:
    parsed_metadata = _parse_metadata_obj(metadata)
    starter_code = str(parsed_metadata.get("starter_code") or "")
    interface_source = starter_code if starter_code.strip() else code
    has_class_method = _primary_solution_method_from_source(interface_source) is not None
    try:
        tree = ast.parse(interface_source)
    except SyntaxError:
        has_public_function = not has_class_method
    else:
        has_public_function = any(isinstance(node, ast.FunctionDef) for node in tree.body)

    common = (
        "### Repair Rules\n"
        "- Fix the underlying algorithm; do not patch only the shown sample or failing case.\n"
        "- If the current code already passes public examples but still fails hidden tests or generated properties, treat the failure as evidence of a missing edge case, invariant, or complexity requirement. Do not make a cosmetic edit that preserves the same algorithmic assumption.\n"
        "- If public examples pass but hidden tests time out, preserve the public API and semantics while replacing the asymptotic bottleneck. Prefer a different algorithmic formulation over adding caches, pruning, or constant-factor tweaks to the same slow search or nested-loop structure.\n"
        "- If the given code is only starter code, an empty method/function, `pass`, `...`, a placeholder comment, or a body missing after `def`/`class`, treat it as an incomplete implementation and write the full correct implementation from the problem statement.\n"
        "- Do not return a skeleton, partial signature, or unchanged starter code. Every public method/function must contain executable logic and all required return/output paths.\n"
        "- Before finalizing, mentally parse the exact code you will output: it must be syntactically valid Python, define the evaluator-facing class/function or executable stdin program, contain no dangling `def`/`class` without an indented body, and contain no leftover markdown, ellipses, patch markers, or explanatory text inside the code block.\n"
        "- If property assertions are present, treat them as semantic feedback about the bug, not as code to remove or silence.\n"
        "- The property/checker may contain a small oracle, brute-force verifier, differential check, or conservation law. Repair the algorithm so the produced result satisfies that property.\n"
        "- Before writing code, internally translate each property failure into an algorithm requirement: what must be true for all valid inputs, what the buggy code violated, and which part of the algorithm must change.\n"
        "- When property feedback reports a mismatch against a tiny brute-force oracle, treat the oracle as a specification witness for small inputs. Derive the general rule from it, then implement a scalable algorithm for the full constraints.\n"
        "- Do not merely special-case the tiny oracle input. Use it to identify the missing case distinction, recurrence, invariant, ordering rule, or objective being optimized.\n"
        "- For exact counting, uniqueness, tie-breaking, mode/frequency, matching, or grouping predicates, do not collapse distinct non-target values into one aggregate unless their identities are provably irrelevant. Preserve the categories needed to avoid overcounting or undercounting.\n"
        "- If structured property feedback summarizes a checker, use it as a specification witness only. Do not copy diagnostic checker logic into the final answer unless that logic is also genuinely part of the required solution algorithm.\n"
        "- The final answer must not include diagnostic property assertions, brute-force self-check blocks, randomized test harnesses, sample checks, or messages copied from checker feedback. Assertions are allowed only when the original problem explicitly requires raising AssertionError as the output behavior.\n"
        "- When a property message gives a concrete semantic predicate, range, order rule, or conservation law, use that predicate as the target behavior even if the original code comments or problem title suggest a different shortcut.\n"
        "- If a property failure comes from a generated probe input with no expected output, do not ignore it; the checker assertion is the expected semantic feedback for that probe.\n"
        "- The final code may omit property instrumentation once the algorithm is fixed.\n"
        "- Return exactly one fenced Python code block and no prose outside it.\n"
        "- The very first characters of your response must be ```python. Do not write analysis, reasoning, headings, bullets, or a problem explanation before the code block.\n"
        "- If the current submission is empty or malformed, still output a complete runnable program inside the code block instead of explaining an approach.\n"
    )
    if has_class_method:
        specific = (
            "- Preserve the class-based public API exactly: same class name, same public method name, same method arguments.\n"
            "- Do not add stdin/stdout parsing, `if __name__ == '__main__'`, or a top-level replacement function.\n"
            "- The fenced code block must contain the full class definition, not only an indented method body or fragment.\n"
            "- You may rewrite helper logic freely inside or outside the class, but the evaluator-facing class method must remain callable.\n"
            "- If the failing call has several JSON-decoded arguments, keep all corresponding method parameters and their order.\n"
        )
    elif has_public_function:
        specific = (
            "- Preserve the exact public function name and signature.\n"
            "- Do not add stdin/stdout parsing, `if __name__ == '__main__'`, or a new public function name.\n"
            "- The fenced code block must contain a full function definition. The first non-comment Python statement should be the required `def ...:` line, not an indented body fragment.\n"
            "- Never output only assignments or loop bodies such as `result = [] ...`; wrap the complete corrected algorithm in the required public function and include the final return.\n"
            "- Use the failing call arity as evidence for the public interface. If the failing input calls `f(a, b)` and the provided code/starter has two parameters, keep both parameters.\n"
            "- If a parameter in a complex-number task is passed as `1j` or another complex value, use its `.imag` or `.real` component as required instead of dropping the parameter.\n"
            "- Do not preserve an invalid precondition from the buggy code when public examples contradict it. For example, if a merge property says the result must be globally sorted and preserve all elements, fix the algorithm for the actual unsorted inputs instead of assuming the inputs were sorted.\n"
        )
    else:
        specific = (
            "- Preserve the contest-style stdin/stdout interface from the problem statement.\n"
            "- The fixed code must be a complete executable Python program, including any needed input parsing and output formatting.\n"
            "- The code block must not be empty. It must contain real executable logic, not just comments, placeholders, or helper definitions that are never called.\n"
            "- Do not switch a stdin/stdout task to a LeetCode-style `class Solution` or method-only API. Classes are allowed only as internal helpers, and the submitted program must still read stdin and print stdout when executed.\n"
            "- If the original code defines solve()/main(), keep or recreate that structure and call it at the end.\n"
            "- If you define solve(), include `if __name__ == '__main__': solve()` or an equivalent unconditional call path.\n"
            "- Handle all test cases described by the input format; do not hard-code the failing input, public samples, or output count.\n"
            "- Optimize for the stated constraints, especially when the failure is TLE or the problem has large aggregate input sizes.\n"
            "- For TLE, change the algorithmic approach rather than only caching, increasing recursion limits, or tweaking constants.\n"
            "- Avoid per-query/per-step simulation when constraints imply preprocessing, batching, offline processing, or exponentiation is required.\n"
            "- For real-valued answers, preserve numerical accuracy through the whole computation. Prefer exact rational/Decimal arithmetic for rational expectations/probabilities, use stable distance/probability formulas, and print enough significant digits such as 20-30 decimals when the judge output is high precision.\n"
            "- Preserve exact output formatting: one answer per required line, no debug prints, no explanations.\n"
        )
    return common + specific + "\n"


def get_check_prompt(question: str, result, metadata):
    ## assumes i/o examples are already truncated!
    ## less pressure on storing 10 MB json because on a single large input-output pair
    # result_by_test_case = result
    # assert len(metadata) == 1, f"metadata = {metadata}"
    # metadata = metadata[0]
    metadata = _parse_metadata_obj(metadata)
    if not metadata:
        return "Context: No specific error context from previous run available or metadata was not parsable.\n"
    if "error_code" not in metadata: # Check after successful parsing
        return "Context: Previous run did not produce a standard error code.\n"
    
    message = ""
    error_code = metadata.get('error_code')

    if error_code == -1: # Compilation Error
        message = f"Context: The program previously failed with a compilation error.\nDetails: {metadata.get('error', 'N/A')}"
    elif error_code == -2: # Wrong Answer
        message = (f"Context: The program previously produced a wrong answer on the shortest known failing case.\n"
                   f"Input: {metadata.get('inputs', 'N/A')}\n"
                   f"Generated Output: {metadata.get('output', 'N/A')}\n"
                   f"Expected Output: {metadata.get('expected', 'N/A')}")
        if _looks_like_precise_decimal_mismatch(metadata):
            message += (
                "\nPrecision focus: the mismatch involves high-precision decimal output. "
                "Avoid losing exactness through premature rounding or binary-float accumulation. "
                "Use a numerically stable algorithm and print enough significant digits, or use exact rational/Decimal arithmetic when the value is naturally rational or expected exactly."
            )
    elif error_code == -3: # Time Limit Exceeded
        message = (f"Context: The program previously hit a time limit exceeded.\n"
                   f"Details: {metadata.get('error', 'N/A')}\n"
                   f"Input: {metadata.get('inputs', 'N/A')}\n"
                   f"Expected Output: {metadata.get('expected', 'N/A')}\n"
                   "Optimization focus: infer the required asymptotic complexity from the stated constraints and replace slow simulation or nested loops with a scalable algorithm. "
                   "Look for monotonicity, sorting/sweeping, prefix sums, coordinate compression, heaps, union-find, graph shortest paths, dynamic programming, matrix/functional exponentiation, or other standard reductions as appropriate. "
                   "Do not micro-optimize an algorithm whose big-O complexity is too high.")
    elif error_code == -4: # Runtime Error
        if _looks_like_assertion_failure(metadata):
             message = (f"Context: A property/assertion check failed. Treat this as semantic feedback about the bug, not as a crash to silence.\n"
                        f"{_format_shortest_input_note(metadata)}"
                        f"Violated Property/Error: {_short_feedback_text(metadata.get('error', metadata.get('error_message', 'N/A')))}\n"
                        "This is intentionally minimal feedback: use the input only as a counterexample to infer the general violated property, not as a case to hard-code.\n"
                        "Repair instruction: fix the underlying algorithm so the property holds; do not remove, bypass, or weaken the assertion unless it is demonstrably inconsistent with the problem statement.\n"
                        "Internal repair checklist before writing code:\n"
                        "1. Restate the violated assertion as a general invariant or postcondition required by the problem.\n"
                        "2. Identify which assumption, recurrence, ordering rule, boundary case, or objective in the current code makes that invariant false.\n"
                        "3. Replace the flawed logic with an algorithm that satisfies the invariant for all valid inputs and the stated constraints.\n"
                        "4. Mentally verify the minimal counterexample and the public examples, then output only the final corrected code.")
             property_feedback = metadata.get("property_feedback")
             if property_feedback:
                 message += "\nStructured property feedback:\n" + _short_feedback_text(property_feedback, 900).strip()
        elif "No evaluation result: missing or invalid call-based target" in str(metadata.get("error_message", "")):
             message = (
                "Context: The previous candidate did not define the evaluator-facing call-based target.\n"
                f"Error Details: {metadata.get('error_message', metadata.get('error', 'N/A'))}\n"
                "Repair instruction: implement a complete solution while preserving the required public API from the starter/interface contract. "
                "For class-based tasks, return the full class with the required method. For function-based tasks, return the full required function. "
                "If the current code is empty or only a skeleton, solve from the problem statement inside that exact target. "
                "Do not replace it with solve(), main(), stdin parsing, helper-only code, a differently named method, or a partial method body. "
                "Include any imports needed by type annotations or helper data structures."
             )
        elif "No evaluation result: missing or invalid stdin program" in str(metadata.get("error_message", "")):
             message = (
                "Context: The previous candidate did not provide an executable stdin/stdout program.\n"
                f"Error Details: {metadata.get('error_message', metadata.get('error', 'N/A'))}\n"
                "Repair instruction: write a complete contest-style Python program from scratch from the problem statement. "
                "The answer must include stdin parsing, the full algorithm, output formatting, and a call to solve()/main() if you define one. "
                "Do not return an empty code block, only a helper, only a function signature, pseudocode, a partial patch, or an explanation. "
                "A valid answer should look like a complete AtCoder/ICPC submission that can be run with `python main.py < input.txt`."
             )
        elif 'inputs' in metadata: # Check if 'inputs' key exists
             message = (f"Context: The program previously encountered a runtime error on the shortest known failing case.\n"
                        f"Input: {metadata.get('inputs', 'N/A')}\n"
                        f"Expected Output: {metadata.get('expected', 'N/A')}\n"
                        f"Error Details: {metadata.get('error', 'N/A')}") # Assuming 'error' might have more details than 'error_message'
        else:
             message = f"Context: The program previously encountered a runtime error.\nError Message: {metadata.get('error_message', 'N/A')}"
    elif error_code == -5: # TestRunnerError
        message = (f"Context: The program previously caused a TestRunnerError.\n"
                   f"Error: {metadata.get('error', 'N/A')}\n"
                   f"Error Message: {metadata.get('error_message', 'N/A')}")
    else:
        message = f"Context: An unspecified error (code: {error_code}) occurred in a previous run.\n"
    return message + "\n" if message else "Context: No specific error details found for the given error code.\n"



def get_generic_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question:\n{question}\n\n"
    if str(code).strip():
        prompt += f"### Current Incorrect Or Incomplete Code:\n```python\n{code}\n```\n\n"
    else:
        prompt += "### Current Incorrect Or Incomplete Code:\n<empty submission>\n\n"
    prompt += _function_interface_note(code, metadata)
    prompt += get_check_prompt(question, result, metadata) + "\n"
    prompt += _repair_rules(code, metadata)
    prompt += (
        "### Fixed Code\n"
        "Return exactly one complete Python code block. Your response must start with ```python on the first line and end after the closing ```.\n"
    )
    return prompt


def get_cllama_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question\n{question}\n\n"
    prompt += f"### Answer\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_deepseekcode_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"### Response:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_deepseek_r1_question_template_answer(question: str, code, result, metadata):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"Response:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"Answer: (use the provided format with backticks)\n\n"
    prompt += f"<｜Assistant｜>"
    return prompt


def get_codeqwen_question_template_answer(question: str, code, result, metadata):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"Response:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"Answer: (use the provided format with backticks)\n\n"
    prompt += f"<|im_end|>\n<|im_start|>assistant\n"
    return prompt



def get_magicoder_question_template_answer(question: str, code, result, metadata):
    prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"@@ Response \n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_mixtral_question_template_answer(question: str, code, result, metadata):
    prompt = f"Question:\n"
    prompt += f"{question}\n\n"
    prompt += f"Answer:\n\n"
    prompt += f"```python\n\n{code}\n``\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_wizard_question_template_answer(question: str, code, result, metadata):
    prompt = f"""### Instruction: You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once., for example:
    ```python
    # YOUR CODE HERE
    ```
"""
    prompt += f"{question}\n\n"
    prompt += f"### Response:```python\n\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_phind_question_template_answer(question: str, code, result, metadata):
    prompt = f"{question}\n\n"
    prompt += f"```python\n{code}\n``` \n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Assistant"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_qwen_question_template_answer(question: str, code, result, metadata):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "abacusai/Dracarys-72B-Instruct", padding_side="left", use_fast=False
    )
    prompt = f"""### Instruction: You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once., for example:
    ```python
    # YOUR CODE HERE
    ```\n\n
"""
    prompt += f"Question:\n{question}\n\n"
    prompt += f"```python\n{code}\n``` \n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Assistant"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"

    messages = [
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt

def format_prompt_self_repair(
    question: str, LanguageModelStyle: LMStyle, code, result, metadata
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle in [LMStyle.OpenAIChat, LMStyle.LocalAPI]:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]
        return chat_messages
    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    elif LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{get_generic_question_template_answer(question, code, result, metadata).rstrip()}\n{AI_PROMPT}"
        return prompt
    elif LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ).rstrip(),
            }
        ]
        return system, prompt
    elif LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, code, result, metadata),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.StarCoderInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct or LanguageModelStyle == LMStyle.DeepSeekAPI:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n{get_deepseekcode_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN}\n\n{get_codeqwen_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1}\n\n{get_deepseek_r1_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n<</SYS>>\n\n{get_cllama_question_template_answer(question, code, result,metadata)}\n[/INST]"
        return prompt
    elif LanguageModelStyle == LMStyle.MagiCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_MAGIC}\n{get_magicoder_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.WizardCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_WIZARD}\n\n{get_wizard_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.Phind:
        prompt = f"### System Prompt\n\n{PromptConstants.SYSTEM_MESSAGE_PHIND}\n\n### User Message\n\n{get_phind_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DracarysQwen:
        prompt = f"{get_qwen_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DracarysLlama:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "abacusai/Dracarys-Llama-3.1-70B-Instruct", padding_side="right", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    if LanguageModelStyle == LMStyle.Eurusx:
        prompt = "[INST] Write Python code to solve the task:\n"
        prompt += f"{get_wizard_question_template_answer(question, code, result,metadata)}"
        prompt += "[/INST]"
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMa:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def test():
    def write_str_or_json(prompt):
        if isinstance(prompt, str):
            fp.write(prompt)
        else:
            fp.write(json.dumps(prompt))
        return

    for lm_style in [LMStyle.OpenAIChat]:
        with open(
            "output/GPT-3.5-Turbo-0125/Scenario.codegeneration_10_0.2_eval_all.json"
        ) as f:
            check_metadata = json.load(f)[0]
        checked_base_question_cotent = check_metadata["question_content"]
        checked_base_codes = check_metadata["code_list"][0]
        checked_base_results = check_metadata["graded_list"][0]
        checked_base_metadata = check_metadata["metadata"][0]
        leetcode_prompt = format_prompt_self_repair(
            checked_base_question_cotent,
            lm_style,
            checked_base_codes,
            checked_base_results,
            checked_base_metadata,
        )

        with open(f"/tmp/leetcode_{lm_style}.txt", "w") as fp:
            write_str_or_json(leetcode_prompt)
    return


if __name__ == "__main__":
    test()

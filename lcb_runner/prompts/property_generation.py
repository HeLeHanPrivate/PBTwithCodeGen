import json
import ast
import copy

from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts.anthropic_compat import HUMAN_PROMPT, AI_PROMPT


PROPERTY_VIOLATION_MARKER = "PGS_PROPERTY_VIOLATION"


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = (
        "You are a careful Python verification engineer. Your job is to propose "
        "property-based assertions for an existing solution without writing or "
        "repairing the solution code. Return compact checker snippets that are safe "
        "to insert before a function return."
    )

    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        "The user provides a programming problem and a Python solution. "
        "The Assistant proposes property-based assertions without changing the core algorithm. "
        "The assistant first thinks about the reasoning process in the mind and then provides the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags.<｜User｜>"
    )


PROPERTY_PROPOSAL_GUIDE = """
PGS-style property/checker proposal rules:
- Propose two to four small, high-signal properties from different angles. Prefer one strong oracle over many weak checks, but include multiple independent families when possible.
- A repair round is property/checker proposal plus the later repair step. Optimize the checker for the repair signal it will produce.
- Make feedback property-oriented: explain a semantic invariant or oracle mismatch, not only a concrete expected answer.
- Make feedback structurally minimal: the checker should isolate one root-cause signal and avoid long traces.
- Use short assertion messages, ideally under 160 characters, because they become repair feedback.
- For guarded exact oracles, include both the observed `result` and the oracle value in the assertion message when values are small, e.g. `f'tiny oracle mismatch: got {result}, expected {brute}'`. This oracle value is computed from the generated input, not copied from hidden tests.
- Use public examples to infer semantics, but never branch on exact example constants.
- Public examples are more trustworthy than comments or assumptions inside the existing buggy solution.
- Comments, docstrings, and helper names inside the existing solution may be wrong; do not use them as evidence unless the problem statement and public examples support them.
- Check postconditions of `result`; do not assert input preconditions unless the problem statement explicitly guarantees them.
- If the property needs an oracle, make it small and guarded, e.g. only for len(input) <= 8 or numeric range <= 50.
- If a small exact oracle is feasible, it is usually the highest-priority property. It should be independent from the buggy code and may simply do nothing on large inputs.
- A property may be a multi-line checker harness, not just a one-line assert.
- The checker is allowed to define local helper functions, loops, counters, brute-force logic, and differential checks against a simple independent implementation.
- For wrong-answer feedback, prefer at least one exact small-input oracle when the input types allow it. Shape/type checks alone are usually too weak to guide repair.
- For optimization, counting, dynamic programming, graph, interval, string, or combinatorics tasks, build a tiny exhaustive verifier under strict guards such as list length <= 6, string length <= 8, node count <= 6, value range <= 20, or query count <= 8.
- If the task asks for a minimum/maximum, the oracle should enumerate candidate choices and assert equality with the optimal objective value, not only bounds.
- If the task asks for a count or modulo count, the oracle should enumerate tiny cases and compare `result` to the exact count or exact count modulo the stated modulus when the modulus is explicit in the problem.
- If exact checking is impossible for general inputs, guard the oracle tightly and return silently for unsupported inputs. Do not assert a guess.
- For public-pass hidden-fail or timeout-prone code, prefer properties that stress the smallest nontrivial boundary where the slow or incomplete assumption becomes visible: ties, repeated values, empty/one-item/maximum-small guards, dense versus sparse structures, disconnected components, overlapping intervals, duplicate strings, and adversarial ordering.
- When a full-size performance oracle would be unsafe, use a boundary-small exact oracle plus a metamorphic check that can be validated without enumerating the full hidden constraints.
- The intended downstream use is counterexample discovery: the property should pass public examples and should have a realistic chance to fail on valid generated probes if the supplied solution is buggy.
- The checker must not assign to `result` or mutate the original inputs.
- The checker must eventually assert or raise AssertionError based on a comparison involving `result`.
- If the original program may mutate parameters, use `orig_<param>` for semantic checks over the original input.
- Guard helper/oracle code with type and size checks so property failures become AssertionError, not AttributeError/TypeError.
- Keep snippets insertion-compatible: no imports, no input(), no print(), no try/except, no classes, no global/nonlocal, no semicolons, and no use of names outside the listed variables/builtins.
- Good properties can be necessary invariants, inverse checks, conservation laws, metamorphic checks, small-data brute-force oracles, or special-case validators implied by the specification.

Good checker snippets:

1. Sorting / merge / selection output:
```json
[
  {
    "name": "sorted_permutation",
    "checker_code": "assert result == sorted(result), 'output must be sorted'\\nassert sorted(result) == sorted(orig_a + orig_b), 'output must preserve all input elements'",
    "reason": "Sortedness and element conservation are independent of a particular merge implementation."
  }
]
```

2. Factorization output:
```json
[
  {
    "name": "factor_product_and_primality",
    "checker_code": "def _is_prime(x):\\n    if x < 2:\\n        return False\\n    d = 2\\n    while d * d <= x:\\n        if x % d == 0:\\n            return False\\n        d += 1\\n    return True\\nprod = 1\\nfor v in result:\\n    prod *= v\\nassert prod == orig_n, 'factors must multiply to the original input'\\nassert all(_is_prime(v) for v in result), 'each factor must be prime'\\nassert result == sorted(result), 'factors must be non-decreasing'",
    "reason": "The product, primality, and ordering properties validate factorization without copying the submitted algorithm."
  }
]
```

3. Counting / combinatorics with small brute force:
```json
[
  {
    "name": "small_range_bruteforce_count",
    "checker_code": "if isinstance(orig_L, int) and isinstance(orig_R, int) and 0 <= orig_R - orig_L <= 50:\\n    brute = 0\\n    for x in range(orig_L, orig_R + 1):\\n        if 0 <= x <= 15:\\n            brute += 1\\n    assert result == brute, f'tiny oracle mismatch: got {result}, expected {brute}'",
    "reason": "A guarded brute-force counter gives a simple semantic oracle on small ranges."
  }
]
```

4. String transformation:
```json
[
  {
    "name": "string_shape_and_membership",
    "checker_code": "assert isinstance(result, str), 'result must be a string'\\nassert len(result) <= len(orig_s), 'result cannot contain more characters than the input'\\nassert all(ch in orig_s for ch in result), 'result characters must come from the input'",
    "reason": "Shape and membership constraints are independent checks for many extraction/filtering tasks."
  }
]
```

5. Dynamic programming / optimization on tiny inputs:
```json
[
  {
    "name": "tiny_input_oracle",
    "checker_code": "if isinstance(orig_nums, list) and len(orig_nums) <= 8:\\n    best = 0\\n    for mask in range(1 << len(orig_nums)):\\n        total = 0\\n        ok = True\\n        for i, v in enumerate(orig_nums):\\n            if mask & (1 << i):\\n                if i > 0 and (mask & (1 << (i - 1))):\\n                    ok = False\\n                total += v\\n        if ok and total > best:\\n            best = total\\n    assert result == best, f'tiny oracle mismatch: got {result}, expected {best}'",
    "reason": "A tiny exhaustive oracle can validate optimization logic without depending on the submitted algorithm."
  }
]
```

6. Differential checker against a simple independent implementation:
```json
[
  {
    "name": "tiny_reference_implementation",
    "checker_code": "if isinstance(orig_items, list) and len(orig_items) <= 7:\\n    ref = []\\n    for x in orig_items:\\n        if x not in ref:\\n            ref.append(x)\\n    assert result == ref, 'result disagrees with simple tiny reference implementation'",
    "reason": "The reference uses direct enumeration for tiny inputs, so it gives repair-oriented feedback without reusing the submitted algorithm."
  }
]
```

7. Metamorphic / inverse consistency:
```json
[
  {
    "name": "round_trip_consistency",
    "checker_code": "if isinstance(result, list):\\n    assert len(result) == len(orig_nums), 'result shape must match input shape'\\n    assert sorted(result) == sorted(orig_nums), 'operation must preserve the multiset of elements'",
    "reason": "Shape and multiset preservation catch semantic corruption without needing exact expected outputs."
  }
]
```

Bad checker snippets:
- `assert result == expected_output`
- `if orig_x == 10: assert result == 6`
- `assert _is_sorted(orig_a)` unless the problem explicitly guarantees sorted input and examples agree
- `import math` or any import; use built-in arithmetic and constants such as `3.141592653589793` if needed
- a property copied from the existing solution's comments/docstring when public examples disagree
- any full function implementation replacing the original solution
- checks that compute the same expression as the original return and compare `result` to it
- a property that only checks "no exception happened" or only checks the type when a stronger semantic property is available
""".strip()


def _redact_expected_from_assertion(value: object) -> str:
    text = str(value)
    try:
        parsed = ast.parse(text)
        if (
            len(parsed.body) == 1
            and isinstance(parsed.body[0], ast.Assert)
            and isinstance(parsed.body[0].test, ast.Compare)
        ):
            return "assert " + ast.unparse(parsed.body[0].test.left)
    except (SyntaxError, ValueError):
        pass
    return text.split("==", 1)[0].strip() if "==" in text else text


def _parse_context(metadata):
    public_samples = None
    raw_metadata = metadata
    if isinstance(metadata, dict):
        public_samples = metadata.get("public_samples")
        raw_metadata = metadata.get("metadata")
    try:
        data = json.loads(raw_metadata)
    except (json.JSONDecodeError, TypeError):
        data = {}
    return data, public_samples


def _format_public_examples(public_samples) -> str:
    if not public_samples:
        return ""
    try:
        payload = json.loads(public_samples.get("input_output", "{}"))
    except (AttributeError, json.JSONDecodeError, TypeError):
        return ""
    inputs = payload.get("inputs") or []
    outputs = payload.get("outputs") or []
    fn_name = payload.get("fn_name")
    rows = []
    for inp, out in list(zip(inputs, outputs))[:3]:
        rows.append(f"- input/call: {inp}; output: {out}")
    if not rows:
        return ""
    header = "### Public Examples For Semantic Inference\n"
    if fn_name:
        header += f"Target function: {fn_name}\n"
    return (
        header
        + "\n".join(rows)
        + "\nUse these to infer general properties only. Do not write assertions that branch on exact example constants.\n\n"
    )


def _function_arg_names_from_code(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    target = _find_target_function(tree)
    if target is not None:
        return _arg_names(target)
    return []


def _target_return_value_count(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    target = _find_target_function(tree)
    if target is None:
        return 0
    return sum(
        1
        for node in ast.walk(target)
        if isinstance(node, ast.Return) and node.value is not None
    )


def _function_name_from_code(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    target = _find_target_function(tree)
    if target is not None:
        return target.name
    return None


def get_property_context(metadata) -> str:
    data, public_samples = _parse_context(metadata)
    if not data:
        return "### Previous Run Context\nNo parsable previous-run feedback is available.\n\n"

    context = [_format_public_examples(public_samples).rstrip(), "### Previous Run Context"]
    if "error_code" in data:
        context.append(f"Error code: {data['error_code']}")
    if data.get("inputs") is not None:
        context.append(
            "Failing call/input with expected answer redacted: "
            + _redact_expected_from_assertion(data["inputs"])
        )
    if data.get("error") is not None:
        context.append(f"Observed error type/message: {data['error']}")
    context.append(
        "Use this only to choose where a property should observe the original program. "
        "Do not branch on these exact constants and do not infer or hard-code the expected answer."
    )
    return "\n".join(item for item in context if item) + "\n\n"


PROPERTY_VARIANT_FOCUS = {
    "oracle": (
        "This variant must focus on guarded exact oracles. Prefer tiny brute force, "
        "complete enumeration, simple independent reference implementations, or exhaustive "
        "search under strict size/value guards. If an exact oracle cannot be written reliably, "
        "return an empty JSON list rather than weak invariants."
    ),
    "invariant": (
        "This variant must focus on output validity, invariants, conservation laws, and "
        "metamorphic relations. Prefer checks that are necessary for all valid solutions and "
        "that can reject plausible wrong outputs without computing the full answer."
    ),
    "boundary": (
        "This variant must focus on boundary and special-case properties implied by the "
        "statement: empty/minimum/one-item cases, all-equal data, duplicates, ties, sorted or "
        "reverse order, disconnected or degenerate structures, and smallest nontrivial cases. "
        "Do not hard-code public examples."
    ),
}


def get_property_prompt(question: str, code: str, metadata, variant: str = "general") -> str:
    prompt = (
        "You will be given a programming problem, an existing Python solution, "
        "and optional feedback from a previous run.\n\n"
        "Role boundary:\n"
        "- You are the Tester, not the Solver.\n"
        "- Never output a full Python program.\n"
        "- Never repair the bug, complete a missing function, rename the public API, "
        "or replace the algorithm.\n"
        "- Do not use expected outputs from examples or feedback to patch the code.\n\n"
        "Task:\n"
        "1. First propose independent specification properties from the problem statement, not from the existing code.\n"
        "2. Prefer guarded tiny brute-force or reference oracles when feasible. On unsupported large inputs, the checker may simply do nothing.\n"
        "3. Also propose useful invariants/metamorphic checks when an exact oracle is not feasible.\n"
        "4. Express them as checker snippets, not as a full program.\n"
        "5. These snippets will be inserted by a separate local instrumenter before returns.\n"
        "6. Do not solve the problem from scratch and do not replace the algorithm.\n\n"
        "Available symbols inside `checker_code`:\n"
        "- `result`: the value the original function is about to return.\n"
        "- original function parameters by their names, after the original code has run.\n"
        "- `orig_<param>`: a local snapshot of each original parameter before the function ran. "
        "Prefer this for semantic checks over input values.\n"
        "- ordinary Python builtins such as len, all, any, sorted, set, sum, min, max, range.\n\n"
        "A property can be more than one assert and can be a small checker harness. "
        "It may include a simple brute-force oracle for small inputs, helper functions, "
        "input-size guards, a special-case validator, a metamorphic relationship, "
        "or a differential check against an independent simple implementation. "
        "The checker snippet must observe the "
        "original `result`; it must not replace or recompute the function's returned value.\n\n"
        "Insertion constraints:\n"
        "- No imports, input(), print(), file/network access, try/except, classes, global/nonlocal, or semicolons.\n"
        "- Do not use variable names from the problem statement unless they appear in the Available Variable Names section.\n"
        "- Assertion messages should name the violated property, not dump large values or traces.\n\n"
        "Prefer checks that can be validated from the input and produced output, such as:\n"
        "- output length/shape/range constraints\n"
        "- permutation or conservation relationships\n"
        "- monotonicity, sortedness, uniqueness, membership, parity, or divisibility constraints\n"
        "- simple cross-checks with a slower local verifier only for small inputs\n\n"
        "Activation strategy:\n"
        "- The best property should pass public examples but fail on plausible generated probe inputs when the supplied solution is semantically wrong.\n"
        "- Optimize for bug revelation under valid inputs: a checker that never fails on buggy implementations is weak even if it is true.\n"
        "- Do not make up expected outputs. Use exact brute force only when the guarded input size makes it reliable; otherwise prefer necessary invariants or return no property.\n"
        "- When an exact oracle computes an expected value for a generated tiny input, include that oracle value in the assertion message. Keep it short; do not dump large arrays.\n"
        "- If no independent checker can be made public-consistent and bug-revealing, return an empty JSON list instead of weak type-only checks.\n"
        "- Prefer tiny exact oracles for small inputs even if the hidden failure is large; the input generator can create many small probes.\n"
        "- For wrong-answer failures, prioritize semantic oracle mismatches over format/type checks. A guarded exact oracle for tiny inputs is usually the most useful repair signal.\n"
        "- For optimization problems, compare against a tiny brute-force optimum. For counting problems, compare against a tiny brute-force count. For graph or reachability problems, compare against a tiny BFS/DFS/Floyd-style reference when the graph is small.\n"
        "- For TLE-prone code, add a guarded property that validates the returned value on tiny inputs, so repair feedback is semantic rather than only performance-related.\n"
        "- For hidden-fail or timeout-prone code that already passes public examples, avoid only trivial toy inputs. Use the largest guarded tiny case your oracle can safely check, plus adversarial structure such as duplicates, ties, overlapping ranges, dense/sparse extremes, or reversed order when the statement allows them.\n"
        "- If exact enumeration is too expensive, pair a smaller exact oracle with a cheaper necessary invariant or metamorphic relation that still exercises a larger valid input shape.\n"
        "- Do not make the guard so narrow that random edge probes are unlikely to exercise it.\n\n"
        "Do not limit yourself to one-line assertions. For MBPP, HumanEval, and LCB tasks, "
        "use whatever compact checker best exposes the bug: a loop, a tiny brute-force search, "
        "a reference implementation for small inputs, a conservation law, or a domain-specific "
        "validator. Keep it guarded and independent so the repair model gets a clear property "
        "failure rather than a noisy crash.\n\n"
        "When examples contradict a word in the problem title or the existing solution's comments, "
        "trust the examples for semantics. For example, if a merge problem's examples use unsorted "
        "input lists, do not assert that inputs are sorted; assert that `result` is sorted and has "
        "the same multiset as the original inputs. If a geometry function's comments use a formula "
        "that disagrees with public examples, do not propose the commented formula as a property.\n\n"
        "Bad outputs:\n"
        "- a full Python program\n"
        "- a corrected solution or new implementation of the target function\n"
        "- properties that branch on exact constants from the failing feedback\n"
        "- checks that merely restate the original return expression or "
        "compare the result to a variable computed with the same formula\n"
        "- properties that compare against the known expected output from feedback\n\n"
        "Independence requirement:\n"
        "Property checks must be independent from the original implementation. "
        "For example, use a small brute-force verifier, a conservation law, "
        "shape/range constraints from the specification, or inverse consistency; "
        "do not compute `expected` by copying the same expression used for `result`.\n"
        "If you cannot propose an independent property, return an empty JSON list.\n\n"
        "Diversity requirement:\n"
        "When possible, return separate properties in these generic families: "
        "(a) guarded exact small oracle, (b) output validity/range/shape invariant, "
        "(c) conservation or metamorphic relation, (d) special boundary case implied by the statement. "
        "All families must remain problem-derived and public-consistent.\n\n"
        "Return exactly one JSON list in one fenced ```json block. Each item must be:\n"
        "{\"name\": \"short_name\", \"checker_code\": \"# optional helper/guard/oracle\\nassert ...\", \"reason\": \"why this is independent\"}\n"
        "Use `checker_code`, not a full program. The checker code will be inserted locally.\n\n"
    )
    focus = PROPERTY_VARIANT_FOCUS.get(variant)
    if focus:
        prompt += "### Required Independent Property Family For This Prompt\n"
        prompt += focus + "\n\n"
    else:
        prompt += (
            "### Required Independent Property Family For This Prompt\n"
            "This general variant may combine oracle, invariant, metamorphic, and boundary properties, "
            "but every property must remain independent and public-consistent.\n\n"
        )
    prompt += PROPERTY_PROPOSAL_GUIDE + "\n\n"
    prompt += f"### Problem\n{question}\n\n"
    prompt += f"### Existing Python Solution\n```python\n{code}\n```\n\n"
    arg_names = _function_arg_names_from_code(code)
    return_value_count = _target_return_value_count(code)
    if return_value_count == 0:
        prompt += (
            "### Instrumentation Availability\n"
            "No function/method return value insertion point was detected in the existing code. "
            "For stdin/stdout programs, this checker path cannot observe a `result` value safely. "
            "Return an empty JSON list instead of inventing stdin parsers, full programs, or helper-only properties.\n\n"
        )
    if arg_names:
        prompt += "### Available Variable Names\n"
        prompt += "- current parameter names: " + ", ".join(arg_names) + "\n"
        prompt += "- original input snapshots: " + ", ".join(f"orig_{name}" for name in arg_names) + "\n"
        prompt += "- return value name at insertion point: result\n"
        prompt += "Use only these parameter/snapshot names, not guessed names from the problem statement.\n\n"
    prompt += get_property_context(metadata)
    prompt += "\n### Property Proposal JSON\n"
    prompt += "```json\n[]\n```\n"
    return prompt


def format_prompt_property_generation(question: str, model_style: LMStyle, code: str, _result=None, metadata=None, variant: str = "general"):
    body = get_property_prompt(question, code, metadata, variant=variant)

    if model_style in [LMStyle.OpenAIChat, LMStyle.DeepSeekAPI, LMStyle.LocalAPI, LMStyle.MistralWeb]:
        return [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": body},
        ]
    if model_style == LMStyle.Claude:
        return f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{body.rstrip()}\n{AI_PROMPT}"
    if model_style == LMStyle.Claude3:
        return PromptConstants.SYSTEM_MESSAGE_GENERIC, [{"role": "user", "content": body.rstrip()}]
    if model_style == LMStyle.DeepSeekR1:
        return PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1 + body + "<｜Assistant｜>"
    if model_style == LMStyle.CodeQwenInstruct:
        return (
            "<|im_start|>system\n"
            + PromptConstants.SYSTEM_MESSAGE_GENERIC
            + "<|im_end|>\n<|im_start|>user\n"
            + body
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

    return PromptConstants.SYSTEM_MESSAGE_GENERIC + "\n\n" + body


def get_property_merge_prompt(question: str, code: str, metadata, proposals: list[dict]) -> str:
    proposal_rows = []
    for idx, item in enumerate(proposals[:12], 1):
        name = str(item.get("name") or f"property_{idx}")[:80]
        checker = str(item.get("checker_code", item.get("assertion", ""))).strip()
        reason = str(item.get("reason") or "")[:300]
        variant = str(item.get("prompt_variant") or item.get("source") or "")[:80]
        proposal_rows.append(
            json.dumps(
                {
                    "idx": idx,
                    "name": name,
                    "variant": variant,
                    "checker_code": checker,
                    "reason": reason,
                },
                ensure_ascii=False,
            )
        )

    prompt = (
        "You will be given a programming problem, an existing Python solution, "
        "and a list of candidate property checker snippets produced by independent generators.\n\n"
        "Role boundary:\n"
        "- You are merging and filtering tests, not solving or repairing the problem.\n"
        "- Do not output a full program or a corrected solution.\n"
        "- Do not add hidden-test-specific hints, exact failing answers, or problem-specific cheats.\n\n"
        "Task:\n"
        "Merge the candidate properties into zero to four high-quality checker snippets. "
        "Remove duplicates, weak type-only checks, unsafe snippets, and checks that depend on guessed variables. "
        "You may rewrite a checker only to make the same independent property safer, more compact, and insertion-compatible.\n\n"
        "Checker constraints:\n"
        "- Each checker observes `result`, original parameters, and `orig_<param>` snapshots.\n"
        "- Prefer guarded tiny exact oracles when reliable; otherwise use necessary invariants or metamorphic checks.\n"
        "- Unsupported large inputs should be skipped by guards, not guessed.\n"
        "- No imports, input(), print(), file/network access, try/except, classes, global/nonlocal, or semicolons.\n"
        "- Do not assign to `result` or mutate the original inputs.\n"
        "- Every returned checker must contain an assert or raise AssertionError involving `result`.\n"
        "- Assertion messages should be short and semantic.\n\n"
    )
    prompt += PROPERTY_PROPOSAL_GUIDE + "\n\n"
    prompt += f"### Problem\n{question}\n\n"
    prompt += f"### Existing Python Solution\n```python\n{code}\n```\n\n"
    arg_names = _function_arg_names_from_code(code)
    if arg_names:
        prompt += "### Available Variable Names\n"
        prompt += "- current parameter names: " + ", ".join(arg_names) + "\n"
        prompt += "- original input snapshots: " + ", ".join(f"orig_{name}" for name in arg_names) + "\n"
        prompt += "- return value name at insertion point: result\n\n"
    prompt += get_property_context(metadata)
    prompt += "### Candidate Properties To Merge\n"
    prompt += "\n".join(proposal_rows) if proposal_rows else "[]"
    prompt += (
        "\n\nReturn exactly one JSON list in one fenced ```json block. "
        "Each item must be {\"name\": \"short_name\", \"checker_code\": \"...\", \"reason\": \"...\"}. "
        "Return [] if none of the candidate properties can be made reliable.\n"
    )
    return prompt


def format_prompt_property_merge(question: str, model_style: LMStyle, code: str, _result=None, metadata=None, proposals=None):
    body = get_property_merge_prompt(question, code, metadata, proposals or [])

    if model_style in [LMStyle.OpenAIChat, LMStyle.DeepSeekAPI, LMStyle.LocalAPI, LMStyle.MistralWeb]:
        return [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": body},
        ]
    if model_style == LMStyle.Claude:
        return f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{body.rstrip()}\n{AI_PROMPT}"
    if model_style == LMStyle.Claude3:
        return PromptConstants.SYSTEM_MESSAGE_GENERIC, [{"role": "user", "content": body.rstrip()}]
    if model_style == LMStyle.DeepSeekR1:
        return PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1 + body + "<｜Assistant｜>"
    if model_style == LMStyle.CodeQwenInstruct:
        return (
            "<|im_start|>system\n"
            + PromptConstants.SYSTEM_MESSAGE_GENERIC
            + "<|im_end|>\n<|im_start|>user\n"
            + body
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

    return PromptConstants.SYSTEM_MESSAGE_GENERIC + "\n\n" + body


def get_property_instrumentation_prompt(question: str, code: str, metadata, properties: list[dict]) -> str:
    rows = []
    for idx, item in enumerate(properties[:8], 1):
        rows.append(
            json.dumps(
                {
                    "idx": idx,
                    "name": str(item.get("name") or f"property_{idx}")[:80],
                    "checker_code": str(item.get("checker_code", item.get("assertion", ""))).strip(),
                    "reason": str(item.get("reason") or "")[:300],
                },
                ensure_ascii=False,
            )
        )

    prompt = (
        "You will be given a programming problem, an existing Python solution, and "
        "property checker snippets. Insert the checker snippets into the existing "
        "solution so they observe each returned value as `result`.\n\n"
        "Role boundary:\n"
        "- You are only instrumenting the code for testing.\n"
        "- Do not repair, optimize, rewrite, rename public APIs, or change the algorithm.\n"
        "- Do not add imports, input/output code, hidden-test hints, or expected answers.\n\n"
        "Instrumentation rules:\n"
        "- Preserve the class/function signatures and all original behavior except for raising AssertionError on property violations.\n"
        "- Before the target function body mutates parameters, create snapshots named `orig_<param>`. Use `.copy()` with a safe fallback to the original object.\n"
        "- Before every non-empty return in the target function, assign the return expression to `result`, run the checker snippets, then `return result`.\n"
        "- Prefix assertion messages with `PGS_PROPERTY_VIOLATION: ` when possible.\n"
        "- If a checker cannot be inserted safely because variable names do not exist, omit that checker. Do not invent new semantics.\n"
        "- Return only the full instrumented Python code in one ```python block.\n\n"
    )
    prompt += f"### Problem\n{question}\n\n"
    prompt += f"### Existing Python Solution\n```python\n{code}\n```\n\n"
    arg_names = _function_arg_names_from_code(code)
    if arg_names:
        prompt += "### Available Variable Names\n"
        prompt += "- current parameter names: " + ", ".join(arg_names) + "\n"
        prompt += "- original input snapshots to create: " + ", ".join(f"orig_{name}" for name in arg_names) + "\n"
        prompt += "- return value name at insertion point: result\n\n"
    prompt += get_property_context(metadata)
    prompt += "### Property Checker Snippets\n"
    prompt += "\n".join(rows) if rows else "[]"
    prompt += "\n\n### Instrumented Python Code\n"
    return prompt


def format_prompt_property_instrumentation(
    question: str,
    model_style: LMStyle,
    code: str,
    _result=None,
    metadata=None,
):
    if isinstance(metadata, dict):
        properties = metadata.get("properties") or []
    else:
        properties = []
    body = get_property_instrumentation_prompt(question, code, metadata, properties)

    system = (
        "You are a careful Python instrumentation engineer. Insert property checks "
        "without repairing or changing the target algorithm."
    )
    if model_style in [LMStyle.OpenAIChat, LMStyle.DeepSeekAPI, LMStyle.LocalAPI, LMStyle.MistralWeb]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ]
    if model_style == LMStyle.Claude:
        return f"{HUMAN_PROMPT}\n{system}\n\n{body.rstrip()}\n{AI_PROMPT}"
    if model_style == LMStyle.Claude3:
        return system, [{"role": "user", "content": body.rstrip()}]
    if model_style == LMStyle.DeepSeekR1:
        return PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1 + body + "<｜Assistant｜>"
    if model_style == LMStyle.CodeQwenInstruct:
        return (
            "<|im_start|>system\n"
            + system
            + "<|im_end|>\n<|im_start|>user\n"
            + body
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    return system + "\n\n" + body


def get_verified_property_prompt(question: str, buggy_code: str, fixed_code: str, metadata=None) -> str:
    """Prompt for post-evaluation property distillation from a verified repair."""
    prompt = (
        "You will be given a programming problem, a buggy Python solution, a repaired "
        "Python solution that has already passed the evaluator, and optional failure "
        "feedback from the buggy run.\n\n"
        "Task:\n"
        "Distill the semantic difference into one to three independent property/checker "
        "snippets. These properties are for future bug finding and repair feedback; they "
        "must not depend on exact hidden-test expected outputs or constants from one case.\n\n"
        "Rules:\n"
        "- Use the repaired solution only to understand the intended semantics; do not copy its implementation.\n"
        "- Prefer properties that would pass the repaired solution and expose the buggy behavior.\n"
        "- Prefer small guarded oracles, conservation laws, shape/range constraints, inverse checks, or metamorphic relations.\n"
        "- The checker snippets will be inserted before returns with `result` bound to the returned value.\n"
        "- Available names are `result`, current parameters, and `orig_<param>` snapshots.\n"
        "- No imports, input(), print(), file access, try/except, classes, global/nonlocal, semicolons, or full programs.\n"
        "- If no reliable independent property can be distilled, return an empty JSON list.\n\n"
        "Return exactly one JSON list in one fenced ```json block. Each item must be:\n"
        "{\"name\": \"short_name\", \"checker_code\": \"assert ...\", \"reason\": \"why this property captures the verified repair\"}\n\n"
    )
    prompt += PROPERTY_PROPOSAL_GUIDE + "\n\n"
    prompt += f"### Problem\n{question}\n\n"
    prompt += f"### Buggy Python Solution\n```python\n{buggy_code}\n```\n\n"
    prompt += f"### Verified Repaired Python Solution\n```python\n{fixed_code}\n```\n\n"

    arg_names = _function_arg_names_from_code(fixed_code) or _function_arg_names_from_code(buggy_code)
    if arg_names:
        prompt += "### Available Variable Names\n"
        prompt += "- current parameter names: " + ", ".join(arg_names) + "\n"
        prompt += "- original input snapshots: " + ", ".join(f"orig_{name}" for name in arg_names) + "\n"
        prompt += "- return value name at insertion point: result\n\n"

    data, _ = _parse_context(metadata)
    if data:
        prompt += "### Buggy Failure Feedback\n"
        if data.get("inputs") is not None:
            prompt += "Failing input/call with expected answer redacted: " + _redact_expected_from_assertion(data["inputs"]) + "\n"
        if data.get("error") is not None or data.get("error_message") is not None:
            prompt += "Observed buggy error: " + str(data.get("error") or data.get("error_message")) + "\n"
        prompt += "Use this only to locate the semantic gap; do not hard-code this exact case.\n\n"

    prompt += "### Distilled Property JSON\n```json\n[]\n```\n"
    return prompt


def format_prompt_verified_property_extraction(
    question: str,
    model_style: LMStyle,
    buggy_code: str,
    fixed_code: str,
    metadata=None,
):
    body = get_verified_property_prompt(question, buggy_code, fixed_code, metadata)
    if model_style in [LMStyle.OpenAIChat, LMStyle.DeepSeekAPI, LMStyle.LocalAPI, LMStyle.MistralWeb]:
        return [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": body},
        ]
    if model_style == LMStyle.Claude:
        return f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{body.rstrip()}\n{AI_PROMPT}"
    if model_style == LMStyle.Claude3:
        return PromptConstants.SYSTEM_MESSAGE_GENERIC, [{"role": "user", "content": body.rstrip()}]
    return PromptConstants.SYSTEM_MESSAGE_GENERIC + "\n\n" + body


def _extract_json_block(text: str) -> str:
    if not text:
        return ""
    marker = "```"
    start = text.find(marker)
    if start != -1:
        end = text.find(marker, start + len(marker))
        if end != -1:
            block = text[start + len(marker):end].strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            return block
    list_start = text.find("[")
    list_end = text.rfind("]")
    if list_start != -1 and list_end > list_start:
        return text[list_start:list_end + 1]
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        return text[obj_start:obj_end + 1]
    return text.strip()


def extract_property_assertions(text: str, _model_style: LMStyle | None = None) -> list[dict]:
    raw = _extract_json_block(text)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("properties") or parsed.get("assertions") or [parsed]
    if not isinstance(parsed, list):
        return []

    assertions = []
    for item in parsed:
        if isinstance(item, str):
            item = {"assertion": item}
        if not isinstance(item, dict):
            continue
        checker_code = str(item.get("checker_code", item.get("assertion", ""))).strip()
        if not checker_code:
            continue
        assertions.append({
            "name": str(item.get("name", f"property_{len(assertions)}"))[:80],
            "checker_code": checker_code,
            "assertion": checker_code,
            "reason": str(item.get("reason", ""))[:300],
        })
    return assertions


def _public_payload(metadata) -> dict:
    _, public_samples = _parse_context(metadata)
    if not public_samples:
        return {}
    try:
        return json.loads(public_samples.get("input_output", "{}"))
    except (AttributeError, json.JSONDecodeError, TypeError):
        return {}


def _property(name: str, checker_code: str, reason: str, source: str = "local") -> dict:
    return {
        "name": name,
        "checker_code": checker_code.strip(),
        "assertion": checker_code.strip(),
        "reason": reason,
        "source": source,
    }


def synthesize_property_probe_inputs(question: str, code: str, metadata=None) -> list[str]:
    """Additional generated calls for property-only probes.

    Keep this deterministic hook empty by default. Concrete probe generation is
    handled by the LLM property proposer and public-case execution feedback; a
    hard-coded keyword-to-input table would leak task-specific hints into the
    repair prompt.
    """
    return []


def synthesize_local_properties(question: str, code: str, metadata=None) -> list[dict]:
    """Deterministic local properties.

    Disabled for benchmark runs. The property checker should discover semantic
    checks from the prompt and feedback, not from a repository-level table of
    task-specific fixes.
    """
    return []


def _arg_names(function_def: ast.FunctionDef) -> list[str]:
    args = []
    for arg in list(function_def.args.posonlyargs) + list(function_def.args.args) + list(function_def.args.kwonlyargs):
        args.append(arg.arg)
    if function_def.args.vararg:
        args.append(function_def.args.vararg.arg)
    if function_def.args.kwarg:
        args.append(function_def.args.kwarg.arg)
    return args


def _find_target_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
            public_methods = [
                item
                for item in methods
                if not item.name.startswith("_") and item.name not in {"__init__"}
            ]
            if public_methods:
                return public_methods[0]
            if methods:
                return methods[0]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _assertion_node(assertion: str) -> ast.Assert | None:
    try:
        parsed = ast.parse(assertion)
    except SyntaxError:
        try:
            parsed = ast.parse(assertion, mode="eval")
        except SyntaxError:
            return None
        return ast.Assert(test=parsed.body, msg=ast.Constant("property violation"))
    if len(parsed.body) != 1:
        return None
    node = parsed.body[0]
    if isinstance(node, ast.Assert):
        return node
    if isinstance(node, ast.Expr):
        return ast.Assert(test=node.value, msg=ast.Constant("property violation"))
    return None


def _checker_nodes(checker_code: str) -> list[ast.stmt]:
    try:
        parsed = ast.parse(checker_code)
    except SyntaxError:
        assertion = _assertion_node(checker_code)
        return [assertion] if assertion is not None else []
    return parsed.body


def _name_set(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _store_names(node: ast.AST) -> set[str]:
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            names.add(item.id)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
    return names


def _has_top_level_return(nodes: list[ast.stmt]) -> bool:
    def stmt_has_return(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.FunctionDef):
            return False
        if isinstance(stmt, ast.Return):
            return True
        return any(stmt_has_return(child) for child in ast.iter_child_nodes(stmt) if isinstance(child, ast.stmt))

    return any(stmt_has_return(node) for node in nodes)


def _looks_structurally_safe(checker_code: str, allowed_names: set[str], return_exprs: set[str]) -> bool:
    banned = [
        "class ",
        "import ",
        "open(",
        "exec(",
        "eval(",
        "__",
        "input(",
    ]
    compact = checker_code.replace(" ", "")
    if any(item in checker_code for item in banned):
        return False
    if len(checker_code) > 3000 or ";" in checker_code:
        return False
    has_return_expr_copy = any(expr and expr.replace(" ", "") in compact for expr in return_exprs)
    if has_return_expr_copy and checker_code.count("assert") <= 1:
        return False
    nodes = _checker_nodes(checker_code)
    if not nodes:
        return False
    if _has_top_level_return(nodes):
        return False
    probe = ast.Module(body=nodes, type_ignores=[])
    if not any(isinstance(node, (ast.Assert, ast.Raise)) for node in ast.walk(probe)):
        return False
    banned_nodes = (ast.Import, ast.ImportFrom, ast.Delete, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try)
    if any(isinstance(node, banned_nodes) for node in ast.walk(probe)):
        return False
    for call in [node for node in ast.walk(probe) if isinstance(node, ast.Call)]:
        if isinstance(call.func, ast.Name) and call.func.id in {"exec", "eval", "open", "input", "compile", "__import__"}:
            return False
    for node in [node for node in ast.walk(probe) if isinstance(node, ast.Raise)]:
        exc = node.exc
        if not (
            isinstance(exc, ast.Call)
            and isinstance(exc.func, ast.Name)
            and exc.func.id == "AssertionError"
        ):
            return False
    names = _name_set(probe)
    safe_builtins = {
        "AssertionError",
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
        "len", "list", "max", "min", "range", "reversed", "round", "set", "sorted",
        "str", "sum", "tuple", "zip", "True", "False", "None",
    }
    local_defs = {node.name for node in ast.walk(probe) if isinstance(node, ast.FunctionDef)}
    assigned = _store_names(probe)
    if "result" in assigned:
        return False
    unknown = names - allowed_names - safe_builtins - local_defs - assigned
    return not unknown and "result" in names


class _PropertyInjector(ast.NodeTransformer):
    def __init__(self, checker_codes: list[str]) -> None:
        self.checker_codes = checker_codes

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            return node
        result_assign = ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())],
            value=node.value,
        )
        injected = []
        for checker_code in self.checker_codes:
            injected.extend(_mark_property_failures(copy.deepcopy(_checker_nodes(checker_code))))
        return_node = ast.Return(value=ast.Name(id="result", ctx=ast.Load()))
        return [result_assign, *injected, return_node]


def _prefixed_message(message: ast.AST | None) -> ast.AST:
    if message is None:
        return ast.Constant(PROPERTY_VIOLATION_MARKER)
    return ast.BinOp(
        left=ast.Constant(PROPERTY_VIOLATION_MARKER + ": "),
        op=ast.Add(),
        right=ast.Call(func=ast.Name(id="str", ctx=ast.Load()), args=[message], keywords=[]),
    )


class _PropertyFailureMarker(ast.NodeTransformer):
    def visit_Assert(self, node: ast.Assert):
        self.generic_visit(node)
        node.msg = _prefixed_message(node.msg)
        return node

    def visit_Raise(self, node: ast.Raise):
        self.generic_visit(node)
        exc = node.exc
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "AssertionError":
            if exc.args:
                exc.args[0] = _prefixed_message(exc.args[0])
            else:
                exc.args.append(ast.Constant(PROPERTY_VIOLATION_MARKER))
        return node


def _mark_property_failures(nodes: list[ast.stmt]) -> list[ast.stmt]:
    marker = _PropertyFailureMarker()
    return [marker.visit(node) for node in nodes]


def instrument_code_with_properties(code: str, properties: list[dict]) -> tuple[str, list[dict]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "", []

    target = _find_target_function(tree)
    if target is None:
        return "", []
    arg_names = _arg_names(target)
    allowed_names = set(arg_names) | {f"orig_{name}" for name in arg_names} | {"result"}
    return_exprs = {
        ast.unparse(node.value)
        for node in ast.walk(target)
        if isinstance(node, ast.Return) and node.value is not None
    }
    return_exprs = {
        expr
        for expr in return_exprs
        if len(expr.replace(" ", "")) > 3 and not expr.isidentifier()
    }

    accepted = []
    checker_codes = []
    for item in properties:
        checker_code = str(item.get("checker_code", item.get("assertion", ""))).strip()
        if _looks_structurally_safe(checker_code, allowed_names, return_exprs):
            accepted.append(item)
            checker_codes.append(checker_code)
    if not checker_codes:
        return "", []

    snapshots = [
        ast.Assign(
            targets=[ast.Name(id=f"orig_{name}", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="copy", ctx=ast.Load()),
                args=[],
                keywords=[],
            ),
        )
        for name in arg_names
    ]
    fallback_snapshots = []
    for name in arg_names:
        fallback_snapshots.append(
            ast.Try(
                body=[snapshots[len(fallback_snapshots)]],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="Exception", ctx=ast.Load()),
                        name=None,
                        body=[
                            ast.Assign(
                                targets=[ast.Name(id=f"orig_{name}", ctx=ast.Store())],
                                value=ast.Name(id=name, ctx=ast.Load()),
                            )
                        ],
                    )
                ],
                orelse=[],
                finalbody=[],
            )
        )
    target.body = fallback_snapshots + target.body
    injector = _PropertyInjector(checker_codes)
    injector.visit(target)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree), accepted
    except Exception:
        return "", []


def test():
    print(
        json.dumps(
            format_prompt_property_generation(
                "Return prime factors of n.",
                LMStyle.DeepSeekR1,
                "def f(n):\n    return []",
                '{"error_code": "-2"}',
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    test()

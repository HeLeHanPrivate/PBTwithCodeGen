import json
import ast
import re
import textwrap
import numpy as np
from tqdm import tqdm
import threading
from queue import Empty, PriorityQueue
import time
import multiprocessing
from pathlib import Path

from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
from lcb_runner.prompts.self_repair import format_prompt_self_repair
from lcb_runner.prompts.checker_extend import format_prompt_checker_extend, get_metadata
from lcb_runner.prompts.test_case_generation import format_prompt_testcase_generate
from lcb_runner.prompts.code_generation import format_prompt_generation
from lcb_runner.prompts.checker_generate import format_prompt_checker_generate
from lcb_runner.prompts.property_generation import (
    PROPERTY_VIOLATION_MARKER,
    extract_property_assertions,
    format_prompt_property_instrumentation,
    format_prompt_property_merge,
    format_prompt_property_generation,
    instrument_code_with_properties,
    synthesize_local_properties,
    synthesize_property_probe_inputs,
)
from lcb_runner.prompts.test_inputer_generation import format_prompt_inputer_generate, execute_inputer_script
from lcb_runner.utils.extraction_utils import extract_code, extract_testcase


run_answer_list = []


def run_exec(samples, code, timeout):
    curr_res = [-2]
    try:
        curr_res, curr_metadata = check_correctness(samples, code, timeout, False)
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
    except Exception as e:
        curr_metadata = {
            "error": repr(e),
            "error_code": -5,
            "error_message": "TestRunnerError",
        }
    finally:
        assert isinstance(curr_res, list), curr_res
        assert isinstance(curr_metadata, dict), curr_metadata
        return curr_res, curr_metadata


def append_property_probe_inputs(samples, probe_inputs):
    if not probe_inputs:
        return samples
    try:
        payload = json.loads(samples["input_output"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return samples
    if payload.get("platform") != "humaneval":
        return samples
    merged = dict(payload)
    merged["inputs"] = list(payload.get("inputs") or []) + list(probe_inputs)
    merged["outputs"] = list(payload.get("outputs") or []) + ["" for _ in probe_inputs]
    return {"input_output": json.dumps(merged)}


def run_exec_with_workid(samples, code, timeout, work_id):
    # print(f"child process started run_exec for work_id={work_id}")
    answer = run_exec(samples, code, timeout)
    return answer, work_id


def run_exec_result_handler(result):
    answer, work_id = result
    global run_answer_list
    run_answer_list[work_id] = answer
    # print(f"child process finished run_exec for work_id={work_id}")


def all_tests_passed(results):
    return bool(results) and all(result is True for result in results)


def normalize_extracted_code(code):
    if not code:
        return code
    try:
        compile(code, "<candidate>", "exec")
        return code
    except SyntaxError as exc:
        if exc.msg != "unexpected indent" or exc.lineno != 1:
            return code
    dedented = textwrap.dedent(code)
    if dedented == code:
        return code
    try:
        compile(dedented, "<candidate>", "exec")
    except SyntaxError:
        return code
    return dedented


def candidate_is_complete_python(code):
    if not str(code).strip():
        return False
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError:
        return False
    return True


class _AssertionStripper(ast.NodeTransformer):
    def visit_Assert(self, node):
        return None

    def visit_Raise(self, node):
        exc = node.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        if isinstance(exc, ast.Name) and exc.id == "AssertionError":
            return None
        return node

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if hasattr(node, "body") and isinstance(node.body, list) and not node.body:
            node.body.append(ast.Pass())
        if hasattr(node, "orelse") and isinstance(node.orelse, list) and not node.orelse:
            node.orelse = []
        if hasattr(node, "finalbody") and isinstance(node.finalbody, list) and not node.finalbody:
            node.finalbody = []
        return node


def strip_final_assertion_checks(code):
    if "assert" not in code and "AssertionError" not in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    stripped = _AssertionStripper().visit(tree)
    ast.fix_missing_locations(stripped)
    try:
        cleaned = ast.unparse(stripped)
        compile(cleaned, "<candidate_without_asserts>", "exec")
    except Exception:
        return code
    return cleaned + "\n"


def parse_metadata(metadata):
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def augment_interface_metadata(metadata, platform, problem):
    parsed = parse_metadata(metadata)
    parsed["platform"] = platform
    parsed["starter_code"] = getattr(problem, "starter_code", "") or ""
    parsed["func_name"] = getattr(problem, "metadata", {}).get("func_name", None)
    return parsed


def _required_class_method(metadata):
    parsed = parse_metadata(metadata)
    starter_code = str(parsed.get("starter_code") or "")
    if not starter_code.strip():
        return None
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and not item.name.startswith("_")
                    and item.name != "__init__"
                ):
                    return node.name, item.name, len(item.args.args)
    class_match = re.search(r"(?m)^class\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*$", starter_code)
    if not class_match:
        return None
    rest = starter_code[class_match.end():]
    method_match = re.search(
        r"(?m)^[ \t]+def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?\s*:",
        rest,
    )
    if method_match:
        args = [part.strip() for part in method_match.group(2).split(",") if part.strip()]
        return class_match.group(1), method_match.group(1), len(args)
    return None


def candidate_preserves_interface(code, metadata):
    required = _required_class_method(metadata)
    parsed_metadata = parse_metadata(metadata)
    func_name = parsed_metadata.get("func_name")
    if required is None:
        if not func_name:
            return True
    else:
        class_name, method_name, arg_count = required
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    top_level_functions = {
        node.name: len(node.args.args)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    class_methods = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_methods[(node.name, item.name)] = len(item.args.args)

    if required is not None:
        if class_methods.get((class_name, method_name)) == arg_count:
            return True
        # LiveCodeBench's call-based evaluator also accepts an evaluator-facing
        # top-level function with the target name. For class starter code, the
        # top-level function has one fewer argument because it has no `self`.
        if top_level_functions.get(method_name) == max(0, arg_count - 1):
            return True
        return False

    if func_name in top_level_functions:
        return True
    if ("Solution", func_name) in class_methods:
        return True
    return False


def metadata_error_code(metadata):
    parsed = parse_metadata(metadata)
    try:
        return int(parsed.get("error_code"))
    except (TypeError, ValueError):
        return None


def metadata_is_assertion_failure(metadata):
    parsed = parse_metadata(metadata)
    error_text = " ".join(
        str(parsed.get(key, ""))
        for key in ("error", "error_message")
    )
    return "AssertionError" in error_text or "AssertError" in error_text


def metadata_is_property_assertion(metadata):
    parsed = parse_metadata(metadata)
    error_text = " ".join(
        str(parsed.get(key, ""))
        for key in ("error", "error_message")
    )
    return PROPERTY_VIOLATION_MARKER in error_text


def metadata_is_input_only_wrong_answer(metadata):
    parsed = parse_metadata(metadata)
    if metadata_error_code(parsed) != -2:
        return False
    expected = parsed.get("expected")
    return expected in ("", None)


def property_feedback_text(properties):
    local_properties = [item for item in properties if item.get("source") == "local"]
    if local_properties:
        properties = local_properties
    rows = []
    for item in properties[:3]:
        name = str(item.get("name", "property"))
        reason = str(item.get("reason", "")).strip()
        checker = str(item.get("checker_code", item.get("assertion", ""))).strip()
        rows.append(f"- {name}")
        if reason:
            rows.append(f"  reason: {reason}")
        if checker:
            messages = re.findall(r"assert\s+.+?,\s*(['\"])(.*?)\1", checker)
            for _, message in messages[:2]:
                rows.append(f"  assertion_message: {message[:160]}")
            compact_checker = " ".join(checker.split())
            if compact_checker:
                rows.append(f"  checker_excerpt: {compact_checker[:500]}")
    return "\n".join(rows)


class MyPipeline:
    def __init__(self, args) -> None:
        self.args = args
        self.checker_code_log = []
        self.condition = threading.Condition()
        self.request_queue = PriorityQueue()  # shared queue between main and worker threads
        self.prompts_answer_list = []
        self.platform = ""
        global run_answer_list
        run_answer_list = []
        self.run_answer_wait_flag = "Waiting Multiprocessing.Pool Answer"
        self.default_error = '{"error_code": "-2"}'
        self.finished = threading.Event()
        self.thread_num = args.num_process_evaluate
        self.multiprocess_num = args.num_process_evaluate
        self.active_workers = 0
        self.testcase_generation_num = 0
        self.property_generation_num = 0
        self.no_public_tescase_num = 0
        self.inputer_retry_num = 0
        self.inputer_failed_script_num = 0
        self.inputer_smoke_success_num = 0
        self.inputer_smoke_failed_num = 0
        self.inputer_batch_success_num = 0
        self.inputer_batch_failed_num = 0
        self.inputer_final_success_num = 0
        self.inputer_final_empty_num = 0
        self.generated_bug_kill_num = 0
        self.generated_repair_check_pass_num = 0
        self.generated_repair_check_reject_num = 0
        self.property_public_pass_accept_num = 0
        self.property_public_fail_assert_accept_num = 0
        self.property_public_direct_seed_accept_num = 0
        self.property_public_direct_seed_reject_num = 0
        self.property_public_direct_seed_missing_num = 0
        self.property_proposed_num = 0
        self.property_accepted_num = 0
        self.property_rejected_num = 0
        self.property_error_num = 0
        self.property_instrumented_num = 0
        self.repair_attempt_num = 0
        self.final_public_gate_revert_num = 0
        self.final_structural_gate_revert_num = 0
        self.public_fail_branch_num = 0
        self.public_fail_branch_fixed_num = 0
        self.public_pass_branch_num = 0
        self.public_pass_branch_fixed_num = 0
        self.repair_trace_lock = threading.Lock()
        self.repair_trace_path = ""
        self.property_trace_lock = threading.Lock()
        self.property_trace_path = ""
        if getattr(args, "save_repair_trace", False):
            output_path = getattr(args, "output_path", "")
            if output_path:
                self.repair_trace_path = output_path.replace(".json", "_repair_trace.jsonl")
                Path(self.repair_trace_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.repair_trace_path).write_text("")
        if getattr(args, "save_property_trace", False):
            output_path = getattr(args, "output_path", "")
            if output_path:
                self.property_trace_path = output_path.replace(".json", "_property_trace.jsonl")
                Path(self.property_trace_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.property_trace_path).write_text("")
        self.public_direct_outputs = self.load_public_direct_outputs(
            getattr(args, "property_public_direct_outputs_path", "")
        )
        self.public_direct_public_pass = self.load_public_direct_public_pass(
            getattr(args, "property_public_direct_outputs_path", "")
        )

    def load_public_direct_outputs(self, path):
        if not path:
            return {}
        try:
            with open(path) as f:
                rows = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not load public direct outputs from {path}: {exc}")
            return {}
        outputs = {}
        for row in rows:
            question_id = row.get("question_id")
            code_list = row.get("code_list") or []
            if question_id and code_list:
                outputs[question_id] = code_list[0]
        print(f"Loaded {len(outputs)} public direct repair candidates from {path}")
        return outputs

    def append_repair_trace(self, event):
        if not self.repair_trace_path:
            return
        safe_event = dict(event)
        with self.repair_trace_lock:
            with open(self.repair_trace_path, "a") as f:
                f.write(json.dumps(safe_event, ensure_ascii=False, default=str) + "\n")

    def append_property_trace(self, event):
        if not self.property_trace_path:
            return
        safe_event = dict(event)
        with self.property_trace_lock:
            with open(self.property_trace_path, "a") as f:
                f.write(json.dumps(safe_event, ensure_ascii=False, default=str) + "\n")

    def load_public_direct_public_pass(self, raw_path):
        if not raw_path:
            return {}
        eval_path = raw_path[:-5] + "_eval_all.json" if raw_path.endswith(".json") else ""
        if not eval_path:
            return {}
        try:
            with open(eval_path) as f:
                rows = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not load public direct eval metadata from {eval_path}: {exc}")
            return {}
        public_pass = {}
        for row in rows:
            question_id = row.get("question_id")
            public_graded = row.get("public_graded_list") or []
            if question_id and public_graded:
                public_pass[question_id] = bool(public_graded[0])
        print(f"Loaded {len(public_pass)} public direct repair public-test decisions from {eval_path}")
        return public_pass
    
    
    def prompts_to_code(
        self,
        worker_id,
        question_content,
        model_style,
        code,
        metadata,
        prompts_to_outputs,
        format_prompt,
        extract_func,
        fallback_to_original=True,
    ):
        output_code = ""
        last_output_code = ""
        try_times = 0
        max_attempts = max(1, int(getattr(self.args, "prompt_extract_max_attempts", 3)))
        while try_times < max_attempts and output_code == "":
            try_times += 1
            prompt = format_prompt(
                question_content,
                model_style,
                code,
                False,
                metadata,
            )
            
            if self.thread_num == 1:
                output = prompts_to_outputs([prompt])[0]
            else:
                # submit the request to the main thread and wait for it
                event = threading.Event()
                self.request_queue.put((worker_id, "prompts_to_code", prompt, event))
                # print(f"worker {worker_id} queued prompts_to_code request and is waiting")
                event.wait()  # wait for the main thread to process
                # print(f"worker {worker_id} resumed")
                output = self.prompts_answer_list[worker_id]
            
            output_code = extract_func(output[0], model_style) if type(output) is list else extract_func(output, model_style)
            if extract_func is extract_code:
                output_code = normalize_extracted_code(output_code)
                if not candidate_is_complete_python(output_code):
                    last_output_code = output_code
                    output_code = ""
            elif output_code:
                last_output_code = output_code
        if output_code == "" and fallback_to_original:
            return code
        return output_code if output_code != "" else last_output_code


    def prompts_to_properties(
        self,
        worker_id,
        question_content,
        model_style,
        code,
        metadata,
        prompts_to_outputs,
        samples=None,
        trace_context=None,
    ):
        trace_context = trace_context or {}
        proposals = synthesize_local_properties(
            question_content,
            code,
            {"metadata": metadata, "public_samples": samples},
        )
        self.append_property_trace({
            **trace_context,
            "event": "property_local_synthesis",
            "worker_id": worker_id,
            "proposal_count": len(proposals),
            "proposals": proposals,
        })
        variants = ["oracle", "invariant", "boundary"]
        max_variants = max(1, int(getattr(self.args, "property_prompt_variants", len(variants))))
        extract_attempts = max(1, int(getattr(self.args, "prompt_extract_max_attempts", 2)))
        for variant in variants[:max_variants]:
            for extract_attempt in range(1, extract_attempts + 1):
                prompt = format_prompt_property_generation(
                    question_content,
                    model_style,
                    code,
                    False,
                    {"metadata": metadata, "public_samples": samples},
                    variant=variant,
                )

                if self.thread_num == 1:
                    output = prompts_to_outputs([prompt])[0]
                else:
                    event = threading.Event()
                    self.request_queue.put((worker_id, "prompts_to_code", prompt, event))
                    event.wait()
                    output = self.prompts_answer_list[worker_id]

                raw = output[0] if isinstance(output, list) else output
                extracted = extract_property_assertions(raw, model_style)
                self.append_property_trace({
                    **trace_context,
                    "event": "property_llm_generation",
                    "worker_id": worker_id,
                    "prompt_variant": variant,
                    "extract_attempt": extract_attempt,
                    "raw_output": raw,
                    "extracted_count": len(extracted),
                    "extracted_properties": extracted,
                })
                if extracted:
                    for item in extracted:
                        item["prompt_variant"] = variant
                        proposals.append(item)
                    break
        unique = []
        seen = set()
        for item in proposals:
            key = item.get("checker_code") or item.get("assertion") or ""
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        self.append_property_trace({
            **trace_context,
            "event": "property_dedup",
            "worker_id": worker_id,
            "proposal_count": len(proposals),
            "unique_count": len(unique),
            "unique_properties": unique,
        })
        merge_min_candidates = max(2, int(getattr(self.args, "property_llm_merge_min_candidates", 2)))
        if (
            getattr(self.args, "property_llm_merge", False)
            and len(unique) >= merge_min_candidates
        ):
            prompt = format_prompt_property_merge(
                question_content,
                model_style,
                code,
                False,
                {"metadata": metadata, "public_samples": samples},
                proposals=unique,
            )
            if self.thread_num == 1:
                output = prompts_to_outputs([prompt])[0]
            else:
                event = threading.Event()
                self.request_queue.put((worker_id, "prompts_to_code", prompt, event))
                event.wait()
                output = self.prompts_answer_list[worker_id]
            raw = output[0] if isinstance(output, list) else output
            merged = []
            seen = set()
            for item in extract_property_assertions(raw, model_style):
                item["prompt_variant"] = "llm_merge"
                key = item.get("checker_code") or item.get("assertion") or ""
                if key and key not in seen:
                    seen.add(key)
                    merged.append(item)
            self.append_property_trace({
                **trace_context,
                "event": "property_llm_merge",
                "worker_id": worker_id,
                "input_count": len(unique),
                "raw_output": raw,
                "merged_count": len(merged),
                "merged_properties": merged,
            })
            if merged:
                return merged
        elif getattr(self.args, "property_llm_merge", False):
            self.append_property_trace({
                **trace_context,
                "event": "property_llm_merge_skipped",
                "worker_id": worker_id,
                "input_count": len(unique),
                "min_candidates": merge_min_candidates,
            })
        return unique
    
    
    def put_run_exec(self, worker_id, samples, output_code, timeout):
        if self.thread_num == 1:
            curr_res, curr_metadata = run_exec(samples, output_code, timeout)
        else:
            # submit the execution request to the main thread and wait for it
            event = threading.Event()
            self.request_queue.put((worker_id, "run_exec", (samples, output_code, timeout), event))
            # print(f"worker {worker_id} queued run_exec request and is waiting")
            event.wait()  # wait for the main thread to process
            # print(f"worker {worker_id} resumed")
            global run_answer_list
            curr_res, curr_metadata = run_answer_list[worker_id]
        return curr_res, curr_metadata
    

    def get_public_input_output(self, problem):
        inputs_outputs_pairs = sorted(problem.public_test_cases, key=lambda x: len(str(x.input)), reverse=False)
        public_input_output = {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in inputs_outputs_pairs
                    ],
                    "outputs": [
                        t.output
                        for t in inputs_outputs_pairs
                    ],
                    "fn_name": problem.metadata.get("func_name", None),
                    "platform": self.platform,
                }
            ),
        }
        if len(problem.public_test_cases) == 0:
            self.no_public_tescase_num += 1
            # print("no_public_tescase_num", self.no_public_tescase_num) # debug
        return public_input_output

    def repair_code(
        self,
        worker_id,
        question_content,
        model_style,
        code,
        metadata,
        prompts_to_outputs,
        samples,
        use_original_metadata,
        args,
        trace_context=None,
    ):
        output_code = ""
        try_times = 0
        fla = True
        fallback_code = code
        if use_original_metadata:
            original_metadata = metadata
        else:
            current_metadata = get_metadata(self.put_run_exec, worker_id, samples, code, args.timeout, self.default_error)
            original_metadata = current_metadata
        original_code = code
        max_attempts = max(1, int(getattr(args, "repair_max_attempts", 5)))
        while try_times < max_attempts and fla:
            try_times += 1
            #print("repair_code", try_times)
            output_code = self.prompts_to_code(
                worker_id,
                question_content,
                model_style,
                original_code,
                original_metadata,
                prompts_to_outputs,
                format_prompt_self_repair,
                extract_code,
                fallback_to_original=False,
            )
            interface_ok = candidate_preserves_interface(output_code, original_metadata)
            if interface_ok:
                curr_res, curr_metadata = self.put_run_exec(worker_id, samples, output_code, args.timeout)
            else:
                curr_res = [False]
                curr_metadata = dict(parse_metadata(original_metadata))
                curr_metadata.update({
                    "error_code": -4,
                    "error_message": "Candidate changed or removed the evaluator-facing public API.",
                    "error": "The repaired code must preserve the class/method signature from starter_code.",
                })
            self.append_repair_trace(
                {
                    **(trace_context or {}),
                    "event": "repair_attempt",
                    "worker_id": worker_id,
                    "attempt": try_times,
                    "use_original_metadata": use_original_metadata,
                    "input_code": original_code,
                    "candidate_code": output_code,
                    "candidate_empty": output_code == "",
                    "candidate_complete": candidate_is_complete_python(output_code),
                    "interface_ok": interface_ok,
                    "public_results": curr_res,
                    "public_pass": all_tests_passed(curr_res),
                    "metadata": curr_metadata,
                }
            )
            if output_code != "" and not use_original_metadata and interface_ok:
                original_metadata = curr_metadata
                original_code = output_code
            elif output_code != "" and not interface_ok:
                original_metadata = curr_metadata
                original_code = output_code
            elif output_code != "" and use_original_metadata:
                # A multi-attempt repair should learn from the candidate's actual
                # failure (syntax, interface, public-test mismatch, timeout, etc.).
                # Keep a separate fallback so public-failing attempts do not replace
                # the original unless one attempt passes the available tests.
                original_metadata = curr_metadata
                original_code = output_code
            if all_tests_passed(curr_res):
                fla = False
                original_code = output_code
        if output_code == "":
            output_code = fallback_code if use_original_metadata else original_code
        elif fla and use_original_metadata:
            # For known public failures, never replace the candidate with a repair
            # that still fails the public tests. Keeping the original failure keeps
            # later feedback meaningful and avoids converting WA/TLE into syntax
            # or interface errors.
            if not getattr(args, "keep_failed_repairs", False):
                output_code = fallback_code
        self.append_repair_trace(
            {
                **(trace_context or {}),
                "event": "repair_result",
                "worker_id": worker_id,
                "attempts": try_times,
                "use_original_metadata": use_original_metadata,
                "kept_failed_repairs": getattr(args, "keep_failed_repairs", False),
                "returned_code": output_code,
                "returned_equals_fallback": output_code == fallback_code,
                "successful_attempt": not fla,
            }
        )
        
        return output_code, fla


    def legacy_checker_extend(self, worker_id, question_content, model_style, code, metadata, prompts_to_outputs, samples, args):
        output_code = ""
        try_times = 0
        fla = True
        max_attempts = max(1, int(getattr(args, "checker_max_attempts", 3)))
        while try_times < max_attempts and fla:
            #print("checker_extend", try_times)
            try_times += 1
            output_code = self.prompts_to_code(worker_id, question_content, model_style, code, metadata, prompts_to_outputs, format_prompt_checker_extend, extract_code)
            if not candidate_preserves_interface(output_code, metadata):
                output_code = ""
                continue
            curr_res, curr_metadata = self.put_run_exec(worker_id, samples, output_code, args.timeout)
            if "error_code" not in curr_metadata.keys() or curr_metadata["error_code"] != -2:
                if output_code != "":
                    fla = False
        if output_code == "":
            output_code = code
        if "assert" in output_code or "raise" in output_code:
            self.property_generation_num += 1
        return output_code, None, fla


    def property_checker_extend(
        self,
        worker_id,
        question_content,
        model_style,
        code,
        metadata,
        prompts_to_outputs,
        samples,
        args,
        public_grade,
        trace_context=None,
    ):
        trace_context = trace_context or {}
        output_code = ""
        property_metadata = None
        property_info = {"accepted_properties": [], "property_feedback": ""}
        try_times = 0
        rejected_num = 0
        accepted_branch = ""
        max_attempts = max(1, int(getattr(args, "checker_max_attempts", 3)))
        while try_times < max_attempts:
            try_times += 1
            property_proposals = self.prompts_to_properties(
                worker_id,
                question_content,
                model_style,
                code,
                metadata,
                prompts_to_outputs,
                samples,
                {**trace_context, "checker_attempt": try_times},
            )
            candidate = ""
            accepted_properties = []
            instrumentation_source = "local_ast"
            if getattr(args, "property_llm_instrument", False) and property_proposals:
                llm_candidate = self.prompts_to_code(
                    worker_id,
                    question_content,
                    model_style,
                    code,
                    {
                        "metadata": metadata,
                        "public_samples": samples,
                        "properties": property_proposals,
                    },
                    prompts_to_outputs,
                    format_prompt_property_instrumentation,
                    extract_code,
                    fallback_to_original=False,
                )
                self.append_property_trace({
                    **trace_context,
                    "event": "property_llm_instrumentation",
                    "worker_id": worker_id,
                    "attempt": try_times,
                    "proposal_count": len(property_proposals),
                    "instrumented_code": llm_candidate,
                    "candidate_complete": candidate_is_complete_python(llm_candidate),
                    "interface_ok": candidate_preserves_interface(llm_candidate, metadata),
                    "has_checker": "assert" in llm_candidate or "raise" in llm_candidate,
                })
                if (
                    llm_candidate
                    and candidate_is_complete_python(llm_candidate)
                    and candidate_preserves_interface(llm_candidate, metadata)
                    and ("assert" in llm_candidate or "raise" in llm_candidate)
                ):
                    candidate = llm_candidate
                    accepted_properties = property_proposals
                    instrumentation_source = "llm"
            if not candidate:
                candidate, accepted_properties = instrument_code_with_properties(code, property_proposals)
            if candidate == "":
                rejected_num += 1
                self.append_property_trace({
                    **trace_context,
                    "event": "property_instrumentation",
                    "worker_id": worker_id,
                    "attempt": try_times,
                    "public_grade": public_grade,
                    "proposal_count": len(property_proposals),
                    "instrumented_count": 0,
                    "instrumented_code": "",
                    "accepted_properties": [],
                    "instrumentation_source": instrumentation_source,
                    "decision": "reject_empty_instrumentation",
                })
                continue

            probe_inputs = synthesize_property_probe_inputs(
                question_content,
                code,
                {"metadata": metadata, "public_samples": samples},
            )
            property_samples = append_property_probe_inputs(samples, probe_inputs)
            curr_res, curr_metadata = run_exec(property_samples, candidate, args.timeout)
            error_code = metadata_error_code(curr_metadata)
            has_checker = "assert" in candidate or "raise" in candidate
            decision = "pending"

            if public_grade:
                # Filtering: a property-injected program must preserve known-good public behavior.
                if all_tests_passed(curr_res) and has_checker:
                    output_code = candidate
                    property_metadata = None
                    feedback = property_feedback_text(accepted_properties)
                    property_info = {
                        "accepted_properties": accepted_properties,
                        "property_feedback": feedback,
                    }
                    self.property_generation_num += 1
                    self.property_public_pass_accept_num += 1
                    accepted_branch = "public_pass_preserve"
                    decision = accepted_branch
                    self.append_property_trace({
                        **trace_context,
                        "event": "property_instrumentation",
                        "worker_id": worker_id,
                        "attempt": try_times,
                        "public_grade": public_grade,
                        "proposal_count": len(property_proposals),
                        "instrumented_count": len(accepted_properties),
                        "instrumented_code": candidate,
                        "accepted_properties": accepted_properties,
                        "instrumentation_source": instrumentation_source,
                        "probe_inputs": probe_inputs,
                        "property_samples": property_samples,
                        "run_results": curr_res,
                        "metadata": curr_metadata,
                        "has_checker": has_checker,
                        "decision": decision,
                    })
                    break
                decision = "reject_public_preservation_failed"
                self.append_property_trace({
                    **trace_context,
                    "event": "property_instrumentation",
                    "worker_id": worker_id,
                    "attempt": try_times,
                    "public_grade": public_grade,
                    "proposal_count": len(property_proposals),
                    "instrumented_count": len(accepted_properties),
                    "instrumented_code": candidate,
                    "accepted_properties": accepted_properties,
                    "instrumentation_source": instrumentation_source,
                    "probe_inputs": probe_inputs,
                    "property_samples": property_samples,
                    "run_results": curr_res,
                    "metadata": curr_metadata,
                    "has_checker": has_checker,
                    "decision": decision,
                })
                rejected_num += 1
                continue

            # For a public-failing solution, keep a property only when it turns the
            # failure into explicit assertion feedback instead of another ordinary
            # WA or unrelated runtime crash.
            if (
                error_code == -4
                and has_checker
                and (
                    metadata_is_property_assertion(curr_metadata)
                    or metadata_is_assertion_failure(curr_metadata)
                )
            ):
                output_code = candidate
                feedback = property_feedback_text(accepted_properties)
                if feedback:
                    curr_metadata = dict(curr_metadata)
                    curr_metadata["property_feedback"] = feedback
                property_metadata = curr_metadata
                property_info = {
                    "accepted_properties": accepted_properties,
                    "property_feedback": feedback,
                }
                self.property_generation_num += 1
                self.property_public_fail_assert_accept_num += 1
                accepted_branch = "public_fail_assertion_feedback"
                decision = accepted_branch
                self.append_property_trace({
                    **trace_context,
                    "event": "property_instrumentation",
                    "worker_id": worker_id,
                    "attempt": try_times,
                    "public_grade": public_grade,
                    "proposal_count": len(property_proposals),
                    "instrumented_count": len(accepted_properties),
                    "instrumented_code": candidate,
                    "accepted_properties": accepted_properties,
                    "instrumentation_source": instrumentation_source,
                    "probe_inputs": probe_inputs,
                    "property_samples": property_samples,
                    "run_results": curr_res,
                    "metadata": curr_metadata,
                    "has_checker": has_checker,
                    "decision": decision,
                })
                break
            decision = "reject_no_assertion_feedback"
            self.append_property_trace({
                **trace_context,
                "event": "property_instrumentation",
                "worker_id": worker_id,
                "attempt": try_times,
                "public_grade": public_grade,
                "proposal_count": len(property_proposals),
                "instrumented_count": len(accepted_properties),
                "instrumented_code": candidate,
                "accepted_properties": accepted_properties,
                "instrumentation_source": instrumentation_source,
                "probe_inputs": probe_inputs,
                "property_samples": property_samples,
                "run_results": curr_res,
                "metadata": curr_metadata,
                "has_checker": has_checker,
                "decision": decision,
            })
            rejected_num += 1

        if output_code == "" and getattr(args, "property_fallback_legacy", False):
            output_code, property_metadata, _ = self.legacy_checker_extend(
                worker_id,
                question_content,
                model_style,
                code,
                metadata,
                prompts_to_outputs,
                samples,
                args,
            )
            property_info = {"accepted_properties": [], "property_feedback": ""}
        if output_code == "":
            output_code = code
        self.property_proposed_num += len(property_proposals) if "property_proposals" in locals() else 0
        self.property_instrumented_num += len(accepted_properties) if "accepted_properties" in locals() else 0
        if output_code != code:
            self.property_accepted_num += 1
        else:
            self.property_rejected_num += 1
        if metadata_error_code(property_metadata) not in (None, "", "0"):
            self.property_error_num += 1
        self.append_property_trace({
            **trace_context,
            "event": "property_checker_result",
            "worker_id": worker_id,
            "public_grade": public_grade,
            "accepted": output_code != code,
            "rejected": rejected_num,
            "property_error_code": metadata_error_code(property_metadata),
            "proposal_count": len(property_proposals) if "property_proposals" in locals() else 0,
            "instrumented_count": len(accepted_properties) if "accepted_properties" in locals() else 0,
            "accepted_branch": accepted_branch,
            "output_code": output_code,
            "property_metadata": property_metadata,
            "property_info": property_info,
        })
        self.checker_code_log.append({
            "mode": "property",
            "accepted": output_code != code,
            "rejected": rejected_num,
            "property_error_code": metadata_error_code(property_metadata),
            "proposal_count": len(property_proposals) if "property_proposals" in locals() else 0,
            "instrumented_count": len(accepted_properties) if "accepted_properties" in locals() else 0,
            "accepted_branch": accepted_branch,
        })
        return output_code, property_metadata, output_code == code, property_info

    def run_generated_property_checks(self, code, accepted_properties, samples, args):
        if not accepted_properties or not samples:
            return None, None, ""
        instrumented, _ = instrument_code_with_properties(code, accepted_properties)
        if not instrumented:
            return [False], {
                "error_code": -4,
                "error_message": "Could not instrument repaired candidate with accepted generated properties.",
            }, ""
        curr_res, curr_metadata = run_exec(samples, instrumented, args.timeout)
        return curr_res, curr_metadata, instrumented

    def public_passes(self, worker_id, samples, code, metadata, args):
        if not candidate_is_complete_python(code):
            return False
        if not candidate_preserves_interface(code, metadata):
            return False
        curr_res, _ = self.put_run_exec(worker_id, samples, code, args.timeout)
        return all_tests_passed(curr_res)


    def extra_testcase(self, worker_id, question_content, model_style, code, platform, samples, prompts_to_outputs, func_name, problem, args):
        # direct test input generate
        # testcase = self.prompts_to_code(worker_id, question_content, model_style, code, platform, prompts_to_outputs, format_prompt_testcase_generate, extract_testcase)
        
        # inputer generate
        retry_times = 0
        collected_inputs = []
        seen_inputs = set()
        max_inputer_attempts = max(3, int(getattr(args, "prompt_extract_max_attempts", 3)))
        target_inputs = 50
        smoke_inputs = min(3, target_inputs)
        min_usable_inputs = 8
        inputer_feedback = ""
        trace_context = {
            "worker_id": worker_id,
            "question_id": problem.question_id,
            "question_title": getattr(problem, "question_title", ""),
            "platform": str(platform),
            "event_scope": "input_generator",
        }

        def add_unique_inputs(items):
            for item in items:
                value = item.get("input") if isinstance(item, dict) else None
                if value and value not in seen_inputs:
                    seen_inputs.add(value)
                    collected_inputs.append(item)

        while retry_times < max_inputer_attempts and len(collected_inputs) < target_inputs:
            testcase_inputer = self.prompts_to_code(
                worker_id,
                question_content,
                model_style,
                code,
                (platform, samples, inputer_feedback),
                prompts_to_outputs,
                format_prompt_inputer_generate,
                extract_code,
                fallback_to_original=False,
            )
            self.append_property_trace({
                **trace_context,
                "event": "inputer_script_generated",
                "attempt": retry_times + 1,
                "feedback": inputer_feedback,
                "script_code": testcase_inputer,
                "script_empty": not testcase_inputer.strip(),
            })
            if not testcase_inputer.strip():
                retry_times += 1
                self.inputer_retry_num += 1
                self.append_property_trace({
                    **trace_context,
                    "event": "inputer_attempt_result",
                    "attempt": retry_times,
                    "stage": "extract",
                    "successful": False,
                    "diagnostics": "empty extracted input generator script",
                    "collected_count": len(collected_inputs),
                    "collected_inputs": collected_inputs,
                })
                continue

            smoke_testcase, smoke_successful, smoke_diagnostics = execute_inputer_script(
                testcase_inputer,
                smoke_inputs,
                1,
                return_diagnostics=True,
            )
            add_unique_inputs(smoke_testcase)
            self.append_property_trace({
                **trace_context,
                "event": "inputer_execution",
                "attempt": retry_times + 1,
                "stage": "smoke",
                "script_code": testcase_inputer,
                "num_executions": smoke_inputs,
                "successful": smoke_successful,
                "diagnostics": smoke_diagnostics,
                "generated_inputs": smoke_testcase,
                "collected_count": len(collected_inputs),
            })
            if not smoke_successful:
                self.inputer_smoke_failed_num += 1
                self.inputer_failed_script_num += 1
                inputer_feedback = smoke_diagnostics or "The previous generator script failed during smoke execution."
                retry_times += 1
                self.inputer_retry_num += 1
                if len(collected_inputs) >= min_usable_inputs:
                    break
                continue
            else:
                self.inputer_smoke_success_num += 1

            remaining = max(0, target_inputs - len(collected_inputs))
            if remaining:
                batch_testcase, batch_successful, batch_diagnostics = execute_inputer_script(
                    testcase_inputer,
                    remaining,
                    1,
                    return_diagnostics=True,
                )
                add_unique_inputs(batch_testcase)
                self.append_property_trace({
                    **trace_context,
                    "event": "inputer_execution",
                    "attempt": retry_times + 1,
                    "stage": "batch",
                    "script_code": testcase_inputer,
                    "num_executions": remaining,
                    "successful": batch_successful,
                    "diagnostics": batch_diagnostics,
                    "generated_inputs": batch_testcase,
                    "collected_count": len(collected_inputs),
                })
                if not batch_successful:
                    self.inputer_batch_failed_num += 1
                    self.inputer_failed_script_num += 1
                    inputer_feedback = batch_diagnostics or "The previous generator script failed during repeated execution."
                    retry_times += 1
                    self.inputer_retry_num += 1
                    if len(collected_inputs) >= min_usable_inputs:
                        break
                    continue
                else:
                    self.inputer_batch_success_num += 1

            if len(collected_inputs) >= min_usable_inputs:
                break
            retry_times += 1
            self.inputer_retry_num += 1
        
        testcase = collected_inputs
        if len(testcase) > 0:
            try:
                inputs = [t['input'] for t in testcase]
                inputs = sorted(inputs, key=lambda x: len(str(x)), reverse=False)
                result = {
                    "input_output": json.dumps(
                        {
                            "inputs": inputs + [
                                t.input
                                for t in problem.public_test_cases
                            ],
                            "outputs": [
                                "" # t.output, "The generated data only includes input"
                                for t in testcase
                            ] + [
                                t.output
                                for t in problem.public_test_cases
                            ],
                            "fn_name": func_name,
                            "platform": self.platform,
                        }
                    ),
                }
                self.inputer_final_success_num += 1
                self.append_property_trace({
                    **trace_context,
                    "event": "inputer_result",
                    "successful": True,
                    "collected_count": len(testcase),
                    "collected_inputs": testcase,
                    "result_samples": result,
                })
                return result
            except Exception as exc:
                self.inputer_final_empty_num += 1
                self.append_property_trace({
                    **trace_context,
                    "event": "inputer_result",
                    "successful": False,
                    "collected_count": len(testcase),
                    "collected_inputs": testcase,
                    "diagnostics": repr(exc),
                })
                return ""
        else:
            self.append_property_trace({
                **trace_context,
                "event": "inputer_result",
                "successful": False,
                "collected_count": 0,
                "collected_inputs": [],
                "diagnostics": "no generated inputs collected",
            })
            return ""


    def solve_one_problem(self, worker_id, question_content, code, public_grade, metadata, platform, problem, model_style, args, prompts_to_outputs):
        self.platform = platform
        samples = self.get_public_input_output(problem)
        base_code = code if str(code).strip() else problem.starter_code
        metadata = augment_interface_metadata(metadata, platform, problem)
        trace_context = {
            "worker_id": worker_id,
            "question_id": problem.question_id,
            "question_title": getattr(problem, "question_title", ""),
            "platform": str(platform),
            "public_grade": public_grade,
            "base_code": base_code,
        }
        if (
            not public_grade
            and getattr(args, "property_public_fail_direct_only", False)
        ):
            checker_extend_code = base_code
            property_metadata = None
            property_info = {"accepted_properties": [], "property_feedback": ""}
            fla = True
        elif getattr(args, "checker_mode", "property") == "legacy":
            checker_extend_code, property_metadata, fla = self.legacy_checker_extend(worker_id, question_content, model_style, base_code, metadata, prompts_to_outputs, samples, args)
            property_info = {"accepted_properties": [], "property_feedback": ""}
        else:
            checker_extend_code, property_metadata, fla, property_info = self.property_checker_extend(
                worker_id,
                question_content,
                model_style,
                base_code,
                metadata,
                prompts_to_outputs,
                samples,
                args,
                public_grade,
                trace_context,
            )
        if not candidate_preserves_interface(checker_extend_code, metadata):
            checker_extend_code = base_code
            property_metadata = None
            property_info = {"accepted_properties": [], "property_feedback": ""}
            fla = True
        if not public_grade:
            self.public_fail_branch_num += 1
            repaired_code = ""
            if (
                property_metadata is None
                and getattr(args, "property_public_fail_direct_only", False)
                and self.public_direct_outputs
            ):
                direct_seed_code = self.public_direct_outputs.get(problem.question_id, "")
                if direct_seed_code:
                    direct_public_pass = self.public_direct_public_pass.get(problem.question_id)
                    if direct_public_pass is True:
                        repaired_code = direct_seed_code
                        self.property_public_direct_seed_accept_num += 1
                        self.append_repair_trace({
                            "event": "direct_seed_accept",
                            "worker_id": worker_id,
                            "question_id": problem.question_id,
                            "question_title": getattr(problem, "question_title", ""),
                            "branch": "public_fail",
                            "seed_public_pass": True,
                            "seed_code": direct_seed_code,
                        })
                    elif direct_public_pass is False:
                        self.property_public_direct_seed_reject_num += 1
                        self.append_repair_trace({
                            "event": "direct_seed_reject",
                            "worker_id": worker_id,
                            "question_id": problem.question_id,
                            "question_title": getattr(problem, "question_title", ""),
                            "branch": "public_fail",
                            "seed_public_pass": False,
                            "seed_code": direct_seed_code,
                        })
                    elif self.public_passes(worker_id, samples, direct_seed_code, metadata, args):
                        repaired_code = direct_seed_code
                        self.property_public_direct_seed_accept_num += 1
                        self.append_repair_trace({
                            "event": "direct_seed_accept",
                            "worker_id": worker_id,
                            "question_id": problem.question_id,
                            "question_title": getattr(problem, "question_title", ""),
                            "branch": "public_fail",
                            "seed_public_pass": True,
                            "seed_code": direct_seed_code,
                        })
                    else:
                        self.property_public_direct_seed_reject_num += 1
                        self.append_repair_trace({
                            "event": "direct_seed_reject",
                            "worker_id": worker_id,
                            "question_id": problem.question_id,
                            "question_title": getattr(problem, "question_title", ""),
                            "branch": "public_fail",
                            "seed_public_pass": False,
                            "seed_code": direct_seed_code,
                        })
                else:
                    self.property_public_direct_seed_missing_num += 1
                    self.append_repair_trace({
                        "event": "direct_seed_missing",
                        "worker_id": worker_id,
                        "question_id": problem.question_id,
                        "question_title": getattr(problem, "question_title", ""),
                        "branch": "public_fail",
                    })

            if not repaired_code:
                repair_metadata = property_metadata if property_metadata is not None else metadata
                repair_input_code = base_code if property_metadata is not None else checker_extend_code
                repaired_code, _ = self.repair_code(
                    worker_id,
                    question_content,
                    model_style,
                    repair_input_code,
                    repair_metadata,
                    prompts_to_outputs,
                    samples,
                    True,
                    args,
                    {
                        "question_id": problem.question_id,
                        "question_title": getattr(problem, "question_title", ""),
                        "branch": "public_fail",
                        "repair_stage": "public_fail_live_repair",
                        "had_property_metadata": property_metadata is not None,
                    },
                )
                if repaired_code and repaired_code != base_code:
                    self.repair_attempt_num += 1
            if (
                getattr(args, "property_public_direct_fallback", False)
                and property_metadata is not None
                and not self.public_passes(worker_id, samples, repaired_code, metadata, args)
            ):
                direct_code, _ = self.repair_code(
                    worker_id,
                    question_content,
                    model_style,
                    base_code,
                    metadata,
                    prompts_to_outputs,
                    samples,
                    True,
                    args,
                    {
                        "question_id": problem.question_id,
                        "question_title": getattr(problem, "question_title", ""),
                        "branch": "public_fail",
                        "repair_stage": "public_fail_direct_fallback",
                    },
                )
                if self.public_passes(worker_id, samples, direct_code, metadata, args):
                    repaired_code = direct_code
            if self.public_passes(worker_id, samples, repaired_code, metadata, args):
                self.public_fail_branch_fixed_num += 1
        else:
            self.public_pass_branch_num += 1
            testcase = ""
            repair_metadata = self.default_error
            generated_property_killed_bug = False
            if not property_info.get("accepted_properties"):
                repaired_code = base_code
            else:
                testcase = self.extra_testcase(worker_id, question_content, model_style, code, platform, samples, prompts_to_outputs, problem.metadata.get("func_name", None), problem, args)
                repaired_code = None
            if testcase != "":
                self.testcase_generation_num += 1
                probe_metadata = get_metadata(
                    self.put_run_exec,
                    worker_id,
                    testcase,
                    checker_extend_code,
                    args.timeout,
                    self.default_error,
                )
                if (
                    metadata_is_property_assertion(probe_metadata)
                    or metadata_is_assertion_failure(probe_metadata)
                ):
                    generated_property_killed_bug = True
                    self.generated_bug_kill_num += 1
                    feedback = property_info.get("property_feedback")
                    if feedback:
                        probe_metadata = dict(probe_metadata)
                        probe_metadata["property_feedback"] = feedback
                self.append_property_trace({
                    **trace_context,
                    "event": "generated_property_probe",
                    "testcase": testcase,
                    "checker_extend_code": checker_extend_code,
                    "accepted_properties": property_info.get("accepted_properties"),
                    "probe_results": None,
                    "probe_metadata": probe_metadata,
                    "generated_property_killed_bug": generated_property_killed_bug,
                })
                if getattr(args, "property_require_generated_kill", False):
                    if generated_property_killed_bug:
                        repair_metadata = probe_metadata
                elif not metadata_is_input_only_wrong_answer(probe_metadata):
                    repair_metadata = probe_metadata
            if (
                repaired_code is None
                and getattr(args, "property_require_generated_kill", False)
                and not generated_property_killed_bug
            ):
                repaired_code = base_code
            if repaired_code is None:
                repaired_code, _ = self.repair_code(
                    worker_id,
                    question_content,
                    model_style,
                    checker_extend_code
                    if (
                        getattr(args, "property_repair_instrumented_code", False)
                        and generated_property_killed_bug
                    )
                    else base_code,
                    repair_metadata,
                    prompts_to_outputs,
                    samples,
                    True,
                    args,
                    {
                        "question_id": problem.question_id,
                        "question_title": getattr(problem, "question_title", ""),
                        "branch": "public_pass",
                        "repair_stage": "property_repair",
                        "generated_property_killed_bug": generated_property_killed_bug,
                        "accepted_property_count": len(property_info.get("accepted_properties") or []),
                    },
                    )
                if repaired_code and repaired_code != base_code:
                    self.repair_attempt_num += 1
            if (
                getattr(args, "property_select_generated_checks", False)
                and testcase != ""
                and property_info.get("accepted_properties")
                and generated_property_killed_bug
            ):
                repair_public_res, _ = self.put_run_exec(worker_id, samples, repaired_code, args.timeout)
                generated_res, generated_metadata, _ = self.run_generated_property_checks(
                    repaired_code,
                    property_info.get("accepted_properties"),
                    testcase,
                    args,
                )
                if all_tests_passed(repair_public_res) and all_tests_passed(generated_res):
                    self.generated_repair_check_pass_num += 1
                else:
                    self.generated_repair_check_reject_num += 1
                    repaired_code = base_code
            if repaired_code != base_code:
                if self.public_passes(worker_id, samples, repaired_code, metadata, args):
                    self.public_pass_branch_fixed_num += 1
                else:
                    self.final_public_gate_revert_num += 1
                    self.append_repair_trace({
                        "event": "final_public_gate_revert",
                        "worker_id": worker_id,
                        "question_id": problem.question_id,
                        "question_title": getattr(problem, "question_title", ""),
                        "branch": "public_pass",
                        "candidate_code": repaired_code,
                        "base_code": base_code,
                    })
                    repaired_code = base_code

        if getattr(args, "strip_final_asserts", False):
            repaired_code = strip_final_assertion_checks(repaired_code)
        if (
            not str(repaired_code).strip()
            or not candidate_is_complete_python(repaired_code)
            or not candidate_preserves_interface(repaired_code, metadata)
        ):
            self.final_structural_gate_revert_num += 1
            self.append_repair_trace({
                "event": "final_structural_gate_revert",
                "worker_id": worker_id,
                "question_id": problem.question_id,
                "question_title": getattr(problem, "question_title", ""),
                "candidate_code": repaired_code,
                "candidate_empty": not str(repaired_code).strip(),
                "candidate_complete": candidate_is_complete_python(repaired_code),
                "interface_ok": candidate_preserves_interface(repaired_code, metadata),
            })
            repaired_code = base_code
        self.prompts_answer_list[worker_id] = repaired_code
        self.active_workers -= 1
        if self.active_workers == 0:
            self.finished.set()


    def get_train_data(self, worker_id, question_content, code, public_grade, metadata, platform, problem, model_style, args, prompts_to_outputs):
        pass


    def our_method_pipeline(self, benchmark, model_style, args, check_metadata_list, prompts_to_outputs):
        global run_answer_list
        outputs = [
            [None for _ in range(args.codegen_n)]
            for _ in range(len(benchmark))
        ]
        
        threads = []
        worker_id = 0
        prompt_index_to_question_idx = []
        prompt_index_to_code_idx = []
        
        yes_num, no_num, strong_public_num = 0, 0, 0
        oracle_skip_full_pass_num = 0
        public_only_routing = getattr(args, "property_public_only_routing", False)
        oracle_skip_full_pass = (
            public_only_routing
            and getattr(args, "property_oracle_skip_full_pass", False)
        )
        for problem_idx, problem in tqdm(enumerate(benchmark)):
            for check_metadata_idx, check_metadata in enumerate(check_metadata_list):
                if problem.question_id == check_metadata['question_id']:
                    question_content = check_metadata["question_content"]
                    code_list = check_metadata["code_list"]
                    output_list = check_metadata["output_list"]
                    graded_list = check_metadata["graded_list"]
                    public_graded_list = check_metadata["public_graded_list"]
                    platform = check_metadata["platform"]
                    metadata = check_metadata["metadata"]

                    for code_idx in range(len(code_list)):
                        public_grade = public_graded_list[code_idx]
                        if public_only_routing:
                            if public_grade:
                                yes_num += 1
                            else:
                                no_num += 1
                                strong_public_num += 1
                            if oracle_skip_full_pass and graded_list[code_idx]:
                                oracle_skip_full_pass_num += 1
                                outputs[problem_idx][code_idx] = output_list[code_idx]
                                continue
                        elif graded_list[code_idx]:
                            yes_num += 1
                            outputs[problem_idx][code_idx] = output_list[code_idx]
                            continue
                        else:
                            no_num += 1
                        if not public_only_routing and public_grade == False:
                            strong_public_num += 1
                        
                        if self.thread_num == 1:
                            self.prompts_answer_list.append("")
                            run_answer_list.append("")
                            self.solve_one_problem(
                                worker_id,
                                question_content,
                                code_list[code_idx],
                                public_grade,
                                metadata[code_idx],
                                platform,
                                problem,
                                model_style,
                                args,
                                prompts_to_outputs,
                            )
                            outputs[problem_idx][code_idx] = self.prompts_answer_list[worker_id]
                            outputs[problem_idx][code_idx] = "```\n" + outputs[problem_idx][code_idx] + "\n```"
                            worker_id += 1
                        else:
                            t = threading.Thread(
                                target=self.solve_one_problem, 
                                args=(
                                    worker_id,
                                    question_content,
                                    code_list[code_idx],
                                    public_grade,
                                    metadata[code_idx],
                                    platform,
                                    problem,
                                    model_style,
                                    args,
                                    prompts_to_outputs,
                                )
                            )
                            worker_id += 1
                            self.prompts_answer_list.append("")
                            run_answer_list.append("")
                            prompt_index_to_question_idx.append(problem_idx)
                            prompt_index_to_code_idx.append(code_idx)
                            self.active_workers += 1
                            threads.append(t)
                            t.start()
                            
        print(
            "yes_num=", yes_num,
            "no_num=", no_num,
            "strong_public_num=", strong_public_num,
            "oracle_skip_full_pass_num=", oracle_skip_full_pass_num,
        )
        
        if self.thread_num != 1:
            pool = None
            prompts = []
            prompt_events = []
            prompts_worker_id = []
            prompt_first_enqueue_at = None
            run_exec_events = []
            run_exec_worker_id = []
            prompt_batch_size = max(1, int(getattr(args, "max_concurrency", 4)))
            prompt_flush_seconds = 2.0

            def run_exec_in_thread(worker_id, data, event):
                run_answer_list[worker_id] = run_exec(data[0], data[1], data[2])
                event.set()

            def flush_pending_prompts(reason):
                nonlocal prompts, prompt_events, prompts_worker_id, prompt_first_enqueue_at
                if not prompts:
                    return
                print(f"\nflushing {len(prompts)} LLM prompts ({reason})")
                try:
                    tmp_outputs = prompts_to_outputs(prompts)
                except Exception as exc:
                    print(f"LLM batch failed, resuming workers with empty outputs: {exc}")
                    tmp_outputs = [[""] for _ in prompts]
                for i, worker_id in enumerate(prompts_worker_id):
                    if i < len(tmp_outputs):
                        self.prompts_answer_list[worker_id] = tmp_outputs[i]
                    else:
                        self.prompts_answer_list[worker_id] = [""]
                for event in prompt_events:
                    event.set()
                prompts = []
                prompt_events = []
                prompts_worker_id = []
                prompt_first_enqueue_at = None
            
            while not self.finished.is_set():
                try:
                    # non-blocking get with up to 1 second timeout
                    worker_id, event_type, data, event = self.request_queue.get(timeout=1)
                    # print(f"\nmain thread handling request from worker {worker_id}: {event_type}")
                    if event_type == "prompts_to_code":
                        # print(f"main thread finished prompt preprocessing for worker {worker_id}")
                        prompts.append(data)
                        prompt_events.append(event)
                        prompts_worker_id.append(worker_id)
                        if prompt_first_enqueue_at is None:
                            prompt_first_enqueue_at = time.time()
                        self.request_queue.task_done()
                    elif event_type == "run_exec":
                        # run_exec/check_correctness already creates child processes
                        # for isolation and timeout handling. Running it inside a
                        # multiprocessing.Pool daemon causes "daemonic processes are
                        # not allowed to have children", so dispatch it from a
                        # regular thread in the main process instead.
                        run_answer_list[worker_id] = self.run_answer_wait_flag
                        threading.Thread(
                            target=run_exec_in_thread,
                            args=(worker_id, data, event),
                            daemon=True,
                        ).start()
                        self.request_queue.task_done()
                    else:
                        # print("main thread has no pending tasks")
                        pass
                except Empty:
                    # queue is empty
                    # print("main thread has no pending tasks")
                    if self.active_workers == 0:
                        break

                new_run_exec_events = []
                new_run_exec_worker_id = []
                for i in range(len(run_exec_worker_id)):
                    tmp_worker_id = run_exec_worker_id[i]
                    if run_answer_list[tmp_worker_id] == self.run_answer_wait_flag:
                        new_run_exec_worker_id.append(run_exec_worker_id[i])
                        new_run_exec_events.append(run_exec_events[i])
                    else:
                        run_exec_events[i].set()
                run_exec_events = []
                run_exec_worker_id = []
                for i in range(len(new_run_exec_worker_id)):
                    run_exec_events.append(new_run_exec_events[i])
                    run_exec_worker_id.append(new_run_exec_worker_id[i])

                if prompts:
                    waited = time.time() - prompt_first_enqueue_at if prompt_first_enqueue_at else 0.0
                    if len(prompts) >= prompt_batch_size:
                        flush_pending_prompts("batch_size")
                    elif self.active_workers == len(prompts):
                        flush_pending_prompts("all_active_workers_waiting")
                    elif waited >= prompt_flush_seconds:
                        flush_pending_prompts("flush_interval")
                
            if pool is not None:
                pool.close()
                pool.join()
            flush_pending_prompts("shutdown")
            
            for prompt_idx, output in enumerate(self.prompts_answer_list):
                question_idx = prompt_index_to_question_idx[prompt_idx]
                code_idx = prompt_index_to_code_idx[prompt_idx]
                outputs[question_idx][code_idx] = "```\n" + output + "\n```"
        
        print(
            "testcase_inputer_generation_num=",
            self.testcase_generation_num,
            "property_generation_num=",
            self.property_generation_num,
            "inputer_retry_num=",
            self.inputer_retry_num,
            "inputer_failed_script_num=",
            self.inputer_failed_script_num,
            "inputer_smoke_success_num=",
            self.inputer_smoke_success_num,
            "inputer_smoke_failed_num=",
            self.inputer_smoke_failed_num,
            "inputer_batch_success_num=",
            self.inputer_batch_success_num,
            "inputer_batch_failed_num=",
            self.inputer_batch_failed_num,
            "inputer_final_success_num=",
            self.inputer_final_success_num,
            "inputer_final_empty_num=",
            self.inputer_final_empty_num,
            "generated_bug_kill_num=",
            self.generated_bug_kill_num,
            "generated_repair_check_pass_num=",
            self.generated_repair_check_pass_num,
            "generated_repair_check_reject_num=",
            self.generated_repair_check_reject_num,
            "property_public_pass_accept_num=",
            self.property_public_pass_accept_num,
            "property_public_fail_assert_accept_num=",
            self.property_public_fail_assert_accept_num,
            "property_public_direct_seed_accept_num=",
            self.property_public_direct_seed_accept_num,
            "property_public_direct_seed_reject_num=",
            self.property_public_direct_seed_reject_num,
            "property_public_direct_seed_missing_num=",
            self.property_public_direct_seed_missing_num,
            "property_proposed_num=",
            self.property_proposed_num,
            "property_accepted_num=",
            self.property_accepted_num,
            "property_rejected_num=",
            self.property_rejected_num,
            "property_error_num=",
            self.property_error_num,
            "property_instrumented_num=",
            self.property_instrumented_num,
            "repair_attempt_num=",
            self.repair_attempt_num,
            "public_fail_branch_num=",
            self.public_fail_branch_num,
            "public_fail_branch_fixed_num=",
            self.public_fail_branch_fixed_num,
            "public_pass_branch_num=",
            self.public_pass_branch_num,
            "public_pass_branch_fixed_num=",
            self.public_pass_branch_fixed_num,
            "final_public_gate_revert_num=",
            self.final_public_gate_revert_num,
            "final_structural_gate_revert_num=",
            self.final_structural_gate_revert_num,
        )
        if self.checker_code_log:
            accepted_num = sum(1 for item in self.checker_code_log if item.get("accepted"))
            rejected_total = sum(int(item.get("rejected", 0)) for item in self.checker_code_log)
            property_error_num = sum(1 for item in self.checker_code_log if item.get("property_error_code") is not None)
            proposal_total = sum(int(item.get("proposal_count", 0)) for item in self.checker_code_log)
            instrumented_total = sum(int(item.get("instrumented_count", 0)) for item in self.checker_code_log)
            print(
                "property_checker_stats=",
                {
                    "attempted": len(self.checker_code_log),
                    "accepted": accepted_num,
                    "rejected": rejected_total,
                    "property_error": property_error_num,
                    "proposed": proposal_total,
                    "instrumented": instrumented_total,
                },
            )
        
        return outputs

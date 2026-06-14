import os
import torch
import argparse

from lcb_runner.utils.scenarios import Scenario, TestCaseForRepair


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `lm_styles.py`",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="If you have a local model, specify it here in conjunction with --model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="trust_remote_code option used in huggingface models",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--testcaseforrepair",
        type=TestCaseForRepair,
        default=TestCaseForRepair.allcase,
        help="what type of test case to use for selfrepair",
    )
    parser.add_argument(
        "--selfdebug",
        action="store_true",
        help="gradually debug to perform selfrepair",
    )
    parser.add_argument(
        "--repairbase",
        type=Scenario,
        default=Scenario.codegeneration,
    )
    parser.add_argument(
        "--repairbase_path",
        type=str,
        default="",
        help="Optional explicit eval_all JSON path to use as repair/checkerextend base instead of deriving one from --repairbase.",
    )
    parser.add_argument(
        "--not_fast",
        action="store_true",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--release_version",
        type=str,
        default="release_latest",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--cot_code_execution",
        action="store_true",
        help="whether to use CoT in code execution scenario",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--codegen_n",
        type=int,
        default=10,
        help="Number of samples for which code generation was run (used to map the code generation file during self-repair)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )
    parser.add_argument(
        "--repairbase_temperature",
        type=float,
        default=None,
        help="Temperature suffix of the codegeneration/selfrepair eval file used as repair base. Defaults to --temperature.",
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=2000, help="Max tokens for sampling"
    )
    parser.add_argument(
        "--multiprocess",
        default=0,
        type=int,
        help="Number of processes to use for generation (vllm runs do not use this)",
    )
    parser.add_argument(
        "--stop",
        default="###",
        type=str,
        help="Stop token (use `,` to separate multiple tokens)",
    )
    parser.add_argument("--continue_existing", action="store_true")
    parser.add_argument("--continue_existing_with_eval", action="store_true")
    parser.add_argument(
        "--use_cache", action="store_true", help="Use cache for generation"
    )
    parser.add_argument(
        "--cache_batch_size", type=int, default=100, help="Batch size for caching"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--debug_size",
        type=int,
        default=15,
        help="Number of benchmark instances to keep when --debug is set.",
    )
    parser.add_argument(
        "--debug_indices",
        type=str,
        default="",
        help="Comma-separated benchmark indices or ranges to keep when --debug is set, e.g. 7,27,140-145.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the results")
    parser.add_argument(
        "--no_retry_empty_code",
        action="store_true",
        help="Disable automatic code-only retry when codegeneration output contains no extractable code.",
    )
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=12,
        help="Number of processes to use for evaluation",
    )
    parser.add_argument("--timeout", type=int, default=6, help="Timeout for evaluation")
    parser.add_argument(
        "--openai_timeout", type=int, default=90, help="Timeout for requests to OpenAI"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for OpenAI-compatible runners. Defaults to the environment named by --api_key_env.",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="INF_API_KEY",
        help="Environment variable used as API key for OpenAI-compatible local/API runners.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=None,
        help="Override the model's OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--chat_template_kwargs",
        type=str,
        default=None,
        help="JSON object passed through extra_body.chat_template_kwargs for compatible APIs.",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Set chat_template_kwargs.enable_thinking=false for compatible Qwen-style APIs.",
    )
    parser.add_argument(
        "--no_verify_ssl",
        action="store_true",
        help="Disable SSL certificate verification for OpenAI-compatible runners.",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="Maximum concurrent requests for OpenAI-compatible local/API runners.",
    )
    parser.add_argument(
        "--checker_mode",
        type=str,
        default="property",
        choices=["property", "legacy"],
        help="checkerextend strategy: property generation/filtering/injection or the legacy direct checker prompt.",
    )
    parser.add_argument(
        "--repair_max_attempts",
        type=int,
        default=5,
        help="Maximum repair attempts per failing solution in selfrepair/checkerextend.",
    )
    parser.add_argument(
        "--save_repair_trace",
        action="store_true",
        help=(
            "Save intermediate repair candidates and gate decisions to a JSONL "
            "trace file next to the scenario output."
        ),
    )
    parser.add_argument(
        "--save_property_trace",
        action="store_true",
        help=(
            "Save per-sample property generation, merge, instrumentation, and "
            "input-generator traces to a JSONL file next to the scenario output."
        ),
    )
    parser.add_argument(
        "--checker_max_attempts",
        type=int,
        default=3,
        help="Maximum checker/property-generation attempts per failing solution in checkerextend.",
    )
    parser.add_argument(
        "--property_fallback_legacy",
        action="store_true",
        help="Allow property checkerextend to fall back to the legacy checker prompt when property attempts are rejected.",
    )
    parser.add_argument(
        "--prompt_extract_max_attempts",
        type=int,
        default=3,
        help="Maximum LLM retries when a prompt response does not contain extractable code/testcase.",
    )
    parser.add_argument(
        "--strip_final_asserts",
        action="store_true",
        help="Remove final assert/AssertionError checks from repaired code after using them as feedback.",
    )
    parser.add_argument(
        "--keep_failed_repairs",
        action="store_true",
        help="Diagnostic mode: keep the last repair candidate even when it still fails public tests. Do not use for no-regress final merges.",
    )
    parser.add_argument(
        "--property_select_generated_checks",
        action="store_true",
        help=(
            "For checkerextend, after repair, accept a public-passing candidate only "
            "if it also passes generated public-only property/input checks. Hidden/full "
            "evaluation is not used for this selection."
        ),
    )
    parser.add_argument(
        "--property_require_generated_kill",
        action="store_true",
        help=(
            "For public-passing buggy code, use generated property/input checks as "
            "repair feedback only when they turn the current buggy code into an "
            "explicit property/assertion failure."
        ),
    )
    parser.add_argument(
        "--property_prompt_variants",
        type=int,
        default=1,
        help=(
            "Number of independent property prompt families to run and merge. "
            "1=oracle only, 2=oracle+invariant, 3=oracle+invariant+boundary."
        ),
    )
    parser.add_argument(
        "--property_llm_merge",
        action="store_true",
        help=(
            "Ask the model to merge/filter generated property checker proposals "
            "before local instrumentation. This is still public-only and does not "
            "repair the solution."
        ),
    )
    parser.add_argument(
        "--property_llm_instrument",
        action="store_true",
        help=(
            "Ask the model to insert accepted property snippets into the candidate "
            "code instead of using the local AST instrumenter."
        ),
    )
    parser.add_argument(
        "--property_llm_merge_min_candidates",
        type=int,
        default=2,
        help=(
            "Minimum number of deduped property candidates required before calling "
            "the optional LLM merge prompt."
        ),
    )
    parser.add_argument(
        "--property_public_direct_fallback",
        action="store_true",
        help=(
            "For public-failing code, if property-feedback repair still fails public "
            "tests, try one direct public-feedback repair candidate and accept it only "
            "when it passes public tests."
        ),
    )
    parser.add_argument(
        "--property_public_fail_direct_only",
        action="store_true",
        help=(
            "For public-failing code, skip property generation and run direct "
            "public-feedback repair. Property/input generation is reserved for "
            "public-passing suspicious code."
        ),
    )
    parser.add_argument(
        "--property_public_direct_outputs_path",
        type=str,
        default="",
        help=(
            "Optional raw selfrepair output JSON used as a public-fail direct "
            "repair candidate source. A candidate is accepted only when it passes "
            "available public tests."
        ),
    )
    parser.add_argument(
        "--property_public_only_routing",
        action="store_true",
        help=(
            "Route checkerextend using only public-test visibility: public-failing "
            "code is repaired from public feedback, and public-passing code is only "
            "changed when generated property/input checks produce a public-visible "
            "property failure."
        ),
    )
    parser.add_argument(
        "--property_oracle_skip_full_pass",
        action="store_true",
        help=(
            "Experiment-only speedup: with public-only routing, use already computed "
            "full evaluation labels to skip candidates that are full-pass. This must "
            "not be reported as a strict public-only run."
        ),
    )
    parser.add_argument(
        "--property_repair_instrumented_code",
        action="store_true",
        help=(
            "When generated property/input checks kill public-passing buggy code, "
            "feed the property-instrumented code to repair so the checker logic is "
            "visible in the code context."
        ),
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=-1,
        help="Tensor parallel size for vllm",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching for vllm",
    )
    parser.add_argument(
        "--custom_output_file",
        type=str,
        default=None,
        help="Path to the custom output file used in `custom_evaluator.py`",
    )
    parser.add_argument(
        "--custom_output_save_name",
        type=str,
        default=None,
        help="Folder name to save the custom output results (output file folder modified if None)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype for vllm")
    # Added to avoid running extra generations (it's slow for reasoning models)
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )

    args = parser.parse_args()

    args.stop = args.stop.split(",")

    if args.tensor_parallel_size == -1:
        args.tensor_parallel_size = torch.cuda.device_count()

    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()

    return args


def test():
    args = get_args()
    print(args)


if __name__ == "__main__":
    test()

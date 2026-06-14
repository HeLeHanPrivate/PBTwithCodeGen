import os
import json

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
    get_public_sample_results,
    add_extra_samples
)


def get_repairbase_eval_all_path(args, model) -> str:
    explicit_path = getattr(args, "repairbase_path", "") or ""
    if explicit_path:
        return explicit_path
    exdirname = ""
    if str(args.release_version) in ("humaneval", "codecontests", "mbpp"):
        exdirname = "----" + str(args.release_version)
    testcase = f"_{args.testcaseforrepair}" if args.repairbase == Scenario.selfrepair else ""
    repairbase_temperature = getattr(args, "repairbase_temperature", None)
    if repairbase_temperature is None:
        repairbase_temperature = args.temperature
    return (
        f"output/{model.model_repr}{exdirname}/"
        f"{args.repairbase}{testcase}_{args.codegen_n}_{repairbase_temperature}_eval_all.json"
    )


def _parse_debug_indices(raw: str, benchmark_size: int) -> list[int]:
    indices = []
    seen = set()
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
        else:
            values = [int(chunk)]
        for idx in values:
            if idx < 0 or idx >= benchmark_size:
                raise ValueError(
                    f"--debug_indices contains out-of-range index {idx}; benchmark size is {benchmark_size}"
                )
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices


def _apply_debug_selection(args, benchmark):
    if not args.debug:
        return benchmark
    if args.debug_indices:
        selected_indices = _parse_debug_indices(args.debug_indices, len(benchmark))
        print(
            f"Running with {len(selected_indices)} selected instances in debug mode: {selected_indices}"
        )
        return [benchmark[idx] for idx in selected_indices]
    print(f"Running with {len(benchmark)} instances in debug mode")
    return benchmark[: args.debug_size]


def main():
    args = get_args()
    # import ipdb; ipdb.set_trace()
    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    benchmark = _apply_debug_selection(args, benchmark)
    # benchmark = benchmark[:20] # debug
    if args.custom_output_save_name is not None:
        output_path = get_output_path(args.custom_output_save_name, args)
    else:
        output_path = get_output_path(model.model_repr, args)
    args.output_path = output_path
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")
    # import ipdb; ipdb.set_trace()
    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"]
            # and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        # import ipdb; ipdb.set_trace()
        results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    # import ipdb; ipdb.set_trace()
    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    all_benchmark = None
    check_metadata_list = None
    if args.scenario == Scenario.testcasegeneration:
        with open(get_repairbase_eval_all_path(args, model)) as f:
            check_metadata_list = json.load(f)
        all_benchmark, format_prompt = build_prompt_benchmark(args)
        save_results, all_benchmark = add_extra_samples(save_results, check_metadata_list, all_benchmark)
        benchmark = all_benchmark
        _, combined_results = sort_and_extract_save_results(
            args.scenario, save_results
        )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration or args.scenario == Scenario.testcasegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair or args.scenario == Scenario.checkerextend:
            metadatas = metrics[2]

            with open(get_repairbase_eval_all_path(args, model)) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        # import ipdb; ipdb.set_trace()
        if args.scenario == Scenario.codegeneration or args.scenario == Scenario.testcasegeneration or args.scenario == Scenario.checkerextend or args.scenario == Scenario.selfrepair:
            if all_benchmark is None:
                all_benchmark, format_prompt = build_prompt_benchmark(args)
                all_benchmark = _apply_debug_selection(args, all_benchmark)
            save_eval_results, metrics = get_public_sample_results(save_eval_results, metrics, all_benchmark, save_results, check_metadata_list)
        
        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()

from typing import Union
import json

from lcb_runner.utils.scenarios import Scenario, TestCaseForRepair
from lcb_runner.lm_styles import LanguageModel
from lcb_runner.evaluation import (
    codegen_metrics,
    test_output_metrics,
    code_execution_metrics,
)

from lcb_runner.prompts import (
    format_prompt_generation,
    format_prompt_test_output,
    format_prompt_execution,
    format_prompt_execution_cot,
    format_prompt_self_repair,
    format_prompt_testcase_generate,
    format_prompt_checker_extend
)
from lcb_runner.utils.extraction_utils import (
    extract_code,
    extract_test_output_code,
    extract_execution_code,
    extract_testcase,
)

from lcb_runner.benchmarks import (
    CodeGenerationProblem,
    TestOutputPredictionProblem,
    CodeExecutionProblem,
    load_code_generation_dataset,
    load_code_generation_dataset_not_fast,
    load_test_prediction_dataset,
    load_code_execution_dataset,
)

# BenchMarkType = list[CodeGenerationProblem | TestOutputPredictionProblem]
BenchMarkType = list[
    Union[CodeGenerationProblem, CodeExecutionProblem, TestOutputPredictionProblem]
]


def build_prompt_benchmark(
    args,
) -> tuple[
    list[CodeExecutionProblem]
    | list[CodeGenerationProblem]
    | list[TestOutputPredictionProblem],
    callable,
]:
    scenario: Scenario = args.scenario

    if scenario == Scenario.codegeneration:
        not_fast: bool = args.not_fast
        if not_fast:
            benchmark = load_code_generation_dataset_not_fast(args.release_version)
        else:
            benchmark = load_code_generation_dataset(
                args.release_version,
                start_date=args.start_date,
                end_date=args.end_date
            )
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_generation
    elif scenario == Scenario.testoutputprediction:
        benchmark = load_test_prediction_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: (x.question_id, x.test_id))
        format_prompt = format_prompt_test_output
    elif scenario == Scenario.selfrepair:
        benchmark = load_code_generation_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_self_repair
    elif scenario == Scenario.codeexecution:
        cot_code_execution: bool = args.cot_code_execution
        benchmark = load_code_execution_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: int(x.id.split("_")[1]))
        if cot_code_execution:
            format_prompt = format_prompt_execution_cot
        else:
            format_prompt = format_prompt_execution
    elif scenario == Scenario.testcasegeneration:
        benchmark = load_code_generation_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_testcase_generate
    elif scenario == Scenario.checkerextend:
        benchmark = load_code_generation_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_checker_extend
    else:
        raise ValueError(f"Scenario {scenario} not implemented")
    return benchmark, format_prompt


def combine_results(
    scenario: Scenario,
    results: list[list[str]],
    model: LanguageModel,
    cot_code_execution: bool = False,
):
    if scenario == Scenario.codegeneration:
        combined_results = [
            (
                outputs_list,
                [extract_code(output, model.model_style) for output in outputs_list],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.testoutputprediction:
        combined_results = [
            (
                outputs_list,
                [
                    extract_test_output_code(output, model.model_style)
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.selfrepair:
        combined_results = [
            (
                [
                    output[0] if type(output) is list else output
                    for output in outputs_list
                ],
                [
                    (
                        extract_code(output[0], model.model_style)
                        if type(output) is list
                        else extract_code(output, model.model_style)
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.testcasegeneration:
        combined_results = [
            (
                [
                    output[0] if type(output) is list else output
                    for output in outputs_list
                ],
                [
                    (
                        extract_testcase(output[0], model.model_style)
                        if type(output) is list
                        else extract_testcase(output, model.model_style)
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.codeexecution:
        combined_results = [
            (
                outputs_list,
                [
                    extract_execution_code(
                        output, model.model_style, cot=cot_code_execution
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.checkerextend:
        combined_results = [
            (
                [
                    output[0] if type(output) is list else output
                    for output in outputs_list
                ],
                [
                    (
                        extract_code(output[0], model.model_style)
                        if type(output) is list
                        else extract_code(output, model.model_style)
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return combined_results


def sort_and_extract_save_results(scenario: Scenario, save_results: list[dict]):
    if scenario == Scenario.codegeneration:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]

    elif scenario == Scenario.testoutputprediction:
        save_results = sorted(
            save_results, key=lambda x: (x["question_id"], x["test_id"])
        )
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.selfrepair:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.testcasegeneration:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.codeexecution:
        save_results = sorted(save_results, key=lambda x: int(x["id"].split("_")[1]))
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.checkerextend:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return save_results, combined_results


def get_metrics(
    scenario: Scenario,
    args,
    benchmark: list[
        CodeGenerationProblem | CodeExecutionProblem | TestOutputPredictionProblem
    ],
    combined_results,
):
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    if scenario == Scenario.codegeneration or scenario == Scenario.selfrepair or scenario == Scenario.testcasegeneration or scenario == Scenario.checkerextend:
        metrics = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
        )

    elif args.scenario == Scenario.testoutputprediction:
        metrics = test_output_metrics(
            eval_samples,
            generations,
            k_list=[1, 5],
        )

    elif args.scenario == Scenario.codeexecution:
        metrics = code_execution_metrics(
            eval_samples,
            generations,
        )

    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    print(metrics[0]["pass@1"])

    return metrics


def check_testtype(testtype, platform):
    if type(testtype) is not str:
        if hasattr(testtype, "FUNCTIONAL"):
            testtype = 'functional'
        elif hasattr(testtype, "STDIN"):
            testtype = 'stdin'
        else:
            raise ValueError(f"Unknown testtype, {testtype}")
    if type(platform) is not str:
        if hasattr(platform, "LEETCODE"):
            platform = 'leetcode'
        elif hasattr(platform, "CODEFORCES"):
            platform = 'codeforces'
        elif hasattr(platform, "ATCODER"):
            platform = 'atcoder'
        elif hasattr(platform, "HUMANEVAL"):
            platform = 'humaneval'
        else:
            raise ValueError(f"Unknown platform, {platform}")
    if testtype == 'functional':
        if platform == 'humaneval' or platform == 'leetcode':
            return True
    if testtype == 'stdin':
        if platform == 'codeforces' or platform == 'atcoder':
            return True
    return False


def get_public_sample_results(check_detail_list: list, check_outline_list: list, benchmark: list, save_results: list, codegen_results: list):
    check_outline_dict = check_outline_list[1]
    total_num, only_public_total, error_num = 0, 0, 0
    for idx in range(len(benchmark)):
        public_test_cases_len = len(benchmark[idx].public_test_cases) + len(benchmark[idx].extra_test)
        public_test_cases_result = []
        only_public_test_cases_result = []
        for code_id in range(len(check_outline_dict[idx])):
            public_answer = check_outline_dict[idx][code_id][:public_test_cases_len]
            fla = True
            for i in public_answer:
                if i == False or i < 0:
                    fla = False
            public_test_cases_result.append(fla)
            only_public_answer = check_outline_dict[idx][code_id][:len(benchmark[idx].public_test_cases)]
            fla = True
            for i in only_public_answer:
                if i == False or i < 0:
                    fla = False
            only_public_test_cases_result.append(fla)
        if codegen_results is not None:
            if codegen_results[idx]["public_graded_list"] != only_public_test_cases_result:
                public_test_cases_result = codegen_results[idx]["public_graded_list"]
                only_public_test_cases_result = codegen_results[idx]["public_graded_list"]
                error_num += 1
        total_num += sum(public_test_cases_result)/len(public_test_cases_result)
        only_public_total += sum(only_public_test_cases_result)/len(public_test_cases_result)
        check_detail_list[idx]["public_graded_list"] = public_test_cases_result
        check_detail_list[idx]["only_public_graded_list"] = only_public_test_cases_result
        if "extra_test" in check_detail_list[idx].keys():
            for save_detail_idx, save_detail in enumerate(save_results):
                if check_detail_list[idx]['question_id'] == save_detail['question_id']:
                    if "extra_test" in save_detail.keys():
                        check_detail_list[idx]['extra_test'] = save_detail['extra_test']
                    else:
                        check_detail_list[idx]['extra_test'] = []
    check_outline_list[0]["public_pass@1"] = total_num/len(benchmark)
    check_outline_list[0]["only_public_pass@1"] = only_public_total/len(benchmark)
    print("public_pass@1", check_outline_list[0]["public_pass@1"], "  only_public_pass@1", check_outline_list[0]["only_public_pass@1"])
    print("error_num", error_num)
    return check_detail_list, check_outline_list


def add_extra_samples(save_detail_list: list, check_metadata_list: list, benchmark: list):
    num, tmp = 0, 0
    for idx in range(len(check_metadata_list)):
        for save_detail_idx, save_detail in enumerate(save_detail_list):
            if check_metadata_list[idx]['question_id'] == save_detail['question_id']:
                if "extra_test" not in save_detail.keys() or len(save_detail['extra_test']) == 0:
                    extra_test_key = "code_list"
                else:
                    extra_test_key = "extra_test"
                check_metadata_list[idx]['extra_test'] = []
                for extra_test_list in save_detail[extra_test_key]:
                    if type(extra_test_list) is list and len(extra_test_list)>=1:
                        check_metadata_list[idx]['extra_test'] += extra_test_list
                        num += 1
                    elif type(extra_test_list) is dict and "input" in extra_test_list.keys():
                        check_metadata_list[idx]['extra_test'] += save_detail[extra_test_key]
                        num += 1
                        break
                check_metadata_list[idx]['extra_test'] = [t for t in check_metadata_list[idx]['extra_test'] if check_testtype(t['testtype'], check_metadata_list[idx]['platform'])]
    print(f"{num} problems add extra_test")
    from lcb_runner.benchmarks.code_generation import Test
    for idx in range(len(benchmark)):
        for save_detail_idx, save_detail in enumerate(save_detail_list):
            if benchmark[idx].question_id == save_detail['question_id']:
                if "extra_test" not in save_detail.keys() or len(save_detail['extra_test']) == 0:
                    extra_test_key = "code_list"
                else:
                    extra_test_key = "extra_test"
                benchmark[idx].extra_test = []
                for extra_test_list in save_detail[extra_test_key]:
                    if type(extra_test_list) is list and len(extra_test_list)>=1:
                        benchmark[idx].extra_test += [Test(**i) for i in extra_test_list]
                        tmp += 1
                    elif type(extra_test_list) is dict and "input" in extra_test_list.keys():
                        benchmark[idx].extra_test += [Test(**i) for i in save_detail[extra_test_key]]
                        tmp += 1
                        break
                benchmark[idx].extra_test = [t for t in benchmark[idx].extra_test if check_testtype(t.testtype, benchmark[idx].platform)]
    assert tmp == num, tmp
    return check_metadata_list, benchmark
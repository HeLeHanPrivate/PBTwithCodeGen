#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.evaluation.pass_k_utils import extract_instance_results
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
from lcb_runner.evaluation.testing_util import run_test


def parse_indices(text: str, size: int) -> list[int]:
    if not text:
        return list(range(size))
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return sorted(set(i for i in indices if 0 <= i < size))


def public_grades(problem, results_for_problem):
    public_len = len(problem.public_test_cases)
    public_plus_extra_len = public_len + len(problem.extra_test)
    public_result = []
    only_public_result = []
    for generation_result in results_for_problem:
        public_answer = generation_result[:public_plus_extra_len]
        only_public_answer = generation_result[:public_len]
        public_result.append(bool(public_answer) and all(item is True for item in public_answer))
        only_public_result.append(bool(only_public_answer) and all(item is True for item in only_public_answer))
    return public_result, only_public_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_all")
    parser.add_argument("--release-version", default="release_v5")
    parser.add_argument("--indices", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-out", default="")
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num-process", type=int, default=12)
    args = parser.parse_args()

    source_path = Path(args.eval_all)
    data = json.loads(source_path.read_text())
    benchmark = load_code_generation_dataset(args.release_version)
    problem_by_question_id = {problem.question_id: problem for problem in benchmark}

    selected_indices = parse_indices(args.indices, len(data))
    selected_benchmark = []
    for i in selected_indices:
        question_id = data[i].get("question_id")
        if question_id not in problem_by_question_id:
            raise SystemExit(f"question_id not found in benchmark: {question_id}")
        selected_benchmark.append(problem_by_question_id[question_id])
    samples = [problem.get_evaluation_sample() for problem in selected_benchmark]
    generations = [[(data[i].get("code_list") or [""])[0]] for i in selected_indices]

    if args.num_process <= 1:
        results = {}
        metadata = {}
        final_metadata = []
        for idx, (sample, generation_list) in enumerate(zip(samples, generations)):
            results[idx] = []
            metadata[idx] = []
            for generation in generation_list:
                curr_res, curr_metadata = run_test(sample, test=generation, debug=False, timeout=args.timeout)
                results[idx].append(curr_res)
                metadata[idx].append(curr_metadata)
            final_metadata.append([json.dumps(item) for item in metadata[idx]])
        metrics = [compute_metrics_from_results(results), results, final_metadata]
    else:
        metrics = codegen_metrics(
            samples,
            generations,
            num_process_evaluate=args.num_process,
            timeout=args.timeout,
            debug=False,
        )
    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]

    updated_rows = []
    public_total = 0
    only_public_total = 0
    for local_idx, original_idx in enumerate(selected_indices):
        problem = selected_benchmark[local_idx]
        row = data[original_idx]
        output_list = row.get("output_list") or [""]
        code_list = row.get("code_list") or [""]
        public_graded, only_public_graded = public_grades(problem, metrics[1][local_idx])
        public_total += sum(public_graded) / len(public_graded)
        only_public_total += sum(only_public_graded) / len(only_public_graded)
        updated = problem.insert_output_evaluation(
            output_list,
            code_list,
            graded[local_idx],
            metadata=metadatas[local_idx],
            original_code_list=row.get("original_code_list", []),
        )
        updated["public_graded_list"] = public_graded
        updated["only_public_graded_list"] = only_public_graded
        updated_rows.append(updated)

    metrics[0]["public_pass@1"] = public_total / len(selected_indices) if selected_indices else 0
    metrics[0]["only_public_pass@1"] = only_public_total / len(selected_indices) if selected_indices else 0

    Path(args.out).write_text(json.dumps(updated_rows, indent=4))
    if args.eval_out:
        Path(args.eval_out).write_text(json.dumps(metrics, indent=4))
    print(json.dumps({
        "selected": len(selected_indices),
        "pass": sum(row["pass@1"] for row in updated_rows),
        "public_pass": metrics[0]["public_pass@1"],
        "only_public_pass": metrics[0]["only_public_pass@1"],
        "out": args.out,
        "eval_out": args.eval_out,
    }, indent=2))


if __name__ == "__main__":
    main()

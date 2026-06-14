import json
import argparse
import numpy as np
from datetime import datetime

from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.evaluation.pass_k_utils import (
    estimate_pass_at_k,
    compute_metrics_from_results,
)
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.utils.path_utils import get_eval_all_output_path
from collections import Counter


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `lm_styles.py`",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )

    parser.add_argument(
        "--eval_all_file",
        type=str,
        default=None,
        help="Alternative way to provide the evaluation file",
    )

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

    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform to filter the evaluation file",
    )

    args = parser.parse_args()

    if args.eval_all_file is None:
        model = LanguageModelStore[args.model]
        args.eval_all_file = get_eval_all_output_path(model, args)

    return args


def find_most_common_list(lst, public_graded_list=None, use_public_output=False):
    if use_public_output:
        tuple_lst = [tuple(lst[idx]) for idx in range(len(lst)) if public_graded_list[idx]]
    else:
        tuple_lst = [tuple(lst[idx]) for idx in range(len(lst))]
    counter = Counter(tuple_lst)
    most_common = counter.most_common(1)
    if most_common:
        return list(most_common[0][0])
    return []


def check_list(lst):
    for element in lst:
        if isinstance(element, int) and element < 0:
            return False
        if isinstance(element, bool) and element is False:
            return False
    return True


def compute_scores(args):
    eval_all_file = args.eval_all_file
    eval_file = eval_all_file.replace("_eval_all.json", "_eval.json")
    with open(eval_all_file, "r") as f:
        results = json.load(f)
    with open(eval_file, "r") as f:
        detail_results = json.load(f)[1]

    for res in results:
        res["contest_date"] = datetime.fromisoformat(res["contest_date"])

    if args.start_date is not None:
        args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        results = [
            result for result in results if args.start_date <= result["contest_date"]
        ]

    if args.end_date is not None:
        args.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        results = [
            result for result in results if result["contest_date"] <= args.end_date
        ]

    if args.platform is not None:
        results = [result for result in results if result["platform"] == args.platform]

    print(len(results))
    totals = [len(x["graded_list"]) for x in results]
    corrects = [sum(x["graded_list"]) for x in results]

    easy_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "easy"]
    med_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "medium"]
    hard_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "hard"]
    easy_corrects = [sum(x["graded_list"]) for x in results if x["difficulty"] == "easy"]
    med_corrects = [sum(x["graded_list"]) for x in results if x["difficulty"] == "medium"]
    hard_corrects = [sum(x["graded_list"]) for x in results if x["difficulty"] == "hard"]
    
    # debug
    
    # for k in [1, 2, 5, 10, 25, 50, 100, 150, 200]:
    #     print(
    #         f"Pass@{k} = ",
    #         estimate_pass_at_k(totals, corrects, k).mean(),
    #         # np.array(
    #         #     [estimate_pass_at_k(t, c, k) for t, c in zip(totals, corrects)]
    #         # ).mean(),
    #     )
    #     print(
    #         f"Easy Pass@{k} = ",
    #         estimate_pass_at_k(easy_totals, easy_corrects, k).mean(),
    #     )
    #     print(
    #         f"Medium Pass@{k} = ",
    #         estimate_pass_at_k(med_totals, med_corrects, k).mean(),
    #     )
    #     print(
    #         f"Hard Pass@{k} = ",
    #         estimate_pass_at_k(hard_totals, hard_corrects, k).mean(),
    #     )
    
    # try:
    #     public_corrects = [sum(x["public_graded_list"]) for x in results]
    #     public_easy_corrects = [sum(x["public_graded_list"]) for x in results if x["difficulty"] == "easy"]
    #     public_med_corrects = [sum(x["public_graded_list"]) for x in results if x["difficulty"] == "medium"]
    #     public_hard_corrects = [sum(x["public_graded_list"]) for x in results if x["difficulty"] == "hard"]
    #     print(
    #         f"After public Pass@1 = ",
    #         estimate_pass_at_k(totals, public_corrects, 1).mean(),
    #     )
    #     print(
    #         f"After public Easy Pass@1 = ",
    #         estimate_pass_at_k(easy_totals, public_easy_corrects, 1).mean(),
    #     )
    #     print(
    #         f"After public Medium Pass@1 = ",
    #         estimate_pass_at_k(med_totals, public_med_corrects, 1).mean(),
    #     )
    #     print(
    #         f"After public Hard Pass@1 = ",
    #         estimate_pass_at_k(hard_totals, public_hard_corrects, 1).mean(),
    #     )
    #     wa = [sum(["error_code\": -2" in y for y in x["metadata"]]) for x in results]
    #     easy_wa = [sum(["error_code\": -2" in y for y in x["metadata"]]) for x in results if x["difficulty"] == "easy"]
    #     med_wa = [sum(["error_code\": -2" in y for y in x["metadata"]]) for x in results if x["difficulty"] == "medium"]
    #     hard_wa = [sum(["error_code\": -2" in y for y in x["metadata"]]) for x in results if x["difficulty"] == "hard"]
    #     print(
    #         f"WA@1 = ",
    #         estimate_pass_at_k(totals, wa, 1).mean(),
    #     )
    #     print(
    #         f"Easy WA@1 = ",
    #         estimate_pass_at_k(easy_totals, easy_wa, 1).mean(),
    #     )
    #     print(
    #         f"Medium WA@1 = ",
    #         estimate_pass_at_k(med_totals, med_wa, 1).mean(),
    #     )
    #     print(
    #         f"Hard WA@1 = ",
    #         estimate_pass_at_k(hard_totals, hard_wa, 1).mean(),
    #     )

    #     for idx in detail_results.keys():
    #         if check_list(find_most_common_list(detail_results[idx], results[int(idx)]["public_graded_list"], use_public_output=False)):
    #             results[int(idx)]['cot_corrects'] = [True]
    #         else:
    #             results[int(idx)]['cot_corrects'] = [False]
    #     codecot_corrects = [sum(x["cot_corrects"]) for x in results]
    #     codecot_easy_corrects = [sum(x["cot_corrects"]) for x in results if x["difficulty"] == "easy"]
    #     codecot_med_corrects = [sum(x["cot_corrects"]) for x in results if x["difficulty"] == "medium"]
    #     codecot_hard_corrects = [sum(x["cot_corrects"]) for x in results if x["difficulty"] == "hard"]
    #     codecot_totals = [len(x["cot_corrects"]) for x in results]
    #     codecot_easy_totals = [len(x["cot_corrects"]) for x in results if x["difficulty"] == "easy"]
    #     codecot_med_totals = [len(x["cot_corrects"]) for x in results if x["difficulty"] == "medium"]
    #     codecot_hard_totals = [len(x["cot_corrects"]) for x in results if x["difficulty"] == "hard"]
    #     print(
    #         f"CodeCoT Pass@1 = ",
    #         estimate_pass_at_k(codecot_totals, codecot_corrects, 1).mean(),
    #     )
    #     print(
    #         f"CodeCoT Easy Pass@1 = ",
    #         estimate_pass_at_k(codecot_easy_totals, codecot_easy_corrects, 1).mean(),
    #     )
    #     print(
    #         f"CodeCoT Medium Pass@1 = ",
    #         estimate_pass_at_k(codecot_med_totals, codecot_med_corrects, 1).mean(),
    #     )
    #     print(
    #         f"CodeCoT Hard Pass@1 = ",
    #         estimate_pass_at_k(codecot_hard_totals, codecot_hard_corrects, 1).mean(),
    #     )
        
    #     for idx in detail_results.keys():
    #         if check_list(find_most_common_list(detail_results[idx], results[int(idx)]["public_graded_list"], use_public_output=True)):
    #             results[int(idx)]['cot_corrects'] = [True]
    #         else:
    #             results[int(idx)]['cot_corrects'] = [False]
    #     codecot_corrects = [sum(x["cot_corrects"]) for x in results]
    #     codecot_easy_corrects = [sum(x["cot_corrects"]) for x in results if x["difficulty"] == "easy"]
    #     codecot_med_corrects = [sum(x["cot_corrects"]) for x in results if x["difficulty"] == "medium"]
    #     codecot_hard_corrects = [sum(x["cot_corrects"]) for x in results if x["difficulty"] == "hard"]
    #     print(
    #         f"CodeCoT Publicused Pass@1 = ",
    #         estimate_pass_at_k(codecot_totals, codecot_corrects, 1).mean(),
    #     )
    #     print(
    #         f"CodeCoT Publicused Easy Pass@1 = ",
    #         estimate_pass_at_k(codecot_easy_totals, codecot_easy_corrects, 1).mean(),
    #     )
    #     print(
    #         f"CodeCoT Publicused Medium Pass@1 = ",
    #         estimate_pass_at_k(codecot_med_totals, codecot_med_corrects, 1).mean(),
    #     )
    #     print(
    #         f"CodeCoT Publicused Hard Pass@1 = ",
    #         estimate_pass_at_k(codecot_hard_totals, codecot_hard_corrects, 1).mean(),
    #     )
    # except:
    #     print("If this is not test for code generation, It is normal.")

    pass_1_list = [result["pass@1"] for result in results]
    print(f"Pass@1: {sum(pass_1_list) / len(pass_1_list)}")

    easy_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "easy"
    ]
    if len(easy_pass_1_list) > 0:
        print(f"Easy Pass@1: {sum(easy_pass_1_list) / len(easy_pass_1_list)}")

    medium_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "medium"
    ]
    if len(medium_pass_1_list) > 0:
        print(f"Medium Pass@1: {sum(medium_pass_1_list) / len(medium_pass_1_list)}")

    hard_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "hard"
    ]
    if len(hard_pass_1_list) > 0:
        print(f"Hard Pass@1: {sum(hard_pass_1_list) / len(hard_pass_1_list)}")


if __name__ == "__main__":
    compute_scores(get_parser())

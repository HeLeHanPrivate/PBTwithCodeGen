import argparse
import json
from copy import deepcopy


def passed(row: dict) -> bool:
    graded = row.get("graded_list") or []
    return bool(graded and graded[0])


def public_passed(row: dict) -> bool:
    graded = row.get("public_graded_list") or []
    return bool(graded and graded[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_eval_all")
    parser.add_argument("candidate_eval_all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument(
        "--mode",
        choices=("public_pass", "any_candidate"),
        default="public_pass",
        help="Which candidate rows to copy into the feedback repairbase.",
    )
    args = parser.parse_args()

    with open(args.base_eval_all) as fp:
        base_rows = json.load(fp)
    with open(args.candidate_eval_all) as fp:
        candidate_rows = json.load(fp)

    by_qid = {row["question_id"]: row for row in candidate_rows}
    merged = deepcopy(base_rows)
    replacements = []
    skipped = []

    for idx, row in enumerate(merged):
        qid = row["question_id"]
        candidate = by_qid.get(qid)
        if candidate is None:
            continue
        if passed(row):
            skipped.append({"idx": idx, "question_id": qid, "reason": "base_already_passed"})
            continue
        if args.mode == "public_pass" and not public_passed(candidate):
            skipped.append({"idx": idx, "question_id": qid, "reason": "candidate_public_not_passed"})
            continue
        merged[idx] = deepcopy(candidate)
        replacements.append(
            {
                "idx": idx,
                "question_id": qid,
                "title": row.get("question_title"),
                "candidate_passed": passed(candidate),
                "candidate_public_passed": public_passed(candidate),
            }
        )

    with open(args.out, "w") as fp:
        json.dump(merged, fp, indent=2)
    report = {
        "base": args.base_eval_all,
        "candidate": args.candidate_eval_all,
        "mode": args.mode,
        "total": len(merged),
        "replacement_count": len(replacements),
        "replacements": replacements,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    with open(args.report_out, "w") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

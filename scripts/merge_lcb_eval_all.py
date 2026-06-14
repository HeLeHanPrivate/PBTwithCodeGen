#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def load_rows(path):
    with open(path) as fp:
        rows = json.load(fp)
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a JSON list")
    return rows


def passed(row):
    values = row.get("graded_list") or []
    return bool(values and values[0])


def public_passed(row):
    values = row.get("public_graded_list") or []
    return bool(values and values[0])


def summarize(rows):
    total = len(rows)
    pass_count = sum(1 for row in rows if passed(row))
    hidden_fail = sum(1 for row in rows if not passed(row) and public_passed(row))
    return {
        "total": total,
        "pass": pass_count,
        "fail": total - pass_count,
        "pass_rate": pass_count / total if total else 0.0,
        "public_fail": total - pass_count - hidden_fail,
        "hidden_fail": hidden_fail,
    }


def cumulative_replacements(rows):
    replacements = []
    for idx, row in enumerate(rows):
        source = row.get("_merged_from")
        if not source:
            continue
        replacements.append({
            "idx": idx,
            "question_id": row.get("question_id"),
            "title": row.get("question_title"),
            "platform": row.get("platform"),
            "from": source,
            "replaced_passed": row.get("_merged_replaced_passed", False),
        })
    return replacements


def merge_rows(base_rows, candidate_paths, replace_passed=False):
    merged = [dict(row) for row in base_rows]
    index_by_qid = {row.get("question_id"): idx for idx, row in enumerate(merged)}
    replacements = []
    skipped = []

    for path in candidate_paths:
        for candidate in load_rows(path):
            qid = candidate.get("question_id")
            if qid not in index_by_qid:
                skipped.append({
                    "path": path,
                    "question_id": qid,
                    "reason": "not_in_base",
                })
                continue
            if not passed(candidate):
                skipped.append({
                    "path": path,
                    "question_id": qid,
                    "reason": "candidate_not_passed",
                })
                continue
            idx = index_by_qid[qid]
            current = merged[idx]
            if passed(current) and not replace_passed:
                skipped.append({
                    "path": path,
                    "question_id": qid,
                    "reason": "base_already_passed",
                })
                continue
            before_pass = passed(current)
            replacement = dict(candidate)
            replacement["_merged_from"] = path
            replacement["_merged_replaced_passed"] = before_pass
            merged[idx] = replacement
            replacements.append({
                "idx": idx,
                "question_id": qid,
                "title": candidate.get("question_title"),
                "platform": candidate.get("platform"),
                "from": path,
                "replaced_passed": before_pass,
            })
    return merged, replacements, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_eval_all")
    parser.add_argument("candidate_eval_all", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument(
        "--replace-passed",
        action="store_true",
        help="Allow a passing candidate to replace an already passing base row.",
    )
    args = parser.parse_args()

    base_rows = load_rows(args.base_eval_all)
    merged, replacements, skipped = merge_rows(
        base_rows,
        args.candidate_eval_all,
        replace_passed=args.replace_passed,
    )
    report = {
        "base": args.base_eval_all,
        "candidates": args.candidate_eval_all,
        "base_summary": summarize(base_rows),
        "merged_summary": summarize(merged),
        "replacements": replacements,
        "replacement_count": len(replacements),
        "cumulative_replacements": cumulative_replacements(merged),
        "cumulative_replacement_count": len(cumulative_replacements(merged)),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "base_summary": report["base_summary"],
        "merged_summary": report["merged_summary"],
        "replacement_count": len(replacements),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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


def summarize(rows):
    total = len(rows)
    pass_count = sum(1 for row in rows if passed(row))
    return {
        "total": total,
        "pass": pass_count,
        "fail": total - pass_count,
        "pass_rate": pass_count / total if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_eval_all")
    parser.add_argument("replacement_eval_all", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    rows = [dict(row) for row in load_rows(args.base_eval_all)]
    index_by_qid = {row.get("question_id"): idx for idx, row in enumerate(rows)}
    replacements = []
    skipped = []

    for path in args.replacement_eval_all:
        for replacement in load_rows(path):
            qid = replacement.get("question_id")
            idx = index_by_qid.get(qid)
            if idx is None:
                skipped.append({"path": path, "question_id": qid, "reason": "not_in_base"})
                continue
            before = rows[idx]
            updated = dict(replacement)
            updated["_updated_from"] = path
            updated["_updated_replaced_passed"] = passed(before)
            rows[idx] = updated
            replacements.append({
                "idx": idx,
                "question_id": qid,
                "path": path,
                "before_passed": passed(before),
                "after_passed": passed(updated),
            })

    report = {
        "base": args.base_eval_all,
        "replacements": args.replacement_eval_all,
        "base_summary": summarize(load_rows(args.base_eval_all)),
        "updated_summary": summarize(rows),
        "replacement_count": len(replacements),
        "replacement_rows": replacements,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "base_summary": report["base_summary"],
        "updated_summary": report["updated_summary"],
        "replacement_count": len(replacements),
        "skipped_count": len(skipped),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

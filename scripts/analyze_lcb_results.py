#!/usr/bin/env python
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: str):
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a JSON list")
    return rows


def parse_metadata(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def row_status(row):
    graded = row.get("graded_list") or []
    public = row.get("public_graded_list") or []
    passed = bool(graded and graded[0])
    public_passed = bool(public and public[0])
    metadata = parse_metadata((row.get("metadata") or [{}])[0])
    return passed, public_passed, metadata


def summarize(rows):
    counters = Counter()
    by_error = Counter()
    by_platform = Counter()
    by_difficulty = Counter()
    examples = defaultdict(list)

    for idx, row in enumerate(rows):
        passed, public_passed, metadata = row_status(row)
        platform = row.get("platform", "unknown")
        difficulty = row.get("difficulty", "unknown")
        by_platform[(platform, passed)] += 1
        by_difficulty[(difficulty, passed)] += 1
        if passed:
            counters["pass"] += 1
            continue

        counters["fail"] += 1
        counters["public_fail" if not public_passed else "hidden_fail"] += 1
        code = metadata.get("error_code", "missing")
        by_error[code] += 1
        if len(examples[code]) < 8:
            examples[code].append(
                {
                    "idx": idx,
                    "question_id": row.get("question_id"),
                    "title": row.get("question_title"),
                    "platform": platform,
                    "difficulty": difficulty,
                    "public_passed": public_passed,
                    "error_message": metadata.get("error_message") or metadata.get("error") or "",
                    "input_chars": len(str(metadata.get("inputs", ""))),
                }
            )

    return {
        "total": len(rows),
        "pass": counters["pass"],
        "fail": counters["fail"],
        "pass_rate": counters["pass"] / len(rows) if rows else 0.0,
        "public_fail": counters["public_fail"],
        "hidden_fail": counters["hidden_fail"],
        "error_counts": dict(sorted(by_error.items(), key=lambda item: str(item[0]))),
        "platform_pass_fail": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_platform.items())},
        "difficulty_pass_fail": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_difficulty.items())},
        "examples": dict(examples),
    }


def compare(before, after):
    rows = []
    for idx, (a, b) in enumerate(zip(before, after)):
        a_pass, a_public, a_meta = row_status(a)
        b_pass, b_public, b_meta = row_status(b)
        if a_pass == b_pass:
            continue
        rows.append(
            {
                "idx": idx,
                "question_id": a.get("question_id"),
                "title": a.get("question_title"),
                "before_pass": a_pass,
                "after_pass": b_pass,
                "before_public": a_public,
                "after_public": b_public,
                "before_error": a_meta.get("error_code"),
                "after_error": b_meta.get("error_code"),
                "before_message": a_meta.get("error_message") or a_meta.get("error") or "",
                "after_message": b_meta.get("error_message") or b_meta.get("error") or "",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_all")
    parser.add_argument("--compare", help="Optional second eval_all file to compare against")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args()

    rows = load_rows(args.eval_all)
    result = {"summary": summarize(rows)}
    if args.compare:
        other = load_rows(args.compare)
        result["delta"] = compare(rows, other)
        result["delta_counts"] = {
            "fixed": sum(1 for item in result["delta"] if item["after_pass"]),
            "regressed": sum(1 for item in result["delta"] if not item["after_pass"]),
        }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

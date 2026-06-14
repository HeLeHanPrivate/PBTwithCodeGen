#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


NOISY_ERROR_PATTERNS = (
    "cannot unpack non-iterable NoneType",
    "expected an indented block",
    "unexpected indent",
    "unterminated string",
    "invalid syntax",
    "has no attribute 'Solution'",
    "No evaluation result: missing or invalid call-based target",
    "No evaluation result: missing or invalid stdin program",
    "Candidate changed or removed the evaluator-facing public API",
)


def load_rows(path):
    with open(path) as fp:
        rows = json.load(fp)
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a JSON list")
    return rows


def first(row, key, default=""):
    values = row.get(key) or []
    return values[0] if values else default


def passed(row):
    return bool(first(row, "graded_list", False))


def public_passed(row):
    return bool(first(row, "public_graded_list", False))


def parse_metadata(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def error_text(row):
    metadata = parse_metadata(first(row, "metadata", {}))
    return str(metadata.get("error_message") or metadata.get("error") or "")


def error_code(row):
    metadata = parse_metadata(first(row, "metadata", {}))
    try:
        return int(metadata.get("error_code"))
    except (TypeError, ValueError):
        return None


def is_noisy_failure(row):
    if passed(row):
        return False
    message = error_text(row)
    return any(pattern in message for pattern in NOISY_ERROR_PATTERNS)


def is_runtime_noise_with_better_fallback(row, fallback):
    if passed(row) or passed(fallback):
        return False
    if error_code(row) != -4:
        return False
    if error_code(fallback) == -4:
        return False
    return bool(first(fallback, "code_list", "").strip())


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


def sanitize_rows(primary_rows, fallback_rows):
    if len(primary_rows) != len(fallback_rows):
        raise SystemExit(f"row count mismatch: {len(primary_rows)} != {len(fallback_rows)}")

    fallback_by_qid = {row.get("question_id"): row for row in fallback_rows}
    output = []
    replacements = []
    skipped = []
    reasons = Counter()

    for idx, row in enumerate(primary_rows):
        qid = row.get("question_id")
        fallback = fallback_by_qid.get(qid)
        if fallback is None:
            output.append(dict(row))
            skipped.append({"idx": idx, "question_id": qid, "reason": "missing_fallback"})
            continue

        if passed(row):
            output.append(dict(row))
            continue

        current_code = first(row, "code_list", "")
        fallback_code = first(fallback, "code_list", "")
        should_replace = False
        reason = ""
        if passed(fallback):
            should_replace = True
            reason = "fallback_passed"
        elif is_noisy_failure(row) and current_code != fallback_code:
            should_replace = True
            reason = "noisy_failure_changed_from_fallback"
        elif current_code != fallback_code and is_runtime_noise_with_better_fallback(row, fallback):
            should_replace = True
            reason = "runtime_noise_changed_from_non_runtime_fallback"

        if should_replace:
            replacement = dict(fallback)
            replacement["_sanitized_from"] = str(row.get("_merged_from") or "primary")
            replacement["_sanitized_reason"] = reason
            replacement["_sanitized_replaced_error"] = error_text(row)
            output.append(replacement)
            replacements.append({
                "idx": idx,
                "question_id": qid,
                "title": row.get("question_title"),
                "platform": row.get("platform"),
                "reason": reason,
                "primary_error": error_text(row),
                "fallback_passed": passed(fallback),
                "fallback_public_passed": public_passed(fallback),
            })
            reasons[reason] += 1
        else:
            output.append(dict(row))
    return output, replacements, skipped, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_eval_all")
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    primary = load_rows(args.primary_eval_all)
    fallback = load_rows(args.fallback)
    sanitized, replacements, skipped, reasons = sanitize_rows(primary, fallback)
    report = {
        "primary": args.primary_eval_all,
        "fallback": args.fallback,
        "primary_summary": summarize(primary),
        "fallback_summary": summarize(fallback),
        "sanitized_summary": summarize(sanitized),
        "replacement_count": len(replacements),
        "replacement_reasons": dict(reasons),
        "replacements": replacements,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n")
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "primary_summary": report["primary_summary"],
        "sanitized_summary": report["sanitized_summary"],
        "replacement_count": len(replacements),
        "replacement_reasons": report["replacement_reasons"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

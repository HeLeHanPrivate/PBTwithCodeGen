#!/usr/bin/env python
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path):
    with open(path) as fp:
        rows = json.load(fp)
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


def first_item(row, key, default=""):
    values = row.get(key) or []
    return values[0] if values else default


def status(row):
    passed = bool(first_item(row, "graded_list", False))
    public_passed = bool(first_item(row, "public_graded_list", False))
    metadata = parse_metadata(first_item(row, "metadata", {}))
    return passed, public_passed, metadata


def error_bucket(metadata):
    message = str(metadata.get("error_message") or metadata.get("error") or "")
    code = metadata.get("error_code", "missing")
    if "cannot unpack non-iterable NoneType" in message:
        return "none_unpack"
    if "expected an indented block" in message:
        return "expected_indented_block"
    if "unexpected indent" in message:
        return "unexpected_indent"
    if "unterminated string" in message:
        return "unterminated_string"
    if code == -3:
        return "tle"
    if code == -2:
        return "wrong_answer"
    if code == -4:
        return "runtime_error"
    return f"error_{code}"


def row_summary(idx, row, metadata, before_row=None):
    code = str(first_item(row, "code_list", ""))
    output = str(first_item(row, "output_list", ""))
    before_code = str(first_item(before_row, "code_list", "")) if before_row else None
    return {
        "idx": idx,
        "question_id": row.get("question_id"),
        "title": row.get("question_title"),
        "platform": row.get("platform"),
        "difficulty": row.get("difficulty"),
        "error_code": metadata.get("error_code"),
        "error_message": metadata.get("error_message") or metadata.get("error") or "",
        "bucket": error_bucket(metadata),
        "input_chars": len(str(metadata.get("inputs", ""))),
        "code_chars": len(code),
        "output_chars": len(output),
        "empty_code": not bool(code.strip()),
        "changed_from_before": None if before_code is None else before_code != code,
    }


def compact_indices(indices):
    if not indices:
        return ""
    indices = sorted(set(indices))
    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def build_report(after, before=None):
    before = before or [None] * len(after)
    if len(before) != len(after):
        raise SystemExit(f"before/after length mismatch: {len(before)} != {len(after)}")

    remaining = []
    hidden = []
    public = []
    fixed = []
    regressed = []
    changed_still_failing = []
    buckets = defaultdict(list)

    for idx, (after_row, before_row) in enumerate(zip(after, before)):
        after_pass, after_public, after_meta = status(after_row)
        before_pass = None
        before_public = None
        changed = None
        if before_row:
            before_pass, before_public, _ = status(before_row)
            changed = first_item(before_row, "code_list", "") != first_item(after_row, "code_list", "")
            if before_pass != after_pass:
                item = row_summary(idx, after_row, after_meta, before_row)
                item["before_pass"] = before_pass
                item["before_public"] = before_public
                if after_pass:
                    fixed.append(item)
                else:
                    regressed.append(item)

        if after_pass:
            continue

        item = row_summary(idx, after_row, after_meta, before_row)
        remaining.append(item)
        buckets[item["bucket"]].append(item)
        if after_public:
            hidden.append(item)
        else:
            public.append(item)
        if changed:
            changed_still_failing.append(item)

    bucket_counts = Counter(item["bucket"] for item in remaining)
    platform_counts = Counter((item["platform"], "hidden" if item in hidden else "public") for item in remaining)
    output = {
        "summary": {
            "total": len(after),
            "remaining_failures": len(remaining),
            "hidden_failures": len(hidden),
            "public_failures": len(public),
            "fixed": len(fixed),
            "regressed": len(regressed),
            "changed_still_failing": len(changed_still_failing),
            "empty_code_failures": sum(1 for item in remaining if item["empty_code"]),
            "bucket_counts": dict(bucket_counts),
            "platform_split": {f"{platform}:{kind}": count for (platform, kind), count in sorted(platform_counts.items())},
        },
        "rerun_indices": {
            "remaining_all": compact_indices(item["idx"] for item in remaining),
            "hidden_failures": compact_indices(item["idx"] for item in hidden),
            "public_failures": compact_indices(item["idx"] for item in public),
            "changed_still_failing": compact_indices(item["idx"] for item in changed_still_failing),
        },
        "top_buckets": {
            name: items[:20]
            for name, items in sorted(buckets.items(), key=lambda pair: (-len(pair[1]), pair[0]))
        },
        "hidden_failures": hidden,
        "fixed": fixed,
        "regressed": regressed,
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("after_eval_all")
    parser.add_argument("--before", help="Optional baseline eval_all for delta-oriented fields")
    parser.add_argument("--out", help="Output JSON path")
    args = parser.parse_args()

    after = load_rows(args.after_eval_all)
    before = load_rows(args.before) if args.before else None
    report = build_report(after, before)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

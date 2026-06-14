#!/usr/bin/env python
import argparse
import glob
import json
from pathlib import Path


def load_rows(path):
    with open(path) as fp:
        rows = json.load(fp)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected list")
    return rows


def passed(row):
    values = row.get("graded_list") or []
    return bool(values and values[0])


def public_passed(row):
    values = row.get("public_graded_list") or []
    return bool(values and values[0])


def summarize(path, baseline_rows=None):
    rows = load_rows(path)
    total = len(rows)
    pass_count = sum(1 for row in rows if passed(row))
    hidden_fail = sum(1 for row in rows if not passed(row) and public_passed(row))
    public_fail = total - pass_count - hidden_fail
    item = {
        "path": path,
        "total": total,
        "pass": pass_count,
        "fail": total - pass_count,
        "pass_rate": pass_count / total if total else 0.0,
        "public_fail": public_fail,
        "hidden_fail": hidden_fail,
    }
    if baseline_rows and len(baseline_rows) == total:
        fixed = 0
        regressed = 0
        for before, after in zip(baseline_rows, rows):
            before_pass = passed(before)
            after_pass = passed(after)
            if not before_pass and after_pass:
                fixed += 1
            elif before_pass and not after_pass:
                regressed += 1
        item["fixed_vs_baseline"] = fixed
        item["regressed_vs_baseline"] = regressed
        item["net_vs_baseline"] = fixed - regressed
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", default="output/Qwen3.6-27B/*_eval_all.json")
    parser.add_argument("--baseline", help="Optional baseline eval_all for fixed/regressed counts")
    parser.add_argument("--out", help="Optional JSON output path")
    parser.add_argument("--min-total", type=int, default=0, help="Ignore eval files with fewer rows")
    args = parser.parse_args()

    baseline_rows = load_rows(args.baseline) if args.baseline else None
    rows = []
    for path in sorted(glob.glob(args.pattern)):
        try:
            item = summarize(path, baseline_rows)
            if item.get("total", 0) >= args.min_total:
                rows.append(item)
        except Exception as exc:
            rows.append({"path": path, "error": str(exc)})

    rows.sort(key=lambda item: (item.get("pass_rate", -1), item.get("pass", -1)), reverse=True)
    result = {"best": rows[0] if rows else None, "results": rows}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

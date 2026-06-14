#!/usr/bin/env python3
"""
Analyze checkerextend results vs direct repair baseline.

Usage:
    python scripts/analyze_checkerextend_vs_direct.py \
        --codegen_eval output/DeepSeek-R1-Distill-Qwen-14B/Scenario.codegeneration_1_1.0_eval_all.json \
        --checker_eval output/DeepSeek-R1-Distill-Qwen-14B/Scenario.checkerextend_1_1.0_eval_all.json
"""

import json
import argparse
from pathlib import Path


def load_eval_all(path: str):
    with open(path) as f:
        return json.load(f)


def analyze(codegen_eval_path: str, checker_eval_path: str):
    codegen = load_eval_all(codegen_eval_path)
    checker = load_eval_all(checker_eval_path)

    # Build question_id -> entry maps
    codegen_map = {e["question_id"]: e for e in codegen}
    checker_map = {e["question_id"]: e for e in checker}

    total = len(codegen_map)
    assert total == len(checker_map), "Mismatch in number of problems"

    # Counters
    baseline_pass = 0          # codegen full pass
    checker_pass = 0           # checkerextend full pass
    public_pass_baseline = 0   # codegen public pass (but full fail)
    public_fail_baseline = 0   # codegen public fail

    # Breakdowns
    fixed_by_checker = 0       # full fail -> full pass
    broken_by_checker = 0      # full pass -> full fail (regression)
    unchanged_pass = 0         # full pass -> full pass
    unchanged_fail = 0         # full fail -> full fail

    # For public-pass baseline: how many did checker fix?
    public_pass_fixed = 0
    public_pass_unchanged = 0
    public_pass_broken = 0

    # For public-fail baseline: how many did checker fix?
    public_fail_fixed = 0
    public_fail_unchanged = 0
    public_fail_broken = 0

    for qid, cg in codegen_map.items():
        ch = checker_map[qid]

        cg_graded = cg.get("graded_list", [False])
        ch_graded = ch.get("graded_list", [False])
        cg_public = cg.get("public_graded_list", [False])
        ch_public = ch.get("public_graded_list", [False])

        cg_pass = bool(cg_graded[0]) if cg_graded else False
        ch_pass = bool(ch_graded[0]) if ch_graded else False
        cg_pub = bool(cg_public[0]) if cg_public else False
        ch_pub = bool(ch_public[0]) if ch_public else False

        if cg_pass:
            baseline_pass += 1
        if ch_pass:
            checker_pass += 1
        if cg_pub and not cg_pass:
            public_pass_baseline += 1
        if not cg_pub:
            public_fail_baseline += 1

        if cg_pass and ch_pass:
            unchanged_pass += 1
        elif not cg_pass and not ch_pass:
            unchanged_fail += 1
        elif not cg_pass and ch_pass:
            fixed_by_checker += 1
        elif cg_pass and not ch_pass:
            broken_by_checker += 1

        if cg_pub and not cg_pass:
            # public-pass at baseline
            if ch_pass:
                public_pass_fixed += 1
            elif ch_pub and not ch_pass:
                public_pass_unchanged += 1
            elif not ch_pub:
                public_pass_broken += 1
        elif not cg_pub:
            # public-fail at baseline
            if ch_pass:
                public_fail_fixed += 1
            elif ch_pub and not ch_pass:
                public_fail_unchanged += 1
            elif not ch_pub:
                public_fail_unchanged += 1

    print("=" * 60)
    print("Checkerextend vs Codegen Baseline")
    print("=" * 60)
    print(f"Total problems:               {total}")
    print(f"Baseline (codegen) pass@1:    {baseline_pass} / {total} = {baseline_pass/total:.2%}")
    print(f"Checkerextend pass@1:         {checker_pass} / {total} = {checker_pass/total:.2%}")
    print(f"Absolute improvement:         +{checker_pass - baseline_pass} ({(checker_pass - baseline_pass)/total:+.2%})")
    print(f"Relative improvement:         {(checker_pass - baseline_pass)/max(baseline_pass,1):+.1%}")
    print()
    print("Transition matrix (codegen -> checkerextend):")
    print(f"  Pass -> Pass:   {unchanged_pass:4d}")
    print(f"  Fail -> Pass:   {fixed_by_checker:4d}  (+)")
    print(f"  Pass -> Fail:   {broken_by_checker:4d}  (-)")
    print(f"  Fail -> Fail:   {unchanged_fail:4d}")
    print()
    print("=" * 60)
    print("Breakdown by baseline public-test status")
    print("=" * 60)
    print(f"Baseline public-pass (but full-fail): {public_pass_baseline}")
    print(f"  -> fixed by checkerextend:          {public_pass_fixed} ({public_pass_fixed/max(public_pass_baseline,1):.1%})")
    print(f"  -> still full-fail:                 {public_pass_unchanged + public_pass_broken}")
    print()
    print(f"Baseline public-fail:                 {public_fail_baseline}")
    print(f"  -> fixed by checkerextend:          {public_fail_fixed} ({public_fail_fixed/max(public_fail_baseline,1):.1%})")
    print(f"  -> still fail:                      {public_fail_baseline - public_fail_fixed}")
    print()
    print("=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("Public-pass cohort is where property-based repair operates.")
    print("Public-fail cohort is where direct repair operates.")
    print(f"Property repair success rate: {public_pass_fixed/max(public_pass_baseline,1):.1%}")
    print(f"Direct repair success rate:   {public_fail_fixed/max(public_fail_baseline,1):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codegen_eval", required=True)
    parser.add_argument("--checker_eval", required=True)
    args = parser.parse_args()
    analyze(args.codegen_eval, args.checker_eval)

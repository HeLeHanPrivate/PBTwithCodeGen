#!/usr/bin/env python3
"""Summarize codegen + checkerextend pipeline results."""
import json
import sys
from pathlib import Path


def load_eval_all(path: str):
    with open(path) as f:
        return json.load(f)


def summarize(eval_all, name):
    total = len(eval_all)
    public_pass = sum(1 for e in eval_all if e.get("public_graded_list", [False])[0])
    private_pass = sum(1 for e in eval_all if e.get("graded_list", [False])[0])
    only_public_pass = sum(1 for e in eval_all if e.get("only_public_graded_list", [False])[0])
    pass_at_1 = sum(e.get("pass@1", 0) for e in eval_all) / total if total else 0
    return {
        "name": name,
        "total": total,
        "public_pass": public_pass,
        "public_pass_rate": public_pass / total if total else 0,
        "private_pass": private_pass,
        "private_pass_rate": private_pass / total if total else 0,
        "only_public_pass": only_public_pass,
        "only_public_pass_rate": only_public_pass / total if total else 0,
        "pass_at_1": pass_at_1,
    }


def main():
    output_dir = Path("output/DeepSeek-R1-Distill-Qwen-14B")
    step1_path = output_dir / "Scenario.codegeneration_1_1.0_eval_all.json"
    step2_path = output_dir / "Scenario.checkerextend_1_1.0_eval_all.json"

    if not step1_path.exists():
        print(f"Step 1 file not found: {step1_path}")
        sys.exit(1)

    step1 = summarize(load_eval_all(str(step1_path)), "Step 1: Code Generation")
    print_summary(step1)

    if step2_path.exists():
        step2 = summarize(load_eval_all(str(step2_path)), "Step 2: Checkerextend")
        print_summary(step2)

        # Delta
        step1_data = load_eval_all(str(step1_path))
        step2_data = load_eval_all(str(step2_path))
        fixed = sum(
            1
            for e1, e2 in zip(step1_data, step2_data)
            if not e1.get("graded_list", [False])[0] and e2.get("graded_list", [False])[0]
        )
        regressed = sum(
            1
            for e1, e2 in zip(step1_data, step2_data)
            if e1.get("graded_list", [False])[0] and not e2.get("graded_list", [False])[0]
        )
        print(f"\nStep 2 delta vs Step 1:")
        print(f"  Fixed (fail -> pass): {fixed}")
        print(f"  Regressed (pass -> fail): {regressed}")
        print(f"  Net gain: {fixed - regressed}")
    else:
        print(f"\nStep 2 file not found yet: {step2_path}")


def print_summary(s):
    print(f"\n{s['name']}")
    print(f"  Total problems: {s['total']}")
    print(f"  Public pass:    {s['public_pass']} / {s['total']} ({s['public_pass_rate']:.1%})")
    print(f"  Private pass:   {s['private_pass']} / {s['total']} ({s['private_pass_rate']:.1%})")
    print(f"  Only-public:    {s['only_public_pass']} / {s['total']} ({s['only_public_pass_rate']:.1%})")
    print(f"  pass@1:         {s['pass_at_1']:.3f}")


if __name__ == "__main__":
    main()

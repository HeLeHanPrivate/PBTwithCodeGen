#!/usr/bin/env python3
"""
Detailed analysis of checkerextend vs codegeneration results.
Reads the two eval_all.json files and prints breakdown statistics.
"""

import json
import sys
from pathlib import Path


def load_eval_all(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/DeepSeek-R1-Distill-Qwen-14B"),
                        help="Directory containing the eval_all JSON files.")
    args = parser.parse_args()
    root = args.root
    gen = load_eval_all(root / "Scenario.codegeneration_1_1.0_eval_all.json")
    chk = load_eval_all(root / "Scenario.checkerextend_1_1.0_eval_all.json")

    assert len(gen) == len(chk) == 880, f"Expected 880 problems, got {len(gen)}, {len(chk)}"

    total = len(gen)

    # Aggregate per-problem boolean outcomes
    gen_public = []
    gen_private = []
    chk_public = []
    chk_private = []

    for g, c in zip(gen, chk):
        # eval_all stores graded_list, public_graded_list, only_public_graded_list as lists of bool
        g_priv = bool(g.get("graded_list", [False])[0])
        g_pub = bool(g.get("public_graded_list", [False])[0])
        c_priv = bool(c.get("graded_list", [False])[0])
        c_pub = bool(c.get("public_graded_list", [False])[0])
        gen_public.append(g_pub)
        gen_private.append(g_priv)
        chk_public.append(c_pub)
        chk_private.append(c_priv)

    # Basic totals
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Total problems: {total}")
    print()
    print("Step 1 (Code Generation)")
    print(f"  Public pass:  {sum(gen_public):>3} / {total} ({100*sum(gen_public)/total:.2f}%)")
    print(f"  Private pass: {sum(gen_private):>3} / {total} ({100*sum(gen_private)/total:.2f}%)")
    print(f"  pass@1:       {sum(gen_private)/total:.4f}")
    print()
    print("Step 2 (Checkerextend)")
    print(f"  Public pass:  {sum(chk_public):>3} / {total} ({100*sum(chk_public)/total:.2f}%)")
    print(f"  Private pass: {sum(chk_private):>3} / {total} ({100*sum(chk_private)/total:.2f}%)")
    print(f"  pass@1:       {sum(chk_private)/total:.4f}")
    print()

    # Delta
    fixed = sum(1 for gp, cp in zip(gen_private, chk_private) if not gp and cp)
    regressed = sum(1 for gp, cp in zip(gen_private, chk_private) if gp and not cp)
    unchanged_pass = sum(1 for gp, cp in zip(gen_private, chk_private) if gp and cp)
    unchanged_fail = sum(1 for gp, cp in zip(gen_private, chk_private) if not gp and not cp)

    print("=" * 70)
    print("PRIVATE PASS DELTA (Step 1 -> Step 2)")
    print("=" * 70)
    print(f"  Remained pass:        {unchanged_pass:>3}")
    print(f"  Fixed (fail -> pass): {fixed:>3}")
    print(f"  Regressed (pass -> fail): {regressed:>3}")
    print(f"  Remained fail:        {unchanged_fail:>3}")
    print(f"  Net gain: {fixed - regressed} problems")
    print()

    # Routing matrix based on public/private combinations
    cases = {
        "public_fail_private_fail": [],
        "public_fail_private_pass": [],
        "public_pass_private_fail": [],
        "public_pass_private_pass": [],
    }
    labels = {
        "public_fail_private_fail": "Public FAIL,  Private FAIL",
        "public_fail_private_pass": "Public FAIL,  Private PASS",
        "public_pass_private_fail": "Public PASS,  Private FAIL",
        "public_pass_private_pass": "Public PASS,  Private PASS",
    }

    for i, (g_pub, g_priv, c_pub, c_priv) in enumerate(zip(gen_public, gen_private, chk_public, chk_private)):
        key = f"{'public_pass' if g_pub else 'public_fail'}_private_{'pass' if g_priv else 'fail'}"
        cases[key].append({
            "idx": i,
            "gen_public": g_pub,
            "gen_private": g_priv,
            "chk_public": c_pub,
            "chk_private": c_priv,
        })

    print("=" * 70)
    print("BREAKDOWN BY STEP 1 STATUS")
    print("=" * 70)
    for key in ["public_fail_private_fail", "public_fail_private_pass", "public_pass_private_fail", "public_pass_private_pass"]:
        items = cases[key]
        n = len(items)
        if n == 0:
            continue
        priv_fixed = sum(1 for it in items if not it["gen_private"] and it["chk_private"])
        priv_passed = sum(1 for it in items if it["chk_private"])
        pub_passed = sum(1 for it in items if it["chk_public"])
        pub_fixed = sum(1 for it in items if not it["gen_public"] and it["chk_public"])
        print(f"\n{labels[key]}: {n} problems")
        print(f"  -> private pass after Step 2: {priv_passed:>3} / {n} ({100*priv_passed/n:.2f}%)")
        print(f"     of which newly fixed:      {priv_fixed:>3}")
        print(f"  -> public pass after Step 2:  {pub_passed:>3} / {n} ({100*pub_passed/n:.2f}%)")
        print(f"     of which newly fixed:      {pub_fixed:>3}")

    # Detailed transition matrix: gen_private x chk_private
    print("\n" + "=" * 70)
    print("PRIVATE PASS TRANSITION MATRIX")
    print("=" * 70)
    print("                 Step 2")
    print("Step 1       Fail    Pass    Total")
    fail_fail = sum(1 for gp, cp in zip(gen_private, chk_private) if not gp and not cp)
    fail_pass = sum(1 for gp, cp in zip(gen_private, chk_private) if not gp and cp)
    pass_fail = sum(1 for gp, cp in zip(gen_private, chk_private) if gp and not cp)
    pass_pass = sum(1 for gp, cp in zip(gen_private, chk_private) if gp and cp)
    print(f"  Fail      {fail_fail:>4}    {fail_pass:>4}    {fail_fail + fail_pass:>4}")
    print(f"  Pass      {pass_fail:>4}    {pass_pass:>4}    {pass_fail + pass_pass:>4}")
    print(f"  Total     {fail_fail + pass_fail:>4}    {fail_pass + pass_pass:>4}    {total:>4}")

    # Public pass transition matrix
    print("\n" + "=" * 70)
    print("PUBLIC PASS TRANSITION MATRIX")
    print("=" * 70)
    print("                 Step 2")
    print("Step 1       Fail    Pass    Total")
    p_fail_fail = sum(1 for gp, cp in zip(gen_public, chk_public) if not gp and not cp)
    p_fail_pass = sum(1 for gp, cp in zip(gen_public, chk_public) if not gp and cp)
    p_pass_fail = sum(1 for gp, cp in zip(gen_public, chk_public) if gp and not cp)
    p_pass_pass = sum(1 for gp, cp in zip(gen_public, chk_public) if gp and cp)
    print(f"  Fail      {p_fail_fail:>4}    {p_fail_pass:>4}    {p_fail_fail + p_fail_pass:>4}")
    print(f"  Pass      {p_pass_fail:>4}    {p_pass_pass:>4}    {p_pass_fail + p_pass_pass:>4}")
    print(f"  Total     {p_fail_fail + p_pass_fail:>4}    {p_fail_pass + p_pass_pass:>4}    {total:>4}")

    # Difficulty breakdown if available
    print("\n" + "=" * 70)
    print("RESULTS BY DIFFICULTY")
    print("=" * 70)
    by_diff = {}
    for g, c in zip(gen, chk):
        diff = g.get("difficulty", "unknown")
        by_diff.setdefault(diff, {"total": 0, "gen_priv_pass": 0, "chk_priv_pass": 0})
        by_diff[diff]["total"] += 1
        if g.get("graded_list", [False])[0]:
            by_diff[diff]["gen_priv_pass"] += 1
        if c.get("graded_list", [False])[0]:
            by_diff[diff]["chk_priv_pass"] += 1

    for diff in sorted(by_diff.keys(), key=lambda x: ("easy", "medium", "hard").index(x) if x in ("easy", "medium", "hard") else 99):
        stats = by_diff[diff]
        print(f"\n  {diff}: {stats['total']} problems")
        print(f"    Step 1 private pass: {stats['gen_priv_pass']:>3} ({100*stats['gen_priv_pass']/stats['total']:.2f}%)")
        print(f"    Step 2 private pass: {stats['chk_priv_pass']:>3} ({100*stats['chk_priv_pass']/stats['total']:.2f}%)")
        print(f"    Delta: +{stats['chk_priv_pass'] - stats['gen_priv_pass']} problems")

    # Platform breakdown
    print("\n" + "=" * 70)
    print("RESULTS BY PLATFORM")
    print("=" * 70)
    by_plat = {}
    for g, c in zip(gen, chk):
        plat = g.get("platform", "unknown")
        by_plat.setdefault(plat, {"total": 0, "gen_priv_pass": 0, "chk_priv_pass": 0})
        by_plat[plat]["total"] += 1
        if g.get("graded_list", [False])[0]:
            by_plat[plat]["gen_priv_pass"] += 1
        if c.get("graded_list", [False])[0]:
            by_plat[plat]["chk_priv_pass"] += 1

    for plat in sorted(by_plat.keys(), key=lambda p: -by_plat[p]["total"]):
        stats = by_plat[plat]
        print(f"\n  {plat}: {stats['total']} problems")
        print(f"    Step 1 private pass: {stats['gen_priv_pass']:>3} ({100*stats['gen_priv_pass']/stats['total']:.2f}%)")
        print(f"    Step 2 private pass: {stats['chk_priv_pass']:>3} ({100*stats['chk_priv_pass']/stats['total']:.2f}%)")
        print(f"    Delta: +{stats['chk_priv_pass'] - stats['gen_priv_pass']} problems")

    # Sample of fixed and regressed problems
    print("\n" + "=" * 70)
    print("SAMPLES")
    print("=" * 70)
    fixed_ids = [
        (gen[i]["question_id"], gen[i]["question_title"])
        for i in range(total)
        if not gen_private[i] and chk_private[i]
    ]
    regressed_ids = [
        (gen[i]["question_id"], gen[i]["question_title"])
        for i in range(total)
        if gen_private[i] and not chk_private[i]
    ]
    print(f"\nFixed examples (first 10 of {len(fixed_ids)}):")
    for qid, title in fixed_ids[:10]:
        print(f"  {qid}: {title}")
    print(f"\nRegressed examples (all {len(regressed_ids)}):")
    for qid, title in regressed_ids:
        print(f"  {qid}: {title}")


if __name__ == "__main__":
    main()

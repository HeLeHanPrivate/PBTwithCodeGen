#!/usr/bin/env python3
"""
Detailed analysis of checkerextend including internal step statistics.
"""

import json
import re
from pathlib import Path


def load_eval_all(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def parse_internal_counters(log_path: Path) -> dict:
    """Extract the final counter lines from the checkerextend log."""
    text = log_path.read_text()
    lines = text.splitlines()

    # Find the last line with property_checker_stats; counters are on the previous line.
    stats_line = None
    counter_line = None
    for i in range(len(lines) - 1, -1, -1):
        if "property_checker_stats=" in lines[i]:
            stats_line = lines[i]
            if i > 0:
                counter_line = lines[i - 1]
            break

    if not counter_line or not stats_line:
        return {}

    match = re.search(
        r"testcase_inputer_generation_num=\s*(\d+)\s+"
        r"property_generation_num=\s*(\d+)\s+"
        r"inputer_retry_num=\s*(\d+)\s+"
        r"inputer_failed_script_num=\s*(\d+)\s+"
        r"generated_bug_kill_num=\s*(\d+)\s+"
        r"generated_repair_check_pass_num=\s*(\d+)\s+"
        r"generated_repair_check_reject_num=\s*(\d+)\s+"
        r"property_public_pass_accept_num=\s*(\d+)\s+"
        r"property_public_fail_assert_accept_num=\s*(\d+)\s+"
        r"property_public_direct_seed_accept_num=\s*(\d+)\s+"
        r"property_public_direct_seed_reject_num=\s*(\d+)\s+"
        r"property_public_direct_seed_missing_num=\s*(\d+)",
        counter_line,
    )
    if not match:
        return {}

    stats_match = re.search(r"property_checker_stats=\s*(\{[^}]+\})", stats_line)
    if not stats_match:
        return {}

    stats = json.loads(stats_match.group(1).replace("'", '"'))
    groups = match.groups()
    return {
        "testcase_inputer_generation_num": int(groups[0]),
        "property_generation_num": int(groups[1]),
        "inputer_retry_num": int(groups[2]),
        "inputer_failed_script_num": int(groups[3]),
        "generated_bug_kill_num": int(groups[4]),
        "generated_repair_check_pass_num": int(groups[5]),
        "generated_repair_check_reject_num": int(groups[6]),
        "property_public_pass_accept_num": int(groups[7]),
        "property_public_fail_assert_accept_num": int(groups[8]),
        "property_public_direct_seed_accept_num": int(groups[9]),
        "property_public_direct_seed_reject_num": int(groups[10]),
        "property_public_direct_seed_missing_num": int(groups[11]),
        "property_checker_stats": stats,
    }


def parse_routing_numbers(log_path: Path) -> dict:
    text = log_path.read_text()
    for line in reversed(text.splitlines()):
        if "yes_num=" in line and "oracle_skip_full_pass_num=" in line:
            match = re.search(
                r"yes_num=\s*(\d+)\s+no_num=\s*(\d+)\s+"
                r"strong_public_num=\s*(\d+)\s+oracle_skip_full_pass_num=\s*(\d+)",
                line,
            )
            if match:
                return {
                    "yes_num": int(match.group(1)),
                    "no_num": int(match.group(2)),
                    "strong_public_num": int(match.group(3)),
                    "oracle_skip_full_pass_num": int(match.group(4)),
                }
    return {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/DeepSeek-R1-Distill-Qwen-14B"),
                        help="Directory containing the eval_all JSON files.")
    parser.add_argument("--log", type=Path, default=Path("logs/step2_checkerextend.log"),
                        help="Path to the checkerextend log file.")
    args = parser.parse_args()
    root = args.root
    log_path = args.log

    gen = load_eval_all(root / "Scenario.codegeneration_1_1.0_eval_all.json")
    chk = load_eval_all(root / "Scenario.checkerextend_1_1.0_eval_all.json")
    total = len(gen)

    gen_private = [bool(g.get("graded_list", [False])[0]) for g in gen]
    gen_public = [bool(g.get("public_graded_list", [False])[0]) for g in gen]
    chk_private = [bool(c.get("graded_list", [False])[0]) for c in chk]
    chk_public = [bool(c.get("public_graded_list", [False])[0]) for c in chk]

    counters = parse_internal_counters(log_path)
    routing = parse_routing_numbers(log_path)

    print("=" * 70)
    print("DETAILED CHECKEREXTEND ANALYSIS")
    print("=" * 70)
    print()

    # Routing
    print("1. ROUTING (which problems entered checkerextend)")
    print("-" * 70)
    if routing:
        yes = routing["yes_num"]
        no = routing["no_num"]
        skip = routing["oracle_skip_full_pass_num"]
        print(f"   Step 1 public pass:            {yes:>3} problems")
        print(f"     - private pass (oracle skip): {skip:>3}")
        print(f"     - private fail (enter repair): {yes - skip:>3}")
        print(f"   Step 1 public fail (enter repair): {no:>3} problems")
        print(f"   Total entering checkerextend:  {no + yes - skip:>3} problems")
    print()

    # Overall
    print("2. OVERALL PASS RATES")
    print("-" * 70)
    print(f"   Step 1 private pass: {sum(gen_private):>3}/{total} ({100*sum(gen_private)/total:.2f}%)")
    print(f"   Step 2 private pass: {sum(chk_private):>3}/{total} ({100*sum(chk_private)/total:.2f}%)")
    print(f"   Step 1 public pass:  {sum(gen_public):>3}/{total} ({100*sum(gen_public)/total:.2f}%)")
    print(f"   Step 2 public pass:  {sum(chk_public):>3}/{total} ({100*sum(chk_public)/total:.2f}%)")
    print()

    # Delta
    fixed = sum(1 for gp, cp in zip(gen_private, chk_private) if not gp and cp)
    regressed = sum(1 for gp, cp in zip(gen_private, chk_private) if gp and not cp)
    print("3. DELTA (Step 1 -> Step 2)")
    print("-" * 70)
    print(f"   Fixed:      {fixed:>3}")
    print(f"   Regressed:  {regressed:>3}")
    print(f"   Net gain:   {fixed - regressed:>3}")
    print()

    # Internal step stats
    print("4. CHECKEREXTEND INTERNAL STEP SUCCESS RATES")
    print("-" * 70)
    if counters:
        stats = counters["property_checker_stats"]
        attempted = stats.get("attempted", 0)
        proposed = stats.get("proposed", 0)
        accepted = stats.get("accepted", 0)
        rejected = stats.get("rejected", 0)
        prop_err = stats.get("property_error", 0)
        instrumented = stats.get("instrumented", 0)

        print(f"   Property generation (per problem):")
        print(f"     Problems attempted:           {attempted:>3}")
        print(f"     Properties proposed:          {proposed:>3}  ({proposed/attempted:.2f} per problem)")
        print(f"     Properties accepted:          {accepted:>3}  ({100*accepted/proposed:.1f}% of proposed)")
        print(f"     Properties rejected:          {rejected:>3}  ({100*rejected/proposed:.1f}% of proposed)")
        print(f"     Properties with exec error:   {prop_err:>3}  ({100*prop_err/proposed:.1f}% of proposed)")
        print(f"     Codes instrumented:           {instrumented:>3}  ({100*instrumented/attempted:.1f}% of attempted)")
        print()

        print(f"   Testcase inputer generation:")
        print(f"     Inputer generation calls:     {counters['testcase_inputer_generation_num']:>3}")
        print(f"     Failed scripts:               {counters['inputer_failed_script_num']:>3}  "
              f"({100*counters['inputer_failed_script_num']/max(1,counters['testcase_inputer_generation_num']):.1f}% of calls)")
        print(f"     Retries:                      {counters['inputer_retry_num']:>3}")
        print()

        print(f"   Generated property effectiveness:")
        print(f"     Generated bug kills:          {counters['generated_bug_kill_num']:>3}")
        print(f"     Generated repair check pass:  {counters['generated_repair_check_pass_num']:>3}")
        print(f"     Generated repair check reject:{counters['generated_repair_check_reject_num']:>3}")
        print()

        print(f"   Branch acceptance:")
        print(f"     Public-pass branch accepted:  {counters['property_public_pass_accept_num']:>3}")
        print(f"     Public-fail assert accepted:  {counters['property_public_fail_assert_accept_num']:>3}")
        print(f"     Direct seed accept:           {counters['property_public_direct_seed_accept_num']:>3}")
        print()

    # Breakdown by Step 1 status
    print("5. SUCCESS RATE BY STEP 1 STATUS")
    print("-" * 70)
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
        cases[key].append((c_priv, c_pub))

    for key in ["public_fail_private_fail", "public_fail_private_pass", "public_pass_private_fail", "public_pass_private_pass"]:
        items = cases[key]
        n = len(items)
        if n == 0:
            continue
        priv_pass = sum(1 for cp, _ in items if cp)
        pub_pass = sum(1 for _, pp in items if pp)
        print(f"\n   {labels[key]}: {n:>3} problems")
        print(f"     -> private pass after Step 2: {priv_pass:>3} ({100*priv_pass/n:>5.1f}%)")
        print(f"     -> public pass after Step 2:  {pub_pass:>3} ({100*pub_pass/n:>5.1f}%)")

    # Difficulty
    print("\n6. BY DIFFICULTY")
    print("-" * 70)
    by_diff = {}
    for g, c in zip(gen, chk):
        diff = g.get("difficulty", "unknown")
        by_diff.setdefault(diff, {"total": 0, "g_priv": 0, "c_priv": 0})
        by_diff[diff]["total"] += 1
        if g.get("graded_list", [False])[0]:
            by_diff[diff]["g_priv"] += 1
        if c.get("graded_list", [False])[0]:
            by_diff[diff]["c_priv"] += 1
    for diff in sorted(by_diff.keys(), key=lambda x: ("easy", "medium", "hard").index(x) if x in ("easy", "medium", "hard") else 99):
        s = by_diff[diff]
        print(f"   {diff:>6}: {s['total']:>3} | Step1 {100*s['g_priv']/s['total']:>5.1f}% | Step2 {100*s['c_priv']/s['total']:>5.1f}% | Δ{s['c_priv']-s['g_priv']:+d}")

    # Platform
    print("\n7. BY PLATFORM")
    print("-" * 70)
    by_plat = {}
    for g, c in zip(gen, chk):
        plat = g.get("platform", "unknown")
        by_plat.setdefault(plat, {"total": 0, "g_priv": 0, "c_priv": 0})
        by_plat[plat]["total"] += 1
        if g.get("graded_list", [False])[0]:
            by_plat[plat]["g_priv"] += 1
        if c.get("graded_list", [False])[0]:
            by_plat[plat]["c_priv"] += 1
    for plat in sorted(by_plat.keys(), key=lambda p: -by_plat[p]["total"]):
        s = by_plat[plat]
        print(f"   {plat:>10}: {s['total']:>3} | Step1 {100*s['g_priv']/s['total']:>5.1f}% | Step2 {100*s['c_priv']/s['total']:>5.1f}% | Δ{s['c_priv']-s['g_priv']:+d}")

    # Direct repair comparison note
    print("\n8. DIRECT REPAIR COMPARISON")
    print("-" * 70)
    print("   No direct-repair baseline was run in this pipeline.")
    print("   The closest proxy is the public-fail branch repair rate:")
    pub_fail_n = len(cases["public_fail_private_fail"])
    pub_fail_fixed = sum(1 for cp, _ in cases["public_fail_private_fail"] if cp)
    print(f"   - {pub_fail_fixed}/{pub_fail_n} ({100*pub_fail_fixed/pub_fail_n:.1f}%) of public-fail problems were fixed")
    print("   To compare vs direct repair, run the same model with --checker_mode=legacy or")
    print("   --property_public_fail_direct_only and evaluate the resulting eval_all.json.")


if __name__ == "__main__":
    main()

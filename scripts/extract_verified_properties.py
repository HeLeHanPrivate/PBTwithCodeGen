#!/usr/bin/env python
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts.property_generation import (
    extract_property_assertions,
    format_prompt_verified_property_extraction,
    instrument_code_with_properties,
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


def parse_metadata(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def select_verified_repairs(after_rows, before_rows=None, limit=None):
    before_by_qid = {row.get("question_id"): row for row in before_rows or []}
    selected = []
    for index, after in enumerate(after_rows):
        if not passed(after):
            continue
        before = before_by_qid.get(after.get("question_id"))
        if before is not None and passed(before):
            continue
        buggy_code = ""
        if before:
            buggy_code = first(before, "code_list", "") or first(before, "output_list", "")
        if not buggy_code:
            buggy_code = first(after, "original_code_list", "")
        fixed_code = first(after, "code_list", "") or first(after, "output_list", "")
        if not fixed_code.strip():
            continue
        selected.append((index, before, after, buggy_code, fixed_code))
        if limit and len(selected) >= limit:
            break
    return selected


def prompt_key(prompt):
    return json.dumps(prompt, ensure_ascii=False) if isinstance(prompt, list) else str(prompt)


async def call_prompts(args, prompts):
    api_key = os.environ.get(args.api_key_env, "EMPTY")
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), trust_env=False)
    client = AsyncOpenAI(api_key=api_key, base_url=args.api_base_url, http_client=http_client)
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    async def one(prompt):
        async with semaphore:
            response = await client.chat.completions.create(
                model=args.model,
                messages=prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or ""

    try:
        return await asyncio.gather(*(one(prompt) for prompt in prompts))
    finally:
        await client.close()
        await http_client.aclose()


def build_items(args):
    after_rows = load_rows(args.after_eval_all)
    before_rows = load_rows(args.before) if args.before else None
    selected = select_verified_repairs(after_rows, before_rows, args.limit)

    items = []
    prompts = []
    for index, before, after, buggy_code, fixed_code in selected:
        metadata = first(before, "metadata", "{}") if before else first(after, "metadata", "{}")
        prompt = format_prompt_verified_property_extraction(
            after.get("question_content", ""),
            LMStyle.LocalAPI,
            buggy_code,
            fixed_code,
            metadata,
        )
        item = {
            "idx": index,
            "question_id": after.get("question_id"),
            "title": after.get("question_title"),
            "platform": after.get("platform"),
            "difficulty": after.get("difficulty"),
            "buggy_metadata": parse_metadata(metadata),
            "buggy_code_chars": len(buggy_code),
            "fixed_code_chars": len(fixed_code),
            "prompt": prompt,
            "prompt_key": prompt_key(prompt),
            "_buggy_code": buggy_code,
            "_fixed_code": fixed_code,
        }
        items.append(item)
        prompts.append(prompt)
    return items, prompts


def attach_outputs(items, outputs):
    for item, output in zip(items, outputs):
        properties = extract_property_assertions(output, LMStyle.LocalAPI)
        fixed_code = item.get("_fixed_code", "")
        item["model_output"] = output
        item["properties"] = properties
        item["property_count"] = len(properties)
        if fixed_code:
            _, accepted = instrument_code_with_properties(fixed_code, properties)
            item["instrumented_fixed_count"] = len(accepted)


def strip_internal_fields(items, include_code=False):
    for item in items:
        buggy_code = item.pop("_buggy_code", "")
        fixed_code = item.pop("_fixed_code", "")
        if include_code:
            item["buggy_code"] = buggy_code
            item["fixed_code"] = fixed_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("after_eval_all")
    parser.add_argument("--before", help="Baseline eval_all. When set, only after-pass / before-fail rows are selected.")
    parser.add_argument("--out", default="output/verified_properties.json")
    parser.add_argument("--prompts-out", help="Optional JSONL path containing only prompts and metadata.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--call-api", action="store_true", help="Call the configured OpenAI-compatible API.")
    parser.add_argument("--include-code", action="store_true", help="Include buggy/fixed code in the output JSON.")
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="INF_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    items, prompts = build_items(args)
    if args.call_api and prompts:
        outputs = asyncio.run(call_prompts(args, prompts))
        attach_outputs(items, outputs)
    strip_internal_fields(items, args.include_code)

    summary = {
        "after_eval_all": args.after_eval_all,
        "before": args.before,
        "selected": len(items),
        "called_api": bool(args.call_api),
        "with_properties": sum(1 for item in items if item.get("property_count", 0) > 0),
    }
    result = {"summary": summary, "items": items}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.prompts_out:
        Path(args.prompts_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.prompts_out, "w") as fp:
            for item in items:
                fp.write(json.dumps({
                    "question_id": item["question_id"],
                    "title": item["title"],
                    "prompt": item["prompt"],
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

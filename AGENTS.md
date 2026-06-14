# Repository Guide

## Project Purpose

This repository is a LiveCodeBench-based code generation and repair project. The main code lives under `lcb_runner/`, with shell scripts under `script/` for running multi-step generation, repair, and evaluation workflows.

## Top-Level Layout

- `README.md`: basic environment setup, dataset/model preparation notes, and quick-start commands.
- `requirements.txt`: Python dependencies for the benchmark runner.
- `lcb_runner/`: main Python package for benchmark loading, prompt construction, model execution, parsing, repair, and evaluation.
- `script/`: runnable shell entrypoints for baseline and quick-run experiments.
- `main_datasets/`: intended local dataset location. In the current checkout it may be empty or populated outside Git.
- `test_api.py`: small API/test helper at the repository root.
- `stress_gpu.py`: GPU stress/testing helper.

## Main `lcb_runner/` Structure

- `lcb_runner/benchmarks/`: dataset/scenario loaders and benchmark objects.
  - `code_generation.py`: code generation benchmark setup and dataset path handling.
  - `code_execution.py`: code execution benchmark support.
  - `test_output_prediction.py`: test output prediction benchmark support.
- `lcb_runner/prompts/`: prompt templates and prompt construction for different scenarios.
  - Includes prompts for code generation, self repair, checker/testcase generation, and test output prediction.
- `lcb_runner/runner/`: model runner implementations and CLI entrypoint.
  - `main.py`: main module used by `python -m lcb_runner.runner.main`.
  - `parser.py`: command-line argument definitions.
  - `base_runner.py`: common runner interface/behavior.
  - Provider-specific runners include OpenAI, Claude, Gemini, Cohere, Mistral, DeepSeek, Fireworks, vLLM, and local API variants.
  - `our_method.py`, `scenario_router.py`, and `runner_utils.py` contain project-specific orchestration logic.
- `lcb_runner/evaluation/`: execution, metric computation, pass@k helpers, score aggregation, and old-result checks.
- `lcb_runner/utils/`: shared utilities for extraction, multiprocessing, paths, and scenario definitions.

## Common Run Entry

`script/quick_run.sh` is the main quick-start experiment script:

```bash
bash script/quick_run.sh [GPU_NUMS] [MODEL_NAME] [DATASET_NAME]
```

Defaults:

- `GPU_NUMS`: `1`
- `MODEL_NAME`: `model/DeepSeek-R1-Distill-Qwen-32B`
- `DATASET_NAME`: `release_v5`

The script runs:

1. Initial `codegeneration` inference and evaluation.
2. A `checkerextend` repair/evaluation pass based on the code generation output.
3. Additional iterative `checkerextend` repair passes.

It sets `HF_DATASETS_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so local datasets and local model files are expected.

## Working Notes for Agents

- Prefer `rg` for repository search.
- Keep generated outputs, model outputs, datasets, and build state out of source edits unless the task explicitly targets them.
- For benchmark changes, check `lcb_runner/runner/parser.py`, `lcb_runner/runner/main.py`, and the relevant scenario modules before changing shell scripts.
- For run-script changes, preserve offline-mode assumptions unless the user explicitly wants online HuggingFace/model access.
- Do not hardcode API keys, absolute paths, or machine-specific endpoints in scripts or source files.
- Put local secrets and environment-specific config in `key.json` (gitignored). Use `key.example.json` as a template.

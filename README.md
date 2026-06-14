# Code Generation and Repair Benchmark Runner

This repository provides a benchmark runner and repair pipeline for code generation tasks. It supports code generation, self-repair, checker/property-based extension, and evaluation across multiple models and datasets.

## Environment Setup

```bash
conda create -n codepbt python=3.11 -y
conda activate codepbt
pip install -r requirements.txt
```

## Data and Model Preparation

### Datasets

Download the benchmark data you want to use and place it under `datasets/` or `main_datasets/` at the project root. You can also override the root directory with the `DATASET_ROOT` environment variable.

Supported dataset identifiers (passed as `RELEASE_VERSION` to the run scripts):

- `release_v5` — LiveCodeBench code generation lite
- `humaneval` — HumanEval in the unified format
- `mbpp` — MBPP in the unified format
- `codecontests` — CodeContests in the unified format

Expected directory layout under the dataset root:

```text
datasets/
├── livecodebench___code_generation_lite/
├── humaneval_livecodebenchtype/
├── mbpp_livecodebenchtype/
└── codecontests_livecodebenchtype/
```

### Models

The runner can call any OpenAI-compatible API endpoint (local vLLM/TGI server, cloud provider, etc.). Model names and endpoints are configured in [`lcb_runner/lm_styles.py`](./lcb_runner/lm_styles.py). Add or edit a `LocalAPI` entry to point at your own endpoint, or use one of the existing provider styles.

## Configuration

Create a local `key.json` from the example file:

```bash
cp key.example.json key.json
# Edit key.json with your API key, base URL, model path, and dataset root
```

`key.json` is gitignored. It can contain:

- `api_key`: API key for OpenAI-compatible endpoints
- `api_base_url`: base URL for the model API
- `api_key_env`: environment variable name to use (default: `INF_API_KEY`)
- `model_path`: local path or identifier for the model
- `dataset_root`: local path to the datasets

Run scripts automatically source `key.json` via [`script/load_keys.sh`](./script/load_keys.sh). You can also set the corresponding environment variables directly instead of using `key.json`.

## Quick Start

Run the full default pipeline:

```bash
bash script/quick_run.sh [TP] [MODEL_NAME] [RELEASE_VERSION]
```

Defaults:

- `TP`: tensor-parallel size, default `1`
- `MODEL_NAME`: default is the value of `MODEL_PATH` from `key.json` / env, or `model/DeepSeek-R1-Distill-Qwen-32B`
- `RELEASE_VERSION`: default `release_v5`

`quick_run.sh` executes three stages:

1. Initial `codegeneration` inference and evaluation.
2. A `checkerextend` repair/evaluation pass based on the code generation output.
3. Additional iterative `checkerextend` repair passes.

## Other Run Scripts

- [`script/run_step1.sh`](./script/run_step1.sh) — code generation only
- [`script/run_direct_repair.sh`](./script/run_direct_repair.sh) — direct self-repair baseline
- [`script/run_checkerextend_trace.sh`](./script/run_checkerextend_trace.sh) — checkerextend with detailed property/repair traces
- [`script/load_keys.sh`](./script/load_keys.sh) — load `key.json` into environment variables

## Command-line Options

The main entrypoint is `python -m lcb_runner.runner.main`. For a full list of flags, see [`lcb_runner/runner/parser.py`](./lcb_runner/runner/parser.py) and the run scripts above.

## Running Local Models with vLLM (optional)

If you are running a model locally, start an OpenAI-compatible server, e.g. with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <model-path> \
    --port 8000 \
    --tensor-parallel-size <TP>
```

Then point the model entry in `lcb_runner/lm_styles.py` at `http://127.0.0.1:8000/v1`, or set `API_BASE_URL` in `key.json`.

### GPU Requirements

- At least 1 GPU is required for local inference.
- The exact GPU count and VRAM depend on the model size; set `CUDA_VISIBLE_DEVICES` and `--tensor_parallel_size` accordingly.

## Acknowledgement

- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench): the benchmark and base codebase this project builds upon.

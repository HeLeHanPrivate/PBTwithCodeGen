#!/bin/bash
set -euo pipefail

# Set INF_API_KEY in your environment if your model backend requires authentication.
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$(dirname "$0")/.."

# Optionally load secrets/environment config from key.json
if [[ -f script/load_keys.sh ]]; then
    source script/load_keys.sh
fi

MODEL="${MODEL_PATH:-model/DeepSeek-R1-Distill-Qwen-14B}"

python -m lcb_runner.runner.main \
  --model "$MODEL" \
  --scenario codegeneration \
  --n 1 \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_tokens 16384 \
  --stop '<｜end▁of▁sentence｜>' \
  --release_version release_v5 \
  --max_concurrency 24 \
  --evaluate \
  --num_process_evaluate 40 \
  --openai_timeout 600 \
  > logs/step1_codegen_v2.log 2>&1

echo "Step 1 finished with exit code $? at $(date)" >> logs/step1_codegen_v2.log

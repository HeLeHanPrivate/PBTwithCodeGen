#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Optionally load secrets/environment config from key.json
if [[ -f script/load_keys.sh ]]; then
    source script/load_keys.sh
fi

MODEL="${MODEL_PATH:-model/DeepSeek-R1-Distill-Qwen-14B}"
VERSION="release_v5"
TEMPERATURE="1.0"
N="1"
CPU_NUMS=40
CUSTOM_NAME="checkerextend_trace_32"

mkdir -p logs

echo "========================================"
echo "Checkerextend with detailed trace (no overwrite)"
echo "========================================"

# Set INF_API_KEY in your environment if your model backend requires authentication.
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
python -m lcb_runner.runner.main \
    --model ${MODEL} \
    --scenario checkerextend \
    --n ${N} \
    --codegen_n ${N} \
    --temperature ${TEMPERATURE} \
    --top_p 0.95 \
    --max_tokens 32768 \
    --stop '<｜end▁of▁sentence｜>' \
    --release_version ${VERSION} \
    --testcaseforrepair publiccase \
    --repairbase codegeneration \
    --repairbase_temperature ${TEMPERATURE} \
    --checker_mode property \
    --property_llm_merge \
    --property_llm_instrument \
    --property_public_only_routing \
    --property_oracle_skip_full_pass \
    --save_property_trace \
    --save_repair_trace \
    --custom_output_save_name ${CUSTOM_NAME} \
    --max_concurrency 32 \
    --openai_timeout 3600 \
    --evaluate \
    --num_process_evaluate ${CPU_NUMS} \
    > logs/checkerextend_trace_32.log 2>&1

saved_eval_all_file="output/${CUSTOM_NAME}/Scenario.checkerextend_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee logs/checkerextend_trace_32_scores.txt

echo "========================================"
echo "Checkerextend trace run completed!"
echo "========================================"

#!/bin/bash


TP=${1:-"1"}
MODEL_NAME=${2:-"model/DeepSeek-R1-Distill-Qwen-32B"}
VERSION=${3:-"release_v5"} # release_v5 humaneval mbpp


BASE_MODEL="DeepSeek-R1-Distill-Qwen-32B"
N=1
TEMPERATURE=0.6


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lcb_runner.runner.main \
    --model ${MODEL_NAME} \
    --n ${N} \
    --scenario codegeneration \
    --temperature ${TEMPERATURE} \
    --top_p 0.95 \
    --max_tokens 32768 \
    --stop '<｜end▁of▁sentence｜>' \
    --tensor_parallel_size ${TP} \
    --release_version ${VERSION} \
    --continue_existing \
    --num_process_evaluate 24 \
    --evaluate

saved_eval_all_file="output/${BASE_MODEL}/Scenario.codegeneration_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_codegeneration_${VERSION}.txt



HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lcb_runner.runner.main \
    --model ${MODEL_NAME} \
    --n 1 \
    --codegen_n ${N} \
    --scenario checkerextend \
    --temperature ${TEMPERATURE} \
    --top_p 0.95 \
    --max_tokens 16384 \
    --stop '<｜end▁of▁sentence｜>' \
    --tensor_parallel_size ${TP} \
    --release_version ${VERSION} \
    --testcaseforrepair "allcase" \
    --num_process_evaluate 24 \
    --repairbase codegeneration \
    --evaluate

saved_eval_all_file="output/${BASE_MODEL}/Scenario.checkerextend_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_checkerextend_${VERSION}.txt

#!/bin/bash


TP=${1:-"1"}
MODEL_NAME=${2:-"model/DeepSeek-R1-Distill-Qwen-32B"}
VERSION=${3:-"release_v5"} # release_v5 humaneval mbpp


CPU_NUMS=24
BASE_MODEL="DeepSeek-R1-Distill-Qwen-32B"
ITERATIVE_NUM=3
N=1
TEMPERATURE=0.6


# iter 1
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
    --num_process_evaluate $CPU_NUMS \
    --evaluate

saved_eval_all_file="output/${BASE_MODEL}/Scenario.codegeneration_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_codegeneration_${VERSION}.txt



# iter 2
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lcb_runner.runner.main \
    --model ${MODEL_NAME} \
    --n 1 \
    --codegen_n ${N} \
    --scenario selfrepair \
    --temperature ${TEMPERATURE} \
    --top_p 0.95 \
    --max_tokens 32768 \
    --stop '<｜end▁of▁sentence｜>' \
    --tensor_parallel_size ${TP} \
    --release_version ${VERSION} \
    --testcaseforrepair "publiccase" \
    --num_process_evaluate $CPU_NUMS \
    --repairbase codegeneration \
    --evaluate

saved_eval_all_file="output/${BASE_MODEL}/Scenario.selfrepair_TestCaseForRepair.publiccase_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_selfrepair_publiccase_${VERSION}.txt



# iter 3-N
for ((i=3; i<=$ITERATIVE_NUM; i++)); do
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m lcb_runner.runner.main \
      --model ${MODEL_NAME} \
      --n 1 \
      --codegen_n ${N} \
      --scenario selfrepair \
      --temperature ${TEMPERATURE} \
      --top_p 0.95 \
      --max_tokens 32768 \
      --stop '<｜end▁of▁sentence｜>' \
      --tensor_parallel_size ${TP} \
      --release_version ${VERSION} \
      --testcaseforrepair "publiccase" \
      --num_process_evaluate $CPU_NUMS \
      --repairbase selfrepair \
      --evaluate

    saved_eval_all_file="output/${BASE_MODEL}/Scenario.selfrepair_TestCaseForRepair.publiccase_${N}_${TEMPERATURE}_eval_all.json"
    echo "=============================="
    python -m lcb_runner.evaluation.compute_scores \
    --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_selfrepair_publiccase_${VERSION}.txt
done
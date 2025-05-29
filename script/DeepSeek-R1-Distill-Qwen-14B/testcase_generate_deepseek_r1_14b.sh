N=1
TEMPERATURE=0.6
MODEL_NAME="model/DeepSeek-R1-Distill-Qwen-14B"
BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"
VERSION="release_v5" # release_v5 humaneval
gpu_id="0,1"
TP=2


CUDA_VISIBLE_DEVICES=${gpu_id} \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lcb_runner.runner.main \
    --model ${MODEL_NAME} \
    --n 1 \
    --codegen_n ${N} \
    --scenario testcasegeneration \
    --temperature ${TEMPERATURE} \
    --top_p 0.95 \
    --max_tokens 32768 \
    --stop '<｜end▁of▁sentence｜>' \
    --tensor_parallel_size ${TP} \
    --release_version ${VERSION} \
    --continue_existing \
    --num_process_evaluate 30 \
    --evaluate


CUDA_VISIBLE_DEVICES=${gpu_id} \
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
    --num_process_evaluate 30 \
    --repairbase testcasegeneration \
    --evaluate

saved_eval_all_file="output/${BASE_MODEL}/Scenario.selfrepair_Scenario.testcasegeneration_TestCaseForRepair.publiccase_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_selfrepair_publiccase_${VERSION}.txt


# CUDA_VISIBLE_DEVICES=${gpu_id} \
# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# python -m lcb_runner.runner.main \
#     --model ${MODEL_NAME} \
#     --n 1 \
#     --codegen_n ${N} \
#     --scenario selfrepair \
#     --temperature ${TEMPERATURE} \
#     --top_p 0.95 \
#     --max_tokens 32768 \
#     --stop '<｜end▁of▁sentence｜>' \
#     --tensor_parallel_size ${TP} \
#     --release_version ${VERSION} \
#     --testcaseforrepair "allcase" \
#     --num_process_evaluate 30 \
#     --repairbase testcasegeneration \
#     --evaluate

# saved_eval_all_file="output/${BASE_MODEL}/Scenario.selfrepair_Scenario.testcasegeneration_TestCaseForRepair.allcase_${N}_${TEMPERATURE}_eval_all.json"
# echo "=============================="
# python -m lcb_runner.evaluation.compute_scores \
#   --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_selfrepair_allcase_${VERSION}.txt

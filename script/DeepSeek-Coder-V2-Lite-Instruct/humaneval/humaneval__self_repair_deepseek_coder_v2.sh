N=1
TEMPERATURE=0.6
MODEL_NAME="model/DeepSeek-Coder-V2-Lite-Instruct"
BASE_MODEL="DeepSeek-Coder-V2-Lite-Instruct"
VERSION="humaneval" # release_v5 humaneval


CUDA_VISIBLE_DEVICES=0,1 \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lcb_runner.runner.main \
    --model ${MODEL_NAME} \
    --n ${N} \
    --scenario codegeneration \
    --temperature ${TEMPERATURE} \
    --top_p 0.95 \
    --max_tokens 32768 \
    --stop '<｜end▁of▁sentence｜>' \
    --release_version ${VERSION} \
    --trust_remote_code \
    --continue_existing \
    --evaluate


saved_eval_all_file="output/${BASE_MODEL}----${VERSION}/Scenario.codegeneration_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}----${VERSION}/results_codegeneration_${VERSION}.txt



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
    --release_version ${VERSION} \
    --testcaseforrepair "allcase" \
    --evaluate

saved_eval_all_file="output/${BASE_MODEL}----${VERSION}/Scenario.selfrepair_TestCaseForRepair.allcase_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}----${VERSION}/results_selfrepair_allcase_${VERSION}.txt



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
    --release_version ${VERSION} \
    --testcaseforrepair "publiccase" \
    --evaluate

saved_eval_all_file="output/${BASE_MODEL}----${VERSION}/Scenario.selfrepair_TestCaseForRepair.publiccase_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}----${VERSION}/results_selfrepair_publiccase_${VERSION}.txt
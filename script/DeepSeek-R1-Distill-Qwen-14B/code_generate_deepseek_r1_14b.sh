N=5
TP=2
TEMPERATURE=0.6
MODEL_NAME="model/DeepSeek-R1-Distill-Qwen-14B"
BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"
VERSION="release_v5" # release_v5 humaneval


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
    --tensor_parallel_size ${TP} \
    --release_version ${VERSION} \
    --continue_existing \
    --num_process_evaluate 24 \
    --evaluate


saved_eval_all_file="output/${BASE_MODEL}/Scenario.codegeneration_${N}_${TEMPERATURE}_eval_all.json"
echo "=============================="
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} | tee output/${BASE_MODEL}/results_codegeneration_${VERSION}.txt


# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# python -m lcb_runner.runner.main \
#     --model model/DeepSeek-R1-Distill-Qwen-14B \
#     --scenario codegeneration \
#     --release_version release_v1 \
#     --trust_remote_code \
#     --n 1 \
#     --continue_existing_with_eval \
#     --evaluate
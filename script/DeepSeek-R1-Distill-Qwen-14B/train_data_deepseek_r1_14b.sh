N=1
TEMPERATURE=0.6
MODEL_NAME="model/DeepSeek-R1-Distill-Qwen-14B"
BASE_MODEL="DeepSeek-R1-Distill-Qwen-14B"
VERSION="codecontests" # release_v5 humaneval codecontests
gpu_id="0,1"
TP=2


# CUDA_VISIBLE_DEVICES=${gpu_id} \
# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# python -m lcb_runner.runner.main \
#     --model ${MODEL_NAME} \
#     --n ${N} \
#     --scenario codegeneration \
#     --temperature ${TEMPERATURE} \
#     --top_p 0.95 \
#     --max_tokens 32768 \
#     --stop '<｜end▁of▁sentence｜>' \
#     --tensor_parallel_size ${TP} \
#     --release_version ${VERSION} \
#     --continue_existing \
#     --num_process_evaluate 24 \
#     --evaluate


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
    --testcaseforrepair "allcase" \
    --num_process_evaluate 24 \
    --repairbase codegeneration \
    --evaluate


CUDA_VISIBLE_DEVICES=${gpu_id} \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lcb_runner.runner.main \
    --model ${MODEL_NAME} \
    --n 1 \
    --codegen_n ${N} \
    --scenario checkerextend \
    --temperature ${TEMPERATURE} \
    --top_p 0.95 \
    --max_tokens 32768 \
    --stop '<｜end▁of▁sentence｜>' \
    --tensor_parallel_size ${TP} \
    --release_version ${VERSION} \
    --testcaseforrepair "allcase" \
    --num_process_evaluate 24 \
    --repairbase codegeneration \
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
    --testcaseforrepair "allcase" \
    --num_process_evaluate 24 \
    --repairbase checkerextend \
    --evaluate

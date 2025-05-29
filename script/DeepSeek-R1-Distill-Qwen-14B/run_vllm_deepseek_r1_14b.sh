#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model model/DeepSeek-R1-Distill-Qwen-14B \
    --trust-remote-code \
    --dtype auto \
    --api-key token-abc123s \
    --port 18889 \
    --max_model_len 32768 \
    --tensor-parallel-size 1
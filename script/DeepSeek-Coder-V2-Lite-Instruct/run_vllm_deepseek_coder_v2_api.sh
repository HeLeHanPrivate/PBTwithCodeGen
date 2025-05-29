#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
    --model model/DeepSeek-Coder-V2-Lite-Instruct \
    --trust-remote-code \
    --dtype auto \
    --api-key token-abc123s \
    --port 18890 \
    --max_model_len 65536 \
    --tensor-parallel-size 1
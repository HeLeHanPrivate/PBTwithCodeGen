## All links are anonymous, some functions may be restricted.

## Prepare Environment


```
conda create -n codepbt python=3.11 -y
conda activate codepbt
pip install -r requirements.txt
```


## Download

1. Download code generation benchmark dataset (such as [Livecodebench](https://huggingface.co/datasets/livecodebench/code_generation)) from HuggingFace.

2. In order to unify the format of [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) and [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) with Livecodebench, it is necessary to manually change the format after downloading or directly download the [release](https://XXXX) we provided.

3. Download model (such as [Deepseek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)) from HuggingFace.



## Quick Start

1. For running the inference, change model name, path or API KEY in [./lcb_runner/lm_styles.py](./lcb_runner/lm_styles.py)

2. Use the following command to perform code generation:

```
bash script/quick_run.sh [GPU_NUMS, default=1] [MODEL_NAME in lm_styles.py, default="model/DeepSeek-R1-Distill-Qwen-32B"] [DATASET_NAME, default="realse_v5"(in LiveCodeBench)]
```


3. Please check the [./lcb_runner/runner/parser.py](./lcb_runner/runner/parser.py) file and the [./script/quick_run.sh](./script/quick_run.sh) file for more details on the flags.



## Acknowledgement

[LivecodeBench](https://github.com/LiveCodeBench/LiveCodeBench): The codebase we built upon.


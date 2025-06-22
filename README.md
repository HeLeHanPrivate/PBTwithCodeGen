## Now, all links are anonymous, some functions may be restricted.
## The full version will come soon

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

1. For running the inference, change model name, path or API KEY in [./lcb_runner/lm_styles.py](./lcb_runner/lm_styles.py) and change data path in [./lcb_runner/benchmarks/code_generation.py(line_142)](./lcb_runner/benchmarks/code_generation.py#L142)

2. Use the following command to perform code generation:

```
bash script/quick_run.sh [GPU_NUMS, default=1] [MODEL_NAME in lm_styles.py, default="model/DeepSeek-R1-Distill-Qwen-32B"] [DATASET_NAME, default="realse_v5"(in LiveCodeBench)]
```


3. Please check the [./lcb_runner/runner/parser.py](./lcb_runner/runner/parser.py) file and the [./script/quick_run.sh](./script/quick_run.sh) file for more details on the flags.



## Local Execution Requirements

**Note:** The following requirements apply if you are running the model locally and not through an API.

Local execution of this model relies on the **vLLM library**.

*   **GPU Requirement:**
*   A minimum of **1 GPU** is required to run the model.
*   For optimal performance, running on a **single NVIDIA A100 GPU** is recommended.
*   **Supported GPU Count:** The current configuration supports execution on **1 to 8 GPUs**.



## Acknowledgement

[LivecodeBench](https://github.com/LiveCodeBench/LiveCodeBench): The codebase we built upon.


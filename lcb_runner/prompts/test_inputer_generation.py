import json
import re # For HumanEval parsing
from enum import Enum # Import Enum
import time # To suggest time-based random seeds
import subprocess
import sys

from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import Test, Platform # Assuming Platform is defined here

# # --- Provided Enums (Ensure these are consistent with your actual lcb_runner definitions) ---
# class Platform(Enum):
#     LEETCODE = "leetcode"
#     CODEFORCES = "codeforces"
#     ATCODER = "atcoder"
#     HUMANEVAL = "humaneval"

# class LMStyle(Enum):
#     DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
#     CodeQwenInstruct = "CodeQwenInstruct"
#     DeepSeekR1 = "DeepSeekR1"
# # --- End of Provided Enums ---


class PromptConstants:
    SYSTEM_MESSAGE_DYNAMIC_INPUT_GENERATION_DEEPSEEK_R1 = (
        "<｜begin of sentence｜>A conversation between User and Assistant. "
        "The user provides a problem description and an example input string for a specific programming platform. "
        "The Assistant's task is to generate a Python script that dynamically creates diverse and valid input strings for the problem. "
        "This script, when executed multiple times, should ideally produce different valid input strings each time. "
        "It should use randomization, ideally seeded with the current time (e.g., milliseconds) to ensure variability. "
        "The script's standard output (e.g., from print statements) must be a single string that perfectly matches the input format required by the described programming platform. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the Python script. "
        "The reasoning process and script are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )

    # System message for DeepSeek-Coder (v1/v2 instruct series) for DYNAMIC INPUT GENERATION
    SYSTEM_MESSAGE_DEEPSEEK_CODER_INPUT_GENERATOR = (
        "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
        "developed by DeepSeek Company. Your task is to help a user by generating a Python script "
        "that dynamically creates diverse and valid input strings for a given programming problem description and platform. "
        "The generated script should use randomization (seeded by current time) and its standard output "
        "must be a single string matching the required input format of the platform."
    )

    # System message for CodeQwen Instruct for DYNAMIC INPUT GENERATION
    SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_SYSTEM_TAG = "<|im_start|>system"
    SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_CONTENT = (
        "You are a helpful AI programming assistant. Your primary goal is to generate a Python script "
        "that dynamically creates diverse and valid input strings for a programming problem based on its description and platform. "
        "The script should use randomization (seeded by current time) and print a single string to standard output, "
        "formatted exactly as required by the problem's input specification."
    )
    SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_END_TAG = "<|im_end|>"
    SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_USER_TAG = "\n<|im_start|>user"
    # The actual user prompt will follow this


def _get_common_dynamic_input_generation_user_content(question: str, example_input_str: str, platform: Platform, original_code_snippet: str, model_specific_output_format_instruction: str) -> str:
    """
    Helper function to create the common part of the user prompt for dynamic input generation.
    """
    user_content = "You are an expert Python programmer specializing in generating diverse and valid problem inputs for competitive programming platforms using randomization. "
    user_content += "Given a problem description, an example of how a specific input is formatted for a platform, and optionally the original code that processes this input, "
    user_content += "your task is to write a Python script. This script should:\n"
    user_content += "1. Utilize randomization to generate diverse input values that are valid according to the problem description (e.g., within specified ranges, adhering to constraints).\n"
    user_content += "2. Seed the random number generator using the current time (e.g., `random.seed(int(time.time() * 1000))`) so that each execution of the script can produce a different input.\n"
    user_content += "3. Print a single string to standard output. This printed string must be formatted *exactly* as a valid input for the described problem on the specified platform.\n"
    user_content += "4. Ensure the generated inputs are non-trivial and explore different aspects or edge cases of the problem if feasible within reasonable generation logic.\n\n"

    user_content += f"### Problem Description:\n{question}\n\n"
    if original_code_snippet and original_code_snippet.strip(): # Only include if there's actual code
        user_content += f"### Original Code Snippet (for context on input parsing, if helpful):\n```python\n{original_code_snippet}\n```\n\n"
    else:
        user_content += "### Original Code Snippet: Not provided or empty.\n\n"


    input_generation_logic_guidance = (
        "Consider the constraints on input values (e.g., N between 1 and 100, array elements between -1000 and 1000). "
        "Your script should randomly generate values within these constraints and then format them into the required string output. "
        "For example, if N is an integer length, generate a random N, then generate N random elements if it's an array problem.\n"
    )

    # Platform specific guidance
    if platform == Platform.HUMANEVAL:
        user_content += f"### Platform: HumanEval\n"
        user_content += f"### Example of an assertion used for testing (You don't need to make a judgment, just call and generate string as 'assert ...(...) is not None'):\n`{example_input_str}`\n\n"
        user_content += "Your Python script should print a string representing a valid assertion for this problem's entry point function, using randomly generated plausible arguments. "
        user_content += "Seed your randomization. For example, if the function takes two integers, generate two random integers and construct the assertion string.\n"
        user_content += input_generation_logic_guidance
        user_content += "Desired output of your script (a single string, varying with each run):\nAn assertion string like 'assert function_name(random_arg1, random_arg2) is not None'\n"
        user_content += "Example script structure:\n```python\nimport random\nimport time\n\nrandom.seed(int(time.time() * 1000))\n\n# Generate random arguments based on problem description\narg1 = random.randint(1, 100)\narg2 = [random.randint(-10, 10) for _ in range(random.randint(1,5))]\n# Format into the assertion string\n# IMPORTANT: The function name in the assertion should be the actual function name from original code\n# Please refer to the problem description for the correct function name to use in the assertion.\n# Assuming 'entry_point' for this generic example:\nassertion_string = f\"assert entry_point({arg1}, {arg2}) is not None\" # Adapt function name as needed\nprint(assertion_string)\n```\n"
    elif platform == Platform.LEETCODE:
        user_content += f"### Platform: LeetCode\n"
        user_content += f"### Example Input String (lines are json loads, joined by '\\n'):\n```\n{example_input_str}\n```\n\n"
        user_content += "Your Python script should generate random valid data based on the problem description, then format this data into a string that, when interpreted (e.g., split by '\\n' and each line json.loads'd), "
        user_content += "provides the arguments to the LeetCode solution function. The printed string itself should exactly match the platform's multi-line JSON string format if applicable. Seed your randomization.\n"
        user_content += input_generation_logic_guidance
        user_content += "Desired output of your script (a single string, potentially multi-line, varying with each run):\nA string formatted for LeetCode input, with randomized content.\n"
        user_content += "Example script structure for an input like '[1,2,3]\\n\"some_string\"':\n```python\nimport random\nimport time\nimport json\n\nrandom.seed(int(time.time() * 1000))\n\n# Generate random data for each part of the LeetCode input\nrandom_list = [random.randint(1, 100) for _ in range(random.randint(1, 5))]\nrandom_string = \"generated_\" + str(random.randint(100, 999))\n\n# Format as JSON strings and join with newline\n# Ensure proper escaping if strings contain special characters for JSON\noutput_parts = [json.dumps(random_list), json.dumps(random_string)]\nprint('\\n'.join(output_parts))\n```\n"
    elif platform in [Platform.CODEFORCES, Platform.ATCODER]:
        user_content += f"### Platform: {platform.value}\n"
        user_content += f"### Example Input String (as read from standard input):\n```\n{example_input_str}\n```\n\n"
        user_content += "Your Python script should generate random valid data according to the problem description, then format this data into a string that exactly matches the standard input format for this problem (e.g., numbers separated by spaces or newlines). Seed your randomization.\n"
        user_content += input_generation_logic_guidance
        user_content += "Desired output of your script (a single string, potentially multi-line, varying with each run):\nA string formatted for standard input, with randomized content.\n"
        user_content += "Example script structure for an input like '3\\n1 2 3':\n```python\nimport random\nimport time\n\nrandom.seed(int(time.time() * 1000))\n\n# Generate N\nN = random.randint(1, 5) # Example constraint\n# Generate N numbers\nnumbers = [str(random.randint(1, 100)) for _ in range(N)]\n\n# Format for standard input\noutput_string = str(N) + '\\n' + ' '.join(numbers)\nprint(output_string)\n```\n"
    else: # Fallback
        user_content += f"### Platform: Generic/Unknown ({platform.value if hasattr(platform, 'value') else platform})\n"
        user_content += f"### Example Input String:\n```\n{example_input_str}\n```\n\n"
        user_content += "Your Python script should generate random valid data and print a string that represents a valid input for the problem, closely following the format of the example input string. Use randomization seeded by current time.\n"
        user_content += input_generation_logic_guidance
        user_content += "Desired output of your script (a single string, potentially multi-line, varying with each run):\nA string formatted like the example input, with randomized content.\n"

    user_content += "\n### Your Task:\nGenerate the Python script that uses randomization (seeded by current time) to produce diverse and valid input strings according to the problem description and platform format.\n"
    user_content += "Ensure your script imports `random` and `time` (and `json` if needed for LeetCode style) and sets `random.seed(int(time.time() * 1000))` at the beginning.\n"
    user_content += "The final output of your script should be a single print statement outputting the formatted input string.\n"
    user_content += model_specific_output_format_instruction # Add model-specific formatting here
    return user_content


def get_deepseek_r1_dynamic_input_generation_prompt(question: str, example_input_str: str, platform: Platform, original_code_snippet: str) -> str:
    output_format_instruction = (
        "Remember to place your reasoning within `<think></think>` tags and the final Python script "
        "within `<answer>\n```python\n# YOUR PYTHON SCRIPT HERE\n```\n</answer>` tags."
    )
    user_content = _get_common_dynamic_input_generation_user_content(question, example_input_str, platform, original_code_snippet, output_format_instruction)
    # DeepSeekR1's system message expects <|User|> to be the start of the user's turn,
    # but the user_content here is the actual content *after* <|User|>.
    # The SYSTEM_MESSAGE already ends with <|User|>. The assistant tag is for the model's response start.
    return user_content + f"<｜Assistant｜>"


def get_deepseek_coder_dynamic_input_generation_prompt(question: str, example_input_str: str, platform: Platform, original_code_snippet: str) -> str:
    # DeepSeek Coder Instruct usually expects a clear instruction block then a response block.
    output_format_instruction = (
        "Please provide ONLY the entire Python script for dynamic input generation, "
        "enclosed in a single ```python ... ``` block. Do not add any explanations before or after the code block."
    )
    user_content = _get_common_dynamic_input_generation_user_content(question, example_input_str, platform, original_code_snippet, output_format_instruction)
    # Format for DeepSeek Coder: ### Instruction: ... ### Response:
    prompt = f"### Instruction:\n{user_content}\n\n"
    prompt += f"### Response:\n" # Model should start its code block after this.
    return prompt


def get_codeqwen_instruct_dynamic_input_generation_prompt(question: str, example_input_str: str, platform: Platform, original_code_snippet: str) -> str:
    # CodeQwen Instruct uses <|im_start|>user ... <|im_end|> then <|im_start|>assistant
    output_format_instruction = (
        "Please provide ONLY the entire Python script for dynamic input generation, "
        "enclosed in a single ```python ... ``` block. Do not add any explanations before or after the code block."
    )
    user_content = _get_common_dynamic_input_generation_user_content(question, example_input_str, platform, original_code_snippet, output_format_instruction)

    # Construct the prompt with CodeQwen's specific tags
    # The system message is added separately in format_prompt_inputer_generate
    prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_USER_TAG}\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def execute_inputer_script(script_code: str, num_executions: int = 5, timeout_seconds: int = 1) -> tuple[list[str], bool]:
    generated_inputs = []
    all_executions_successful = True

    # Ensure necessary imports are present, trying not to add them if already there.
    # A more robust way would be to parse the script, but this is simpler.
    if "import random" not in script_code:
        script_code = "import random\n" + script_code
    if "import time" not in script_code:
        script_code = "import time\n" + script_code
    if "import json" not in script_code and "json.dumps(" in script_code: # Heuristic for LeetCode
         script_code = "import json\n" + script_code

    for i in range(num_executions):
        try:
            process = subprocess.run(
                [sys.executable, "-c", script_code],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False
            )
            if process.returncode == 0:
                output = process.stdout.strip()
                if output:
                    # Storing as a dictionary consistent with how 'samples' might be structured elsewhere
                    generated_inputs.append({"input": output, "output": ""}) # Assuming 'output' is for expected output, blank here
                else:
                    print(f"Warning: Inputer script execution {i+1} produced no output.")
            else:
                print(f"Error: Inputer script execution {i+1} failed with return code {process.returncode}.")
                print(f"Stderr: {process.stderr.strip()}")
                all_executions_successful = False
                break
        except subprocess.TimeoutExpired:
            print(f"Error: Inputer script execution {i+1} timed out after {timeout_seconds} seconds.")
            all_executions_successful = False
            break
        except Exception as e:
            print(f"Error: An unexpected error occurred during inputer script execution {i+1}: {e}")
            all_executions_successful = False
            break
        if num_executions > 1: # Only sleep if multiple executions to allow for different seeds
            time.sleep(0.01)
    return generated_inputs, all_executions_successful


def format_prompt_inputer_generate(
    question: str, LanguageModelStyle: LMStyle, code: str, results, platform_and_samples # results was unused, removed
) -> str:
    platform, samples_dict = platform_and_samples # Unpack the tuple
    # Extract example input string from samples_dict
    # This part needs to be robust to the structure of samples_dict["input_output"]["inputs"]
    example_input_str = "N/A" # Default
    try:
        # Assuming 'inputs' is a list and we take the first one as an example
        # Or it might be a single string directly for some platforms
        raw_inputs_data = samples_dict.get("input_output", {}).get("inputs")
        if isinstance(raw_inputs_data, list) and raw_inputs_data:
            example_input_str = str(raw_inputs_data[0])
        elif isinstance(raw_inputs_data, str):
            example_input_str = raw_inputs_data
    except Exception as e:
        # print(f"Warning: Could not extract example_input_str from samples: {e}")
        example_input_str = "N/A"


    if LanguageModelStyle == LMStyle.DeepSeekR1:
        user_content = get_deepseek_r1_dynamic_input_generation_prompt(question, example_input_str, platform, code)
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DYNAMIC_INPUT_GENERATION_DEEPSEEK_R1}{user_content}"
        return prompt
    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct: # For DeepSeek-Coder-V2 etc.
        system_message = PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_CODER_INPUT_GENERATOR
        user_content = get_deepseek_coder_dynamic_input_generation_prompt(question, example_input_str, platform, code)
        prompt = f"{system_message}\n\n{user_content}" # user_content already contains ### Response:
        return prompt
    elif LanguageModelStyle == LMStyle.CodeQwenInstruct: # For Qwen2.5-Coder etc.
        system_part = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_SYSTEM_TAG}\n{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_CONTENT}{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_INPUT_GENERATOR_END_TAG}"
        user_and_assistant_part = get_codeqwen_instruct_dynamic_input_generation_prompt(question, example_input_str, platform, code)
        prompt = f"{system_part}{user_and_assistant_part}" # user_and_assistant_part already starts with user tag and ends with assistant tag
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle.value if hasattr(LanguageModelStyle, 'value') else LanguageModelStyle} not implemented for dynamic input generation."
        )


# --- Test Function (Updated to use actual Enums and test dynamic generation) ---
def test():
    print("--- Testing format_prompt_inputer_generate ---")

    # Common elements for tests
    test_question_he = "Write a function `add(a: int, b: int)` that returns the sum of a and b. `a` and `b` are between -100 and 100."
    test_code_he = "def add(a, b):\n  return a + b"
    # HumanEval example input is part of an assertion string
    test_samples_he = {"input_output": {"inputs": ["assert add(1, 2) == 3"]}} # Example: the prompt expects to see how args are formatted


    test_question_lc = "Given a list of up to 10 integers (each between 1 and 100), find the maximum."
    test_code_lc = "class Solution:\n    def findMax(self, nums: list[int]) -> int:\n        return max(nums)"
    test_samples_lc = {"input_output": {"inputs": ['[1,2,3,4,5]']}} # LeetCode inputs are often JSON strings


    test_question_cf = "Read N (1 <= N <= 5), then N integers (1 to 100), print their sum."
    test_code_cf = "n = int(input())\nnums = list(map(int, input().split()))\nprint(sum(nums))"
    test_samples_cf = {"input_output": {"inputs": ["3\n1 2 3"]}} # Codeforces takes stdin string


    # Test DeepSeekR1
    print("\n--- Test Case 1: DeepSeekR1 - HumanEval ---")
    prompt_dsr1_he = format_prompt_inputer_generate(test_question_he, LMStyle.DeepSeekR1, test_code_he, False, (Platform.HUMANEVAL, test_samples_he))
    print(prompt_dsr1_he)

    # Test DeepSeekCodeInstruct
    print("\n--- Test Case 2: DeepSeekCodeInstruct - LeetCode ---")
    prompt_dsci_lc = format_prompt_inputer_generate(test_question_lc, LMStyle.DeepSeekCodeInstruct, test_code_lc, False, (Platform.LEETCODE, test_samples_lc))
    print(prompt_dsci_lc)

    # Test CodeQwenInstruct
    print("\n--- Test Case 3: CodeQwenInstruct - Codeforces ---")
    prompt_cq_cf = format_prompt_inputer_generate(test_question_cf, LMStyle.CodeQwenInstruct, test_code_cf, False, (Platform.CODEFORCES, test_samples_cf))
    print(prompt_cq_cf)


    # Test execution of a sample generated script
    print("\n--- Testing execute_inputer_script ---")
    # This is a hypothetical script that an LLM *might* generate for the HumanEval `add` problem
    sample_generated_script_he = """
import random
import time

random.seed(int(time.time() * 1000))

a = random.randint(-100, 100)
b = random.randint(-100, 100)
# For HumanEval, the prompt asked for argument string
args_string = f"assert solve({a}, {b}) is not None"
print(args_string)
"""
    generated_inputs_he, success_he = execute_inputer_script(sample_generated_script_he, num_executions=3)
    print(f"HumanEval Script Execution Success: {success_he}, Generated Inputs: {generated_inputs_he}")

    # Hypothetical script for LeetCode findMax
    sample_generated_script_lc = """
import random
import time
import json

random.seed(int(time.time() * 1000))

num_elements = random.randint(1, 10)
elements = [random.randint(1, 100) for _ in range(num_elements)]
print(json.dumps(elements))
"""
    generated_inputs_lc, success_lc = execute_inputer_script(sample_generated_script_lc, num_executions=3)
    print(f"LeetCode Script Execution Success: {success_lc}, Generated Inputs: {generated_inputs_lc}")


    print("\n--- End of Tests ---")


if __name__ == "__main__":
    test()
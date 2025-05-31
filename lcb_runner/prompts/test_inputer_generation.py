import json
import re # For HumanEval parsing
from enum import Enum # Import Enum
import time # To suggest time-based random seeds
import subprocess
import sys

from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import Test, Platform

# # --- Provided Enums ---
# class Platform(Enum):
#     LEETCODE = "leetcode"
#     CODEFORCES = "codeforces"
#     ATCODER = "atcoder"
#     HUMANEVAL = "humaneval"

# class LMStyle(Enum):
#     OpenAIChat = "OpenAIChat"
#     OpenAIReasonPreview = "OpenAIReasonPreview"
#     OpenAIReason = "OpenAIReason"
#     Claude = "Claude"
#     Claude3 = "Claude3"
#     Gemini = "Gemini"
#     GeminiThinking = "GeminiThinking"
#     MistralWeb = "MistralWeb"
#     CohereCommand = "CohereCommand"
#     DataBricks = "DataBricks"
#     DeepSeekAPI = "DeepSeekAPI"
#     LocalAPI = "LocalAPI"
#     GenericBase = "GenericBase"
#     DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
#     CodeLLaMaInstruct = "CodeLLaMaInstruct"
#     StarCoderInstruct = "StarCoderInstruct"
#     CodeQwenInstruct = "CodeQwenInstruct"
#     QwQ = "QwQ"
#     LLaMa3 = "LLaMa3"
#     DeepSeekR1 = "DeepSeekR1"
# # --- End of Provided Enums ---


class PromptConstants:
    SYSTEM_MESSAGE_DYNAMIC_INPUT_GENERATION_DEEPSEEK_R1 = ( # Renamed for clarity
        "<｜begin of sentence｜>A conversation between User and Assistant. "
        "The user provides a problem description and an example input string for a specific programming platform. "
        "The Assistant's task is to generate a Python script that dynamically creates diverse and valid input strings for the problem. "
        "This script, when executed multiple times, should ideally produce different valid input strings each time. "
        "It should use randomization, ideally seeded with the current time (e.g., milliseconds) to ensure variability. "
        "The script's standard output (e.g., from print statements) must be a single string that perfectly matches the input format required by the described programming platform. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the Python script. "
        "The reasoning process and script are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )

def get_deepseek_r1_dynamic_input_generation_prompt(question: str, example_input_str: str, platform: Platform, original_code_snippet: str) -> str:
    """
    Generates a prompt for DeepSeek-R1 to create a Python script that dynamically
    and randomly outputs diverse problem input strings.
    """
    prompt = "You are an expert Python programmer specializing in generating diverse and valid problem inputs for competitive programming platforms using randomization. "
    prompt += "Given a problem description, an example of how a specific input is formatted for a platform, and optionally the original code that processes this input, "
    prompt += "your task is to write a Python script. This script should:\n"
    prompt += "1. Utilize randomization to generate diverse input values that are valid according to the problem description (e.g., within specified ranges, adhering to constraints).\n"
    prompt += "2. Ideally, seed the random number generator using the current time (e.g., `random.seed(int(time.time() * 1000))`) so that each execution of the script can produce a different input.\n"
    prompt += "3. Print a single string to standard output. This printed string must be formatted *exactly* as a valid input for the described problem on the specified platform.\n"
    prompt += "4. Ensure the generated inputs are non-trivial and explore different aspects or edge cases of the problem if feasible within reasonable generation logic.\n\n"


    prompt += f"### Problem Description:\n{question}\n\n"
    prompt += f"### Original Code Snippet (for context on input parsing, if helpful):\n```python\n{original_code_snippet}\n```\n\n"

    input_generation_logic_guidance = (
        "Consider the constraints on input values (e.g., N between 1 and 100, array elements between -1000 and 1000). "
        "Your script should randomly generate values within these constraints and then format them into the required string output. "
        "For example, if N is an integer length, generate a random N, then generate N random elements if it's an array problem.\n"
    )

    if platform == Platform.HUMANEVAL:
        prompt += f"### Platform: HumanEval\n"
        prompt += f"### Example of an assertion used for testing (You don't need to make a judgment, just call and generate the second half as 'assert ...(...) is not None'):\n`{example_input_str}`\n\n"
        prompt += "Your Python script should print a string representing a valid assertion for this problem's entry point function, using randomly generated plausible arguments. "
        prompt += "Seed your randomization. For example, if the function takes two integers, generate two random integers and construct the assertion string.\n"
        prompt += input_generation_logic_guidance
        prompt += "Desired output of your script (a single string, varying with each run):\nAn assertion string like 'assert function_name(random_arg1, random_arg2) is not None'\n"
        prompt += "Example script structure:\n```python\nimport random\nimport time\n\nrandom.seed(int(time.time() * 1000))\n\n# Generate random arguments based on problem description\narg1 = random.randint(1, 100)\narg2 = [random.randint(-10, 10) for _ in range(random.randint(1,5))]\n# Format into the assertion string\n# IMPORTANT: The function name in the assertion should be the actual function name from original code\n# Please refer to the problem description for the correct function name to use in the assertion.\n# Assuming 'entry_point' for this generic example:\nassertion_string = f\"assert entry_point({arg1}, {arg2}) is not None\" # Adapt function name as needed\nprint(assertion_string)\n```\n"


    elif platform == Platform.LEETCODE:
        prompt += f"### Platform: LeetCode\n"
        prompt += f"### Example Input String (lines are json loads, joined by '\\n'):\n```\n{example_input_str}\n```\n\n"
        prompt += "Your Python script should generate random valid data based on the problem description, then format this data into a string that, when interpreted (e.g., split by '\\n' and each line json.loads'd), "
        prompt += "provides the arguments to the LeetCode solution function. The printed string itself should exactly match the platform's multi-line JSON string format if applicable. Seed your randomization.\n"
        prompt += input_generation_logic_guidance
        prompt += "Desired output of your script (a single string, potentially multi-line, varying with each run):\nA string formatted for LeetCode input, with randomized content.\n"
        prompt += "Example script structure for an input like '[1,2,3]\\n\"some_string\"':\n```python\nimport random\nimport time\nimport json\n\nrandom.seed(int(time.time() * 1000))\n\n# Generate random data for each part of the LeetCode input\nrandom_list = [random.randint(1, 100) for _ in range(random.randint(1, 5))]\nrandom_string = \"generated_\" + str(random.randint(100, 999))\n\n# Format as JSON strings and join with newline\n# Ensure proper escaping if strings contain special characters for JSON\noutput_parts = [json.dumps(random_list), json.dumps(random_string)]\nprint('\\n'.join(output_parts))\n```\n"


    elif platform in [Platform.CODEFORCES, Platform.ATCODER]:
        prompt += f"### Platform: {platform.value}\n"
        prompt += f"### Example Input String (as read from standard input):\n```\n{example_input_str}\n```\n\n"
        prompt += "Your Python script should generate random valid data according to the problem description, then format this data into a string that exactly matches the standard input format for this problem (e.g., numbers separated by spaces or newlines). Seed your randomization.\n"
        prompt += input_generation_logic_guidance
        prompt += "Desired output of your script (a single string, potentially multi-line, varying with each run):\nA string formatted for standard input, with randomized content.\n"
        prompt += "Example script structure for an input like '3\\n1 2 3':\n```python\nimport random\nimport time\n\nrandom.seed(int(time.time() * 1000))\n\n# Generate N\nN = random.randint(1, 5) # Example constraint\n# Generate N numbers\nnumbers = [str(random.randint(1, 100)) for _ in range(N)]\n\n# Format for standard input\noutput_string = str(N) + '\\n' + ' '.join(numbers)\nprint(output_string)\n```\n"

    else: # Fallback for generic/unknown
        prompt += f"### Platform: Generic/Unknown ({platform.value if hasattr(platform, 'value') else platform})\n"
        prompt += f"### Example Input String:\n```\n{example_input_str}\n```\n\n"
        prompt += "Your Python script should generate random valid data and print a string that represents a valid input for the problem, closely following the format of the example input string. Use randomization seeded by current time.\n"
        prompt += input_generation_logic_guidance
        prompt += "Desired output of your script (a single string, potentially multi-line, varying with each run):\nA string formatted like the example input, with randomized content.\n"


    prompt += "\n### Your Task:\nGenerate the Python script that uses randomization (seeded by current time) to produce diverse and valid input strings according to the problem description and platform format.\n"
    prompt += "Ensure your script imports `random` and `time` and sets `random.seed(int(time.time() * 1000))` at the beginning.\n"
    prompt += "The final output of your script should be a single print statement outputting the formatted input string.\n"
    prompt += "<answer>\n" # Adjusted to match SYSTEM_MESSAGE format
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    # prompt += "```python\nimport random\nimport time\n\n# Seed for variability, ideally at the very start of the script\nrandom.seed(int(time.time() * 1000))\n\n# YOUR PYTHON SCRIPT LOGIC TO GENERATE AND PRINT THE INPUT STRING GOES HERE\n# Remember to print only the final formatted input string.\n\n# Example (replace with actual logic):\n# if platform == Platform.CODEFORCES: # pseudo-code for illustration\n#   n = random.randint(1, 10)\n#   print(n)\n#   elements = [str(random.randint(1, 100)) for _ in range(n)]\n#   print(\" \".join(elements))\n\n```\n"
    prompt += "</answer>\n" # Adjusted
    prompt += f"<｜Assistant｜>" # This is specific to DeepSeek R1 style, may need adjustment for other models
    return prompt



def execute_inputer_script(script_code: str, num_executions: int = 5, timeout_seconds: int = 1) -> tuple[list[str], bool]:
    """
    Executes the LLM-generated inputer script multiple times and collects its outputs.

    Args:
        script_code: The Python code string for the inputer script.
        num_executions: The number of times to run the script.
        timeout_seconds: Timeout for each execution of the script.

    Returns:
        A tuple: (list_of_generated_inputs, success_flag)
        The list contains successfully generated input strings.
        The success_flag is False if any execution resulted in a runtime error or timeout,
        indicating the inputer script itself might be flawed. Otherwise, True.
    """
    generated_inputs = []
    all_executions_successful = True

    if "import random" not in script_code:
        script_code = "import random\n" + script_code
    if "import time" not in script_code:
        script_code = "import time\n" + script_code
    if "import json" not in script_code and "json.dumps" in script_code : # If LeetCode style might need it
         script_code = "import json\n" + script_code

    for i in range(num_executions):
        try:
            # Execute the script in a separate process for isolation and timeout
            # Using sys.executable ensures we use the same Python interpreter
            process = subprocess.run(
                [sys.executable, "-c", script_code],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False # Do not raise CalledProcessError on non-zero exit codes
            )

            if process.returncode == 0:
                output = process.stdout.strip()
                if output: # Ensure there is some output
                    generated_inputs.append({"input": output, "output": ""})
                else:
                    print(f"Warning: Inputer script execution {i+1} produced no output.")
                    # Optionally, treat no output as an error:
                    # all_executions_successful = False
                    # break
            else:
                print(f"Error: Inputer script execution {i+1} failed with return code {process.returncode}.")
                print(f"Stderr: {process.stderr.strip()}")
                all_executions_successful = False
                break # Stop on first error

        except subprocess.TimeoutExpired:
            print(f"Error: Inputer script execution {i+1} timed out after {timeout_seconds} seconds.")
            all_executions_successful = False
            break
        except Exception as e:
            print(f"Error: An unexpected error occurred during inputer script execution {i+1}: {e}")
            all_executions_successful = False
            break
        time.sleep(0.01) # Small delay to ensure different millisecond seeds if runs are very fast

    return generated_inputs, all_executions_successful



def format_prompt_inputer_generate(
    question: str, LanguageModelStyle: LMStyle, code: str, results, platform_and_samples
) -> str:
    platform = platform_and_samples[0]
    samples = platform_and_samples[1]
    input_sample_str = str(samples["input_output"]["inputs"][0])

    if LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt_content = get_deepseek_r1_dynamic_input_generation_prompt(question, input_sample_str, platform, code)
        # The SYSTEM_MESSAGE already includes the <|User|> tag for the start of user input
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DYNAMIC_INPUT_GENERATION_DEEPSEEK_R1}{prompt_content}" # Removed the extra \n\n
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle.value if hasattr(LanguageModelStyle, 'value') else LanguageModelStyle} not implemented for dynamic input generation."
        )


# --- Test Function (Updated to use actual Enums and test dynamic generation) ---
def test():
    # --- Test Case 1: HumanEval ---
    question_he = "Write a function `add(a: int, b: int)` that returns the sum of a and b. `a` and `b` are between -100 and 100."
    code_he = "def add(a, b):\n  return a + b"
    samples_he = {"input_output": {"inputs": ["assert add(1, 2) == 3"]}}
    platform_he = Platform.HUMANEVAL
    print("--- HumanEval Dynamic Input Prompt (DeepSeekR1) ---")
    prompt_he = format_prompt_inputer_generate(question_he, LMStyle.DeepSeekR1, code_he, platform_he, samples_he)
    print(prompt_he)
    print("---------------------------------------------------\n")

    # --- Test Case 2: LeetCode ---
    question_lc = "Given a list of up to 10 integers (each between 1 and 100), find the maximum."
    code_lc = "class Solution:\n    def findMax(self, nums: list[int]) -> int:\n        return max(nums)"
    # Example of LeetCode input: first line is one arg, second line another
    samples_lc = {"input_output": {"inputs": ['[1,2,3,4,5]']}} # Simpler LeetCode example for this test
    platform_lc = Platform.LEETCODE
    print("--- LeetCode Dynamic Input Prompt (DeepSeekR1) ---")
    prompt_lc = format_prompt_inputer_generate(question_lc, LMStyle.DeepSeekR1, code_lc, platform_lc, samples_lc)
    print(prompt_lc)
    print("--------------------------------------------------\n")

    # --- Test Case 3: Codeforces ---
    question_cf = "Read N (1 <= N <= 5), then N integers (1 to 100), print their sum."
    code_cf = "n = int(input())\nnums = list(map(int, input().split()))\nprint(sum(nums))"
    samples_cf = {"input_output": {"inputs": ["3\n1 2 3"]}}
    platform_cf = Platform.CODEFORCES
    print("--- Codeforces Dynamic Input Prompt (DeepSeekR1) ---")
    prompt_cf = format_prompt_inputer_generate(question_cf, LMStyle.DeepSeekR1, code_cf, platform_cf, samples_cf)
    print(prompt_cf)
    print("----------------------------------------------------\n")

if __name__ == "__main__":
    test()
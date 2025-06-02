import json
from enum import Enum # Import Enum
from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle

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


class PromptConstants:
    
    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin of sentence｜>A conversation between User and Assistant. "
        "The user provides a problem description and a Python program. The Assistant's primary goal is to **instrument the provided program with property-based verification checks**. "
        "These checks should be inserted **before the program would normally return its result**. "
        "The enhanced program must adhere to the following behavior: "
        "1. If all embedded property verifications pass, the program should proceed to output its original computed answer. "
        "2. If any property verification fails, the program must explicitly throw an `AssertError` (or a similar clear, unrecoverable error indicating a property violation). "
        "This mechanism allows the user to verify the program's correctness against specified properties for any given test case, without needing an external expected answer. "
        "**Important: The Assistant should NOT attempt to fix any bugs in the original program or alter its core problem-solving logic.** The focus is solely on adding robust property checks based on the problem description. "
        "Properties should reflect invariants, post-conditions, or relationships described or implied by the problem statement. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Identify potential properties. Determine where to insert checks. Formulate assertion logic. </think> <answer> ```python\n# Enhanced code here\n``` </answer>.<｜User｜>"
    )
    
    FORMATTING_WITHOUT_STARTER_CODE = "Enclose your code within delimiters as follows."
    
    # System message for DeepSeek-Coder (v1/v2 instruct series) for property checking
    # We will combine this with the detailed task instruction in the user prompt
    SYSTEM_MESSAGE_DEEPSEEK_CODER_CHECKER = (
        "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
        "developed by DeepSeek Company. Your task is to help a user by instrumenting "
        "their Python program with property-based verification checks based on a "
        "problem description. You should NOT fix bugs or change the core logic. "
        "Your goal is to make the program self-verifying: output the original result "
        "if properties hold, or throw an AssertError if a property is violated."
    )

    # System message for CodeQwen Instruct for property checking
    # CodeQwen uses a specific chat format
    SYSTEM_MESSAGE_CODEQWEN_CHECKER_SYSTEM_TAG = "<|im_start|>system"
    SYSTEM_MESSAGE_CODEQWEN_CHECKER_CONTENT = (
        "You are a helpful AI programming assistant. Your primary goal is to instrument the "
        "provided Python program with property-based verification checks based on the problem description. "
        "These checks should be inserted before the program would normally return its result. "
        "The enhanced program must: 1. If all properties pass, output its original computed answer. "
        "2. If any property fails, throw an `AssertError`. "
        "Do NOT attempt to fix bugs or alter core logic. Focus on adding robust property checks."
    )
    SYSTEM_MESSAGE_CODEQWEN_CHECKER_END_TAG = "<|im_end|>"
    SYSTEM_MESSAGE_CODEQWEN_CHECKER_USER_TAG = "\n<|im_start|>user"
    # The actual user prompt will follow this



def get_metadata(put_run_exec, worker_id, samples, output_code, timeout, default_feedback):
    curr_res, curr_metadata = put_run_exec(worker_id, samples, output_code, timeout)
    import numpy as np
    if not np.all(curr_res):
        return curr_metadata
    else:
        return default_feedback
        

def check_testtype(testtype, platform):
    if type(testtype) is not str:
        if hasattr(testtype, "FUNCTIONAL"):
            testtype = 'functional'
        elif hasattr(testtype, "STDIN"):
            testtype = 'stdin'
        else:
            raise ValueError(f"Unknown testtype, {testtype}")
    if type(platform) is not str:
        if hasattr(platform, "LEETCODE"):
            platform = 'leetcode'
        elif hasattr(platform, "CODEFORCES"):
            platform = 'codeforces'
        elif hasattr(platform, "ATCODER"):
            platform = 'atcoder'
        elif hasattr(platform, "HUMANEVAL"):
            platform = 'humaneval'
        else:
            raise ValueError(f"Unknown platform, {platform}")
    if testtype == 'functional':
        if platform == 'humaneval' or platform == 'leetcode':
            return True
    if testtype == 'stdin':
        if platform == 'codeforces' or platform == 'atcoder':
            return True
    return False



def get_check_prompt(question: str, result, metadata):
    ## assumes i/o examples are already truncated!
    ## less pressure on storing 10 MB json because on a single large input-output pair
    # result_by_test_case = result
    # assert len(metadata) == 1, f"metadata = {metadata}"
    # metadata = metadata[0]
    try:
        metadata = json.loads(metadata) # Ensure parsing happens here
    except (json.JSONDecodeError, TypeError):
        # Handle cases where metadata_str is not a valid JSON string or None
        # print(f"Warning: Could not parse metadata_str: {metadata_str}")
        return "Context: No specific error context from previous run available or metadata was not parsable.\n"
    if "error_code" not in metadata: # Check after successful parsing
        return "Context: Previous run did not produce a standard error code.\n"
    
    message = ""
    error_code = metadata.get('error_code')

    if error_code == -1: # Compilation Error
        message = f"Context: The program previously failed with a compilation error.\nDetails: {metadata.get('error', 'N/A')}"
    elif error_code == -2: # Wrong Answer
        message = (f"Context: The program previously produced a wrong answer.\n"
                   f"Input: {metadata.get('inputs', 'N/A')}\n"
                   f"Generated Output: {metadata.get('output', 'N/A')}\n"
                   f"Expected Output: {metadata.get('expected', 'N/A')}")
    elif error_code == -3: # Time Limit Exceeded
        message = (f"Context: The program previously hit a time limit exceeded.\n"
                   f"Details: {metadata.get('error', 'N/A')}\n"
                   f"Input: {metadata.get('inputs', 'N/A')}\n"
                   f"Expected Output: {metadata.get('expected', 'N/A')}")
    elif error_code == -4: # Runtime Error
        if 'inputs' in metadata: # Check if 'inputs' key exists
             message = (f"Context: The program previously encountered a runtime error.\n"
                        f"Input: {metadata.get('inputs', 'N/A')}\n"
                        f"Expected Output: {metadata.get('expected', 'N/A')}\n"
                        f"Error Details: {metadata.get('error', 'N/A')}") # Assuming 'error' might have more details than 'error_message'
        else:
             message = f"Context: The program previously encountered a runtime error.\nError Message: {metadata.get('error_message', 'N/A')}"
    elif error_code == -5: # TestRunnerError
        message = (f"Context: The program previously caused a TestRunnerError.\n"
                   f"Error: {metadata.get('error', 'N/A')}\n"
                   f"Error Message: {metadata.get('error_message', 'N/A')}")
    else:
        message = f"Context: An unspecified error (code: {error_code}) occurred in a previous run.\n"
    return message + "\n" if message else "Context: No specific error details found for the given error code.\n"

    # if metadata["error_code"] == -1:
    #     # time limit exceeded
    #     message = f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"
    # elif metadata["error_code"] == -2:
    #     # wrong answer
    #     message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata['inputs']}\nGenerated Output: {metadata['output']}\nExpected: {metadata['expected']}"
    # elif metadata["error_code"] == -3:
    #     # time limit exceeded
    #     message = f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}"
    #     pass
    # elif metadata["error_code"] == -4:
    #     # runtime error
    #     if 'inputs' in metadata.keys():
    #         message = f"The above code is incorrect and got a runtime error.\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n{metadata['error']}"
    #     else:
    #         message = f"The above code is incorrect and got a runtime error.\nError_Message: {metadata['error_message']}"
    # elif metadata["error_code"] == -5:
    #     # TestRunnerError
    #     message = f"The above code is incorrect and got a runtime error.\n{metadata['error']}\nError_Message: {metadata['error_message']}"
    # else:
    #     raise NotImplementedError(
    #         f"metadata['error_code'] = {metadata['error_code']} not implemented || {metadata=}"
    #     )
    # return message


# For example: ### Example Response:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n    return a\n```\n\n\n### Example Answer:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n\n    # Checker\n    total = 1\n    for i in a:\n        total = total * i\n    assert total == n\n    \n    return a\n```\n




def get_deepseek_r1_question_template_answer(question: str, code: str, result, metadata):
    # v1
    # prompt = "You will be given a question (problem specification) and a Python program, then you will generate additional checker code in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer.\n\n"
    # prompt = "You will be given a question (problem specification) and a Python program, then you will generate additional checker (such as 'assert ...' or 'raise ...Error') in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer.\n\n"
    # prompt = "You will be given a question (problem specification), a Python program and a wrong answer testcase, then you will generate additional checker (such as 'assert ...', 'try ... except ...' or 'raise ...') in the Python program to prevent Wrong Answer and Runtime Errer by actively throw error with logs.\n\n"
    # prompt = "You are a meticulous code debugging assistant specializing in enhancing programs with validation safeguards. When the user provides a flawed program and a failing test case, your task is to modify the code to implement either:\nRuntime Assertion Checks: Insert validation logic at critical execution points to verify program state invariants\nCross-verification Mechanisms: Implement dual-path execution (comparing results from both optimized and reference implementations)\nCore Objectives:\nMaintain original program structure\nInsert strategic validation points without altering core logic\nForce explicit failure via exceptions on inconsistency detection\nEnable error identification without pre-known expected answers\n\n"
    # prompt = "You are a OnlineJudge Checker assistant specializing in enhancing programs with validation checks or expanding cross-verification (using a slower but more reliable method). When the user provides a problem description and a program, your task is to modify a given program to include validation checks or cross-verification logic before outputting results. The modified program must: 1. Act as a self-checking mechanism (like an Online Judge's checker). 2. Output the original answer only if all validations pass. 3. Explicitly throw an error/exception if inconsistencies are detected. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code. For example:\n### Example Response:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n    return a\n```\n\n\n### Example Answer:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n\n    # Checker\n    total = 1\n    for i in a:\n        total = total * i\n    assert total == n\n    \n    return a\n```\n"
    # prompt = "You are a Tester assistant specializing in enhancing programs with property-based verification or cross-verification (using a slower but more reliable method). When the user provides a problem description and a program, your task is to modify a given program to include property-based verification or cross-verification logic before outputting results. The modified program must: 1, Output the original answer only if all validations pass. 2, Explicitly throw an AssertError if inconsistencies are detected. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code. For example:\n### Example Original Code:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n    return a\n```\n\n\n### Example Your Answer:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n\n    # Checker\n    total = 1\n    for i in a:\n        total = total * i\n    assert total == n\n    \n    return a\n```\n End of the Example\n\n"
    # prompt += f"Question: {question}\n\n"
    # prompt += f"Original Code:\n```python\n{code}\n```\n\n"
    # prompt += get_check_prompt(question, result, metadata)
    # prompt += f"Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    # prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    # prompt += f"Answer: (use the provided format with backticks)\n\n"
    # prompt += f"<｜Assistant｜>"
    # return prompt
    
    
    # v2
    user_prompt = (
        "You are a Tester assistant. Your task is to enhance the given Python program by embedding property-based verification checks. "
        "These checks should be derived from the problem description. The goal is to make the program self-verifying: "
        "it should output its original result if all properties hold for the given input, or throw an `AssertError` if any property is violated. "
        "Do NOT modify the original problem-solving logic or fix any bugs. Focus solely on adding verification code. "
        "Think about what properties the output (and intermediate states, if relevant) should satisfy based on the problem description.\n\n"
        "For example:\n"
        "### Problem Description Example:\n"
        "Write a function `PrimeFactorization(n)` that returns a list of prime factors of `n`.\n\n"
        "### Example Original Code:\n"
        "```python\n"
        "def PrimeFactorization(n):\n"
        "    i = 2\n"
        "    a = []\n"
        "    while i * i <= n:\n"
        "        if (n // i) * i == n:\n"
        "            n = n // i\n"
        "            a.append(i)\n"
        "        else:"
        "            i += 1 \n"
        "    if n > 1:\n"
        "        a.append(n)\n"
        "    return a\n"
        "```\n\n"
        "### Example Your Enhanced Code (with Property Verification):\n"
        "```python\n"
        "def PrimeFactorization(n_orig):\n" # Use n_orig to preserve original input for checker
        "    n = n_orig # work with a copy\n"
        "    i = 2\n"
        "    a = []\n"
        "    while i * i <= n:\n"
        "        if (n // i) * i == n:\n"
        "            n = n // i\n"
        "            a.append(i)\n"
        "        else:\n"
        "            i += 1\n"
        "    if n > 1:\n"
        "        a.append(n)\n\n"
        "    # Property Verification (Checker)\n"
        "    # Property 1: The product of all factors should equal the original number.\n"
        "    product_of_factors = 1\n"
        "    for factor in a:\n"
        "        product_of_factors *= factor\n"
        "    assert product_of_factors == n_orig, f\"Product of factors {product_of_factors} does not equal original number {n_orig}\"\n\n"
        "    # Property 2 (Optional example): All returned factors should be prime (more complex to check here, but illustrates thinking)\n"
        "    # For simplicity, this example focuses on the product property.\n\n"
        "    return a\n"
        "```\n"
        "End of the Example.\n\n"
    )

    user_prompt += f"Now, consider the following problem and code:\n\n"
    user_prompt += f"### Problem Description:\n{question}\n\n"
    user_prompt += f"### Original Code:\n```python\n{code}\n```\n\n"

    # The get_check_prompt provides context if the code has known errors.
    # This can help the LLM understand *why* a checker might be needed,
    # but it should still focus on properties from the description, not just fixing this one case.
    error_context_message = get_check_prompt(question, result, metadata)
    if error_context_message:
        user_prompt += f"### Context from a Previous Run (if available, for understanding potential weaknesses - do NOT just fix this specific error):\n{error_context_message}\n\n"

    # Removed FORMATTING_WITHOUT_STARTER_CODE as system message now handles output format
    user_prompt += "Please provide the enhanced Python code with embedded property-based verification. "
    user_prompt += "Remember to place your reasoning within `<think></think>` tags and the final enhanced code within `<answer>\n```python\n...\n```\n</answer>` tags."
    user_prompt += f"<｜Assistant｜>"
    return user_prompt



def get_deepseek_coder_question_template_answer(question: str, code: str, result, metadata):
    # This prompt is for models like DeepSeek-Coder-Instruct (v1, v2)
    # It doesn't use the <think>/<answer> tags like DeepSeek-R1

    user_prompt_content = (
        "Your task is to enhance the given Python program by embedding property-based verification checks. "
        "These checks should be derived from the problem description. The goal is to make the program self-verifying: "
        "it should output its original result if all properties hold for the given input, or throw an `AssertError` if any property is violated. "
        "Do NOT modify the original problem-solving logic or fix any bugs. Focus solely on adding verification code. "
        "Think about what properties the output (and intermediate states, if relevant) should satisfy based on the problem description.\n\n"
        "For example:\n"
        "### Problem Description Example:\n"
        "Write a function `PrimeFactorization(n)` that returns a list of prime factors of `n`.\n\n"
        "### Example Original Code:\n"
        "```python\n"
        "def PrimeFactorization(n):\n"
        "    i = 2\n"
        "    a = []\n"
        "    while i * i <= n:\n"
        "        if (n // i) * i == n:\n"
        "            n = n // i\n"
        "            a.append(i)\n"
        "        else:\n" # Corrected example bug from your R1 prompt
        "            i += 1 \n"
        "    if n > 1:\n"
        "        a.append(n)\n"
        "    return a\n"
        "```\n\n"
        "### Example Your Enhanced Code (with Property Verification):\n"
        "```python\n"
        "def PrimeFactorization(n_orig):\n"
        "    n = n_orig\n"
        "    i = 2\n"
        "    a = []\n"
        "    while i * i <= n:\n"
        "        if (n // i) * i == n:\n"
        "            n = n // i\n"
        "            a.append(i)\n"
        "        else:\n"
        "            i += 1\n"
        "    if n > 1:\n"
        "        a.append(n)\n\n"
        "    # Property Verification (Checker)\n"
        "    product_of_factors = 1\n"
        "    for factor in a:\n"
        "        product_of_factors *= factor\n"
        "    assert product_of_factors == n_orig, f\"Product of factors {product_of_factors} does not equal original number {n_orig}\"\n\n"
        "    return a\n"
        "```\n"
        "End of the Example.\n\n"
    )

    user_prompt_content += f"Now, consider the following problem and code:\n\n"
    user_prompt_content += f"### Problem Description:\n{question}\n\n"
    user_prompt_content += f"### Original Code:\n```python\n{code}\n```\n\n"

    error_context_message = get_check_prompt(question, result, metadata)
    if error_context_message and "No specific error" not in error_context_message : # Only add if there's a real error context
        user_prompt_content += f"### Context from a Previous Run (if available, for understanding potential weaknesses - do NOT just fix this specific error):\n{error_context_message}\n\n"

    user_prompt_content += "Please provide ONLY the entire enhanced Python program with embedded property-based verification, enclosed in a single ```python ... ``` block. "
    user_prompt_content += "Do not add any explanations before or after the code block."

    prompt = f"### Instruction:\n{user_prompt_content}\n\n" # user_prompt_content now contains the detailed task.
    prompt += f"### Response:\n" # Model should start its code block after this.
    return prompt




def get_codeqwen_instruct_question_template_answer(question: str, code: str, result, metadata):
    # This prompt is for CodeQwen Instruct, which uses <|im_start|> and <|im_end|>

    user_prompt_content = (
        "Your task is to enhance the given Python program by embedding property-based verification checks. "
        "These checks should be derived from the problem description. The goal is to make the program self-verifying: "
        "it should output its original result if all properties hold for the given input, or throw an `AssertError` if any property is violated. "
        "Do NOT modify the original problem-solving logic or fix any bugs. Focus solely on adding verification code. "
        "Think about what properties the output (and intermediate states, if relevant) should satisfy based on the problem description.\n\n"
        "For example:\n"
        "### Problem Description Example:\n"
        "Write a function `PrimeFactorization(n)` that returns a list of prime factors of `n`.\n\n"
        "### Example Original Code:\n"
        "```python\n"
        "def PrimeFactorization(n):\n"
        "    i = 2\n"
        "    a = []\n"
        "    while i * i <= n:\n"
        "        if (n // i) * i == n:\n"
        "            n = n // i\n"
        "            a.append(i)\n"
        "        else:\n" # Corrected example bug
        "            i += 1 \n"
        "    if n > 1:\n"
        "        a.append(n)\n"
        "    return a\n"
        "```\n\n"
        "### Example Your Enhanced Code (with Property Verification):\n"
        "```python\n"
        "def PrimeFactorization(n_orig):\n"
        "    n = n_orig\n"
        "    i = 2\n"
        "    a = []\n"
        "    while i * i <= n:\n"
        "        if (n // i) * i == n:\n"
        "            n = n // i\n"
        "            a.append(i)\n"
        "        else:\n"
        "            i += 1\n"
        "    if n > 1:\n"
        "        a.append(n)\n\n"
        "    # Property Verification (Checker)\n"
        "    product_of_factors = 1\n"
        "    for factor in a:\n"
        "        product_of_factors *= factor\n"
        "    assert product_of_factors == n_orig, f\"Product of factors {product_of_factors} does not equal original number {n_orig}\"\n\n"
        "    return a\n"
        "```\n"
        "End of the Example.\n\n"
    )

    user_prompt_content += f"Now, consider the following problem and code:\n\n"
    user_prompt_content += f"### Problem Description:\n{question}\n\n"
    user_prompt_content += f"### Original Code:\n```python\n{code}\n```\n\n"

    error_context_message = get_check_prompt(question, result, metadata)
    if error_context_message and "No specific error" not in error_context_message : # Only add if there's a real error context
        user_prompt_content += f"### Context from a Previous Run (if available, for understanding potential weaknesses - do NOT just fix this specific error):\n{error_context_message}\n\n"

    user_prompt_content += "Please provide ONLY the entire enhanced Python program with embedded property-based verification, enclosed in a single ```python ... ``` block. "
    user_prompt_content += "Do not add any explanations before or after the code block."

    prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_CHECKER_USER_TAG}\n{user_prompt_content}<|im_end|>\n<|im_start|>assistant\n"
    # The model should start its code generation after this.
    return prompt
    


def format_prompt_checker_extend(
    question: str, LanguageModelStyle: LMStyle, code: str, result, metadata
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1}\n\n{get_deepseek_r1_question_template_answer(question, code, result, metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct: # For DeepSeek-Coder-V2 etc.
        system_message = PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_CODER_CHECKER
        user_content = get_deepseek_coder_question_template_answer(question, code, result, metadata)
        # Typical format: System\nUser_Content\nAssistant_Marker (if any)
        prompt = f"{system_message}\n\n{user_content}" # Response marker is in user_content
        return prompt
    elif LanguageModelStyle == LMStyle.CodeQwenInstruct: # For Qwen2.5-Coder etc.
        # CodeQwen's system message needs to be constructed carefully with its user message part
        system_part = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_CHECKER_SYSTEM_TAG}\n{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_CHECKER_CONTENT}{PromptConstants.SYSTEM_MESSAGE_CODEQWEN_CHECKER_END_TAG}"
        user_and_assistant_part = get_codeqwen_instruct_question_template_answer(question, code, result, metadata)
        prompt = f"{system_part}{user_and_assistant_part}" # user_and_assistant_part already starts with user tag and ends with assistant tag
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )




def test():
    print("--- Test for format_prompt_checker_extend ---")

    # 1. Define a sample problem description
    sample_question = (
        "Given a list of integers `nums`, return a new list where each element is squared. "
        "The input list will contain between 1 and 10 integers. Each integer will be between -100 and 100."
    )

    # 2. Define a sample "original code" (could be correct or incorrect for this test's purpose)
    sample_code = (
        "def square_elements(nums):\n"
        "    squared_list = []\n"
        "    for x in nums:\n"
        "        squared_list.append(x * x)\n"
        "    return squared_list"
    )

    # 3. Define a sample 'result' (False indicates the original code might have failed or needs checking)
    #    For this task, 'result' might not be strictly necessary if we always want to add PBT.
    #    Let's assume 'result=False' means we definitely want to run the checker extension.
    sample_result_passed = True # Let's test a case where it previously passed, but we still want PBT
    sample_result_failed = False

    # 4. Define sample 'metadata' as a JSON string
    # Case 1: No prior error (or passed)
    metadata_no_error_str = json.dumps({
        "notes": "Previously passed basic tests, but adding PBT for robustness."
        # No "error_code" key, or a positive one if you have such a convention
    })

    # Case 2: Previous Wrong Answer
    metadata_wrong_answer_str = json.dumps({
        "error_code": -2,
        "inputs": "[1, 2, 3]",
        "output": "[1, 4, 10]", # Example of a wrong output
        "expected": "[1, 4, 9]"
    })

    # Case 3: Previous Runtime Error
    metadata_runtime_error_str = json.dumps({
        "error_code": -4,
        "inputs": "[1, 'a', 3]", # Input causing a TypeError
        "expected": "[1, 1, 9]", # Hypothetical expected
        "error": "TypeError: unsupported operand type(s) for *: 'str' and 'str'"
    })


    print("\n--- Scenario 1: No prior specific error, adding PBT for robustness ---")
    prompt1 = format_prompt_checker_extend(
        sample_question,
        LMStyle.DeepSeekR1,
        sample_code,
        sample_result_passed, # 'result' might be True or False
        metadata_no_error_str
    )
    print(prompt1)

    print("\n--- Scenario 2: Previous Wrong Answer ---")
    prompt2 = format_prompt_checker_extend(
        sample_question,
        LMStyle.DeepSeekR1,
        sample_code, # Could be a version of the code that produced the WA
        sample_result_failed,
        metadata_wrong_answer_str
    )
    print(prompt2)

    print("\n--- Scenario 3: Previous Runtime Error ---")
    prompt3 = format_prompt_checker_extend(
        sample_question,
        LMStyle.DeepSeekR1,
        sample_code, # Could be a version of the code that produced the RTE
        sample_result_failed,
        metadata_runtime_error_str
    )
    print(prompt3)

    # Test the NotImplementedError for other LM styles
    try:
        print("\n--- Scenario 4: Test NotImplementedError ---")
        format_prompt_checker_extend(
            sample_question,
            LMStyle.OpenAIChat, # An unimplemented style
            sample_code,
            sample_result_failed,
            metadata_wrong_answer_str
        )
    except NotImplementedError as e:
        print(f"Caught expected error: {e}")

    print("\n--- End of Test ---")
    return


if __name__ == "__main__":
    test()
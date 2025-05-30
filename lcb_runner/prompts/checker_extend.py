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
    
    # SYSTEM_MESSAGE_DEEPSEEK_R1 = (
    #     "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
    #     #"The user gives a error program, and the Assistant needs to locate the error by adding additional checker code in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer. "
    #     #"The user gives a error program, and the Assistant needs to locate the error by adding additional checker (such as 'assert ...' or 'raise ...Error') in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer. "
    #     #"The user gives a error program and a wrong answer testcase, and the Assistant needs to locate the error by adding additional checker (such as 'assert ...', 'try ... except ...' or 'raise ...') in the Python program to prevent Wrong Answer and Runtime Errer by actively throw error with logs. "
    #     #"The user gives a error program and a wrong answer testcase, and the Assistant needs to add validation checks or expand cross-verification (using a less efficient but more reliable method) to the original program, ensuring that the program can detect errors even in the absence of an expected answer, and actively throw exceptions instead of terminating normally with incorrect output. "
    #     #"The user provides a problem description and a program. The Assistant needs to enhance the original program by adding validation checks or expanding cross-verification (using a less efficient but more reliable method, unrestricted by time limits) before the program outputs its answer. This transforms the program into a Checker similar to those used on Online Judge platforms, ensuring the following conditions: If the verification passes or cross-verification confirms correctness, output the original answer. Otherwise, throw an exception. This allows users to immediately determine whether the program is correct after inputting a test case, without relying on an external expected answer. The program should actively signal errors through exceptions rather than terminating normally with potentially incorrect output. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code."
    #     "The user provides a problem description and a program. The Assistant needs to enhance the original program by adding property-based verification before the program outputs its answer, ensuring the following conditions: If the property-based verification passes, output the original answer. Otherwise, throw an AssertError. This allows users to immediately determine whether the program is correct after inputting a test case, without relying on an external expected answer. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code by adding property-based verification."
    #     "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    #     "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    # )
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
    
    

def format_prompt_checker_extend(
    question: str, LanguageModelStyle: LMStyle, code: str, result, metadata
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1}\n\n{get_deepseek_r1_question_template_answer(question, code, result, metadata)}"
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
import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import Test, Platform


class PromptConstants:
    
    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        #"The user gives a error program, and the Assistant needs to locate the error by adding additional checker code in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer. "
        #"The user gives a error program, and the Assistant needs to locate the error by adding additional checker (such as 'assert ...' or 'raise ...Error') in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer. "
        #"The user gives a error program and a wrong answer testcase, and the Assistant needs to locate the error by adding additional checker (such as 'assert ...', 'try ... except ...' or 'raise ...') in the Python program to prevent Wrong Answer and Runtime Errer by actively throw error with logs. "
        #"The user gives a error program and a wrong answer testcase, and the Assistant needs to add validation checks or expand cross-verification (using a less efficient but more reliable method) to the original program, ensuring that the program can detect errors even in the absence of an expected answer, and actively throw exceptions instead of terminating normally with incorrect output. "
        #"The user provides a problem description and a program. The Assistant needs to enhance the original program by adding validation checks or expanding cross-verification (using a less efficient but more reliable method, unrestricted by time limits) before the program outputs its answer. This transforms the program into a Checker similar to those used on Online Judge platforms, ensuring the following conditions: If the verification passes or cross-verification confirms correctness, output the original answer. Otherwise, throw an exception. This allows users to immediately determine whether the program is correct after inputting a test case, without relying on an external expected answer. The program should actively signal errors through exceptions rather than terminating normally with potentially incorrect output. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code."
        "The user provides a problem description and a program. The Assistant needs to enhance the original program by adding property-based verification before the program outputs its answer, ensuring the following conditions: If the property-based verification passes, output the original answer. Otherwise, throw an AssertError. This allows users to immediately determine whether the program is correct after inputting a test case, without relying on an external expected answer. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code by adding property-based verification."
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
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
    metadata = json.loads(metadata)
    if "error_code" not in metadata:
        return ""
    if metadata["error_code"] == -1:
        # time limit exceeded
        message = f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"
    elif metadata["error_code"] == -2:
        # wrong answer
        message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata['inputs']}\nGenerated Output: {metadata['output']}\nExpected: {metadata['expected']}"
    elif metadata["error_code"] == -3:
        # time limit exceeded
        message = f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}"
        pass
    elif metadata["error_code"] == -4:
        # runtime error
        if 'inputs' in metadata.keys():
            message = f"The above code is incorrect and got a runtime error.\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n{metadata['error']}"
        else:
            message = f"The above code is incorrect and got a runtime error.\nError_Message: {metadata['error_message']}"
    elif metadata["error_code"] == -5:
        # TestRunnerError
        message = f"The above code is incorrect and got a runtime error.\n{metadata['error']}\nError_Message: {metadata['error_message']}"
    else:
        raise NotImplementedError(
            f"metadata['error_code'] = {metadata['error_code']} not implemented || {metadata=}"
        )
    return message


# For example: ### Example Response:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n    return a\n```\n\n\n### Example Answer:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n\n    # Checker\n    total = 1\n    for i in a:\n        total = total * i\n    assert total == n\n    \n    return a\n```\n


def get_deepseek_r1_question_template_answer(question: str, code: str, result, metadata):
    # prompt = "You will be given a question (problem specification) and a Python program, then you will generate additional checker code in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer.\n\n"
    # prompt = "You will be given a question (problem specification) and a Python program, then you will generate additional checker (such as 'assert ...' or 'raise ...Error') in the Python program to locate the error and actively throw error logs to prevent Wrong Answer and Runtime Errer.\n\n"
    # prompt = "You will be given a question (problem specification), a Python program and a wrong answer testcase, then you will generate additional checker (such as 'assert ...', 'try ... except ...' or 'raise ...') in the Python program to prevent Wrong Answer and Runtime Errer by actively throw error with logs.\n\n"
    # prompt = "You are a meticulous code debugging assistant specializing in enhancing programs with validation safeguards. When the user provides a flawed program and a failing test case, your task is to modify the code to implement either:\nRuntime Assertion Checks: Insert validation logic at critical execution points to verify program state invariants\nCross-verification Mechanisms: Implement dual-path execution (comparing results from both optimized and reference implementations)\nCore Objectives:\nMaintain original program structure\nInsert strategic validation points without altering core logic\nForce explicit failure via exceptions on inconsistency detection\nEnable error identification without pre-known expected answers\n\n"
    # prompt = "You are a OnlineJudge Checker assistant specializing in enhancing programs with validation checks or expanding cross-verification (using a slower but more reliable method). When the user provides a problem description and a program, your task is to modify a given program to include validation checks or cross-verification logic before outputting results. The modified program must: 1. Act as a self-checking mechanism (like an Online Judge's checker). 2. Output the original answer only if all validations pass. 3. Explicitly throw an error/exception if inconsistencies are detected. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code. For example:\n### Example Response:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n    return a\n```\n\n\n### Example Answer:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n\n    # Checker\n    total = 1\n    for i in a:\n        total = total * i\n    assert total == n\n    \n    return a\n```\n"
    prompt = "You are a Tester assistant specializing in enhancing programs with property-based verification or cross-verification (using a slower but more reliable method). When the user provides a problem description and a program, your task is to modify a given program to include property-based verification or cross-verification logic before outputting results. The modified program must: 1, Output the original answer only if all validations pass. 2, Explicitly throw an AssertError if inconsistencies are detected. Please note that you do not need to correct the code or solve the problem, but rather determine the correctness of the code. For example:\n### Example Original Code:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n    return a\n```\n\n\n### Example Your Answer:\n```python\ndef PrimeFactorization(n):\n    i = 2\n    a = []\n    while i * i <= n:\n        if (n // i) * i == n:\n            n = n // i\n            a.append(i)\n    if n > 1:\n        a.append(n)\n\n    # Checker\n    total = 1\n    for i in a:\n        total = total * i\n    assert total == n\n    \n    return a\n```\n End of the Example\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"Original Code:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"Answer: (use the provided format with backticks)\n\n"
    prompt += f"<｜Assistant｜>"
    return prompt


def format_prompt_checker_generate(
    question: str, LanguageModelStyle: LMStyle, code: str, result, samples
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    sample_input = samples["input_output"]["inputs"]
    sample_output = samples["input_output"]["outputs"]
    if LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1}\n\n{get_deepseek_r1_question_template_answer(question, code, result, metadata)}"
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )


def test():
    def write_str_or_json(prompt):
        if isinstance(prompt, str):
            fp.write(prompt)
        else:
            fp.write(json.dumps(prompt))
        return

    for lm_style in [LMStyle.OpenAIChat]:
        with open(
            "output/GPT-3.5-Turbo-0125/Scenario.codegeneration_10_0.2_eval_all.json"
        ) as f:
            check_metadata = json.load(f)[0]
        checked_base_question_cotent = check_metadata["question_content"]
        checked_base_codes = check_metadata["code_list"][0]
        checked_base_results = check_metadata["graded_list"][0]
        checked_base_metadata = check_metadata["metadata"][0]
        leetcode_prompt = format_prompt_checker_generate(
            checked_base_question_cotent,
            lm_style,
            checked_base_codes,
            checked_base_results,
            checked_base_metadata,
        )

        with open(f"/tmp/leetcode_{lm_style}.txt", "w") as fp:
            write_str_or_json(leetcode_prompt)
    return


if __name__ == "__main__":
    test()

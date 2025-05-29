import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import Test, Platform


class PromptConstants:

    SYSTEM_MESSAGE_CHAT_GENERIC_FOR_FUNCTION = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function.\
 You will calculate the input and output of the testcase and\
 write the whole assertion statement in the markdown code block with the correct output."
    
    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        "The user asks a question, and the Assistant generate extra test cases for the question. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )


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


def get_deepseek_r1_question_template_answer(question: str, code: str, platform: Platform):
    prompt = "You will be given a question (problem specification) and will write extra test cases to help to check the correctness of the code.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"Response:\n```python\n{code}\n```\n\n"
    if check_testtype("functional", platform):
        prompt += 'Format:\n```json\n[{"input": "# THE FUNCTION POSITIONAL PARAMETERS SEPARATED BY LINE BREAKS", "output": "# YOUR OUTPUT HERE", "testtype": "functional"}, ...]\n```\n\n'
    elif check_testtype("stdin", platform):
        prompt += 'Format:\n```json\n[{"input": "# YOUR INPUT STRING HERE", "output": "# YOUR OUTPUT STRING HERE", "testtype": "stdin"}, ...]\n```\n\n'
    else:
        raise ValueError("Cant find true platform and testtype")
    prompt += f"Answer: (use the provided json format with backticks)\n\n"
    prompt += f"<｜Assistant｜>"
    return prompt


def format_prompt_testcase_generate(
    question: str, LanguageModelStyle: LMStyle, code: str, result, platform: Platform
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1}\n\n{get_deepseek_r1_question_template_answer(question, code, platform)}"
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
        leetcode_prompt = format_prompt_testcase_generate(
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

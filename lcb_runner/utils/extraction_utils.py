from lcb_runner.lm_styles import LMStyle
import json
from lcb_runner.benchmarks.code_generation import Test


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    elif lmstyle == LMStyle.GenericBase:
        return model_output.strip()
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return ""
        # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def extract_testcase(model_output: str, lmstyle: LMStyle):
    testcase = extract_code(model_output, lmstyle)
    try:
        testcase = json.loads(testcase)
        if type(testcase) is list:
            testcase = [test for test in testcase]
        elif type(testcase) is dict:
            testcase = [testcase]
        else:
            raise ValueError("Unknown extra testcase")
        new_testcase = []
        for test in testcase:
            if type(test) is dict and "input" in test.keys() and "output" in test.keys() and "testtype" in test.keys():
                try:
                    testtype_testcase = Test(**test)
                    new_testcase.append(test)
                except:
                    pass
        if len(new_testcase) >= 1:
            return new_testcase
        else:
            return ""
    except:
        return ""


def extract_test_output_code(model_output: str, lmstyle: LMStyle = None):
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    if lmstyle and lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        # first try to extract ```python if not then try ```
        indexlines = [
            i
            for i, line in enumerate(outputlines)
            if "```python" in line or "```Python" in line
        ]
        if indexlines:
            start_index = indexlines[0]
        else:
            start_index = None
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if start_index is not None:
            indexlines = [i for i in indexlines if i > start_index]
            indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def extract_execution_code(model_output: str, lmstyle: LMStyle, cot: bool = False):
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()

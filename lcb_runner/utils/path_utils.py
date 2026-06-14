import pathlib

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario, TestCaseForRepair


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_cache_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    testcase, selfdebug = "", ""
    if args.scenario == Scenario.selfrepair:
        testcase: TestCaseForRepair = args.testcaseforrepair
        if args.selfdebug:
            selfdebug = "debug_"
        else:
            selfdebug = "_"
        if args.repairbase != Scenario.codegeneration:
            selfdebug += str(args.repairbase) + "_"
    n = args.n
    temperature = args.temperature
    path = f"cache/{model_repr}/{scenario}{selfdebug}{testcase}_{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    testcase, selfdebug = "", ""
    if args.scenario == Scenario.selfrepair:
        testcase: TestCaseForRepair = args.testcaseforrepair
        if args.selfdebug:
            selfdebug = "debug_"
        else:
            selfdebug = "_"
        if args.repairbase != Scenario.codegeneration:
            selfdebug += str(args.repairbase) + "_"
    exdirname = ""
    # if args.scenario == Scenario.codegeneration and args.selfdebug:
    #     scenario = Scenario.testcasegeneration # exec extra test case
    if str(args.release_version) == "humaneval" or str(args.release_version) == "codecontests" or str(args.release_version) == "mbpp":
        exdirname = "----" + str(args.release_version)
    if args.scenario == Scenario.selfrepair or args.scenario == Scenario.testcasegeneration:
        n = args.codegen_n
    else:
        n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}{exdirname}/{scenario}{selfdebug}{testcase}_{n}_{temperature}{cot_suffix}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    testcase, selfdebug = "", ""
    if args.scenario == Scenario.selfrepair:
        testcase: TestCaseForRepair = args.testcaseforrepair
        if args.selfdebug:
            selfdebug = "debug_"
        else:
            selfdebug = "_"
        if args.repairbase != Scenario.codegeneration:
            selfdebug += str(args.repairbase) + "_"
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}/{scenario}{selfdebug}{testcase}_{n}_{temperature}{cot_suffix}_eval_all.json"
    return path

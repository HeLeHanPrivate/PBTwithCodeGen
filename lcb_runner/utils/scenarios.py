from enum import Enum


class Scenario(Enum):
    codegeneration = "codegeneration"
    selfrepair = "selfrepair"
    testoutputprediction = "testoutputprediction"
    codeexecution = "codeexecution"
    testcasegeneration = "testcasegeneration"
    checkerextend = "checkerextend"


class TestCaseForRepair(Enum):
    allcase = "allcase"
    publiccase = "publiccase"
    simplecasegen = "simplecasegen"
    cotcasegen = "cotcasegen"
    checkergen = "checkergen"
    dynamicinput = "dynamicinput"
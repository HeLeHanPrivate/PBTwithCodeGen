import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset, load_from_disk


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"
    HUMANEVAL = "humaneval"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict
    extra_test: list[Test] = None

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date) if not isinstance(self.contest_date, datetime) else self.contest_date
        self.metadata = json.loads(self.metadata)  # type: ignore
        
        if self.platform != Platform.HUMANEVAL:
            self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
            self.public_test_cases = [Test(**t) for t in self.public_test_cases]
            if self.extra_test is not None:
                self.extra_test = json.loads(self.extra_test)  # type: ignore
                self.extra_test = [Test(**t) for t in self.extra_test]
            else:
                self.extra_test = []
            try:
                self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
            except:
                self.private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                        )
                    )
                )  # type: ignore
            self.private_test_cases = [Test(**t) for t in self.private_test_cases]
        else:
            self.public_test_cases = [Test(**{"input": t, "output": "", "testtype": TestType.FUNCTIONAL}) for t in self.public_test_cases]
            self.private_test_cases = [Test(**{"input": self.private_test_cases + "\n\n" + f"check({self.metadata['func_name']})", "output": "", "testtype": TestType.FUNCTIONAL})]
            if self.extra_test is not None:
                self.extra_test = [Test(**{"input": t, "output": "", "testtype": TestType.FUNCTIONAL}) for t in self.extra_test]
            else:
                self.extra_test = []

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "extra_test": self.extra_test,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        test_cases = sorted(self.public_test_cases, key=lambda x: len(str(x.input)), reverse=False)
        test_cases += sorted(self.extra_test, key=lambda x: len(str(x.input)), reverse=False)
        test_cases += sorted(self.private_test_cases, key=lambda x: len(str(x.input)), reverse=False)
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                    "platform": self.platform.value
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    # dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)
    if release_version == "humaneval":
        dataset = load_dataset(f"datasets/humaneval_livecodebenchtype")["train"]
    elif release_version == "codecontests":
        dataset = load_from_disk(f"datasets/codecontests_livecodebenchtype")
    elif release_version == "mbpp":
        dataset = load_dataset(f"datasets/mbpp_livecodebenchtype")["train"]
    else:
        dataset = load_dataset(f"datasets/livecodebench___code_generation_lite/release_latest-version_tag={release_version}")["test"]

    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()

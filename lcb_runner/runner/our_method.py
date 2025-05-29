import json
import numpy as np
from tqdm import tqdm
import threading
from queue import PriorityQueue
import time
import multiprocessing

from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
# print(
#     check_correctness(
#         {
#             "input_output": json.dumps(
#                 {
#                     "inputs": [
#                         json.dumps([1] * 100000)
#                         + "\n"
#                         + json.dumps([100000, -100000] * (100000 // 2))
#                     ],
#                     "outputs": [json.dumps([100000, 0] * (100000 // 2))],
#                     "fn_name": "mostFrequentIDs",
#                 }
#             )
#         },
#         "class Solution:\n    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:\n        from collections import defaultdict\n        \n        # Count of each ID\n        count = defaultdict(int)\n        # How many IDs exist for a given frequency\n        freq_of_count = defaultdict(int)\n        \n        max_freq = 0\n        ans = []\n        \n        for i in range(len(nums)):\n            x = nums[i]\n            change = freq[i]\n            \n            old_freq = count[x]\n            new_freq = old_freq + change\n            \n            # If there was an old frequency, decrease its usage\n            if old_freq > 0:\n                freq_of_count[old_freq] -= 1\n                if freq_of_count[old_freq] == 0:\n                    del freq_of_count[old_freq]\n            \n            # Update with the new frequency\n            count[x] = new_freq\n            freq_of_count[new_freq] += 1\n            \n            # Update max_freq if needed\n            if new_freq > max_freq:\n                max_freq = new_freq\n            \n            # If the collection at max_freq is empty, reduce max_freq until we find a non-empty bin\n            while max_freq > 0 and max_freq not in freq_of_count:\n                max_freq -= 1\n            \n            # If the collection is empty, max_freq will be 0\n            ans.append(max_freq)\n        \n        return ans",
#         6,
#         debug=True,
#     )
# )
# def check_correctness(sample, code_str, timeout, debug=True)

from lcb_runner.prompts.self_repair import format_prompt_self_repair
# def format_prompt_self_repair(question: str, LanguageModelStyle: LMStyle, code, result, metadata)
from lcb_runner.prompts.checker_extend import format_prompt_checker_extend
# def format_prompt_checker_extend(question: str, LanguageModelStyle: LMStyle, code: str, result, metadata)
from lcb_runner.prompts.test_case_generation import format_prompt_testcase_generate
# def format_prompt_testcase_generate(question: str, LanguageModelStyle: LMStyle, code: str, result, platform: Platform)
from lcb_runner.prompts.code_generation import format_prompt_generation
# def format_prompt_generation(question: CodeGenerationProblem, LanguageModelStyle: LMStyle)
from lcb_runner.prompts.checker_generate import format_prompt_checker_generate
# def format_prompt_checker_extend(question: str, LanguageModelStyle: LMStyle, code: str, result, samples)

from lcb_runner.utils.extraction_utils import extract_code, extract_testcase
# def extract_code(model_output: str, lmstyle: LMStyle)


run_answer_list = []


def run_exec(samples, code, timeout):
    curr_res = [-2]
    try:
        curr_res, curr_metadata = check_correctness(samples, code, timeout, False)
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
    except Exception as e:
        curr_metadata = {
            "error": repr(e),
            "error_code": -5,
            "error_message": "TestRunnerError",
        }
    finally:
        assert isinstance(curr_res, list), curr_res
        assert isinstance(curr_metadata, dict), curr_metadata
        return curr_res, curr_metadata

def run_exec_with_workid(samples, code, timeout, work_id):
    # print(f"新建子进程开始 {work_id} 的run_exec处理")
    answer = run_exec(samples, code, timeout)
    return answer, work_id


def run_exec_result_handler(result):
    answer, work_id = result
    global run_answer_list
    run_answer_list[work_id] = answer
    # print(f"子进程完成 {work_id} 的run_exec处理")


class MyPipeline:
    def __init__(self, args) -> None:
        self.checker_code_log = []
        self.condition = threading.Condition()
        self.request_queue = PriorityQueue()  # 主线程与子线程共享的队列
        self.prompts_answer_list = []
        self.platform = ""
        global run_answer_list
        run_answer_list = []
        self.run_answer_wait_flag = "Waiting Multiprocessing.Pool Answer"
        self.finished = threading.Event()
        self.thread_num = args.num_process_evaluate
        self.multiprocess_num = args.num_process_evaluate
        self.active_workers = 0
        self.testcase_generation_yes_num = 0
        self.testcase_generation_no_num = 0
    
    
    def prompts_to_code(self, worker_id, question_content, model_style, code, metadata, prompts_to_outputs, format_prompt, extract_func):
        output_code = ""
        try_times = 0
        while try_times < 3 and output_code == "":
            try_times += 1
            prompt = format_prompt(
                question_content,
                model_style,
                code,
                False,
                metadata,
            )
            
            if self.thread_num == 1:
                output = prompts_to_outputs([prompt])[0]
            else:
                # 提交结果给主线程并等待
                event = threading.Event()
                self.request_queue.put((worker_id, "prompts_to_code", prompt, event))
                # print(f"子线程 {worker_id} 提交prompts_to_code请求并等待")
                event.wait()  # 等待主线程处理
                # print(f"子线程 {worker_id} 收到恢复信号")
                output = self.prompts_answer_list[worker_id]
            
            output_code = extract_func(output[0], model_style) if type(output) is list else extract_func(output, model_style)
        if output_code == "":
            return code
        return output_code
    
    
    def put_run_exec(self, worker_id, samples, output_code, timeout):
        if self.thread_num == 1:
            curr_res, curr_metadata = run_exec(samples, output_code, timeout)
        else:
            # 提交结果给主线程并等待
            event = threading.Event()
            self.request_queue.put((worker_id, "run_exec", (samples, output_code, timeout), event))
            # print(f"子线程 {worker_id} 提交run_exec请求并等待")
            event.wait()  # 等待主线程处理
            # print(f"子线程 {worker_id} 收到恢复信号")
            global run_answer_list
            curr_res, curr_metadata = run_answer_list[worker_id]
        return curr_res, curr_metadata
    

    def get_public_input_output(self, problem):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in problem.public_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in problem.public_test_cases
                    ],
                    "fn_name": problem.metadata.get("func_name", None),
                }
            ),
        }

    def repair_code(self, worker_id, question_content, model_style, code, metadata, prompts_to_outputs, samples, args):
        output_code = ""
        try_times = 0
        fla = True
        original_metadata = metadata
        while try_times < 3 and fla:
            try_times += 1
            #print("repair_code", try_times)
            output_code = self.prompts_to_code(worker_id, question_content, model_style, code, original_metadata, prompts_to_outputs, format_prompt_self_repair, extract_code)
            curr_res, curr_metadata = self.put_run_exec(worker_id, samples, output_code, args.timeout)
            if np.all(curr_res):
                fla = False
        if output_code == "":
            output_code = code
        
        return output_code, fla


    def checker_extend(self, worker_id, question_content, model_style, code, metadata, prompts_to_outputs, samples, args):
        output_code = ""
        try_times = 0
        fla = True
        while try_times < 3 and fla:
            #print("checker_extend", try_times)
            output_code = self.prompts_to_code(worker_id, question_content, model_style, code, metadata, prompts_to_outputs, format_prompt_checker_extend, extract_code)
            curr_res, curr_metadata = self.put_run_exec(worker_id, samples, output_code, args.timeout)
            if "error_code" not in curr_metadata.keys() or curr_metadata["error_code"] != -2:
                fla = False
        return output_code, fla


    def extra_testcase(self, worker_id, question_content, model_style, code, platform, prompts_to_outputs, func_name, problem, args):
        testcase = self.prompts_to_code(worker_id, question_content, model_style, code, platform, prompts_to_outputs, format_prompt_testcase_generate, extract_testcase)
        if testcase == code:
            testcase = []
        if len(testcase) > 0:
            return {
                "input_output": json.dumps(
                    {
                        "inputs": [
                            t['input']
                            for t in testcase
                        ] + [
                            t.input
                            for t in problem.public_test_cases
                        ],
                        "outputs": [
                            "" # t.output, "The generated data only includes input"
                            for t in testcase
                        ] + [
                            t.output
                            for t in problem.public_test_cases
                        ],
                        "fn_name": func_name,
                        "platform": self.platform,
                    }
                ),
            }
        else:
            return ""


    def solve_one_problem(self, worker_id, question_content, code, public_grade, metadata, platform, problem, model_style, args, prompts_to_outputs):
        self.platform = platform
        samples = self.get_public_input_output(problem)
        checker_extend_code, fla = self.checker_extend(worker_id, question_content, model_style, code, metadata, prompts_to_outputs, samples, args)
        if not public_grade:
            repaired_code, _ = self.repair_code(worker_id, question_content, model_style, checker_extend_code, metadata, prompts_to_outputs, samples, args)
        else:
            testcase = self.extra_testcase(worker_id, question_content, model_style, code, platform, prompts_to_outputs, problem.metadata.get("func_name", None), problem, args)
            if testcase != "":
                curr_res, curr_metadata = self.put_run_exec(worker_id, testcase, checker_extend_code, args.timeout)
                if not np.all(curr_res):
                    self.testcase_generation_yes_num += 1
                else:
                    self.testcase_generation_no_num += 1
                repaired_code, _ = self.repair_code(worker_id, question_content, model_style, checker_extend_code, metadata, prompts_to_outputs, testcase, args)
            else:
                self.testcase_generation_no_num += 1
                repaired_code, _ = self.repair_code(worker_id, question_content, model_style, checker_extend_code, '{"error_code": "-2", "inputs": "Unknown", "output": "Unknown", "expected": "Unknown"}', prompts_to_outputs, samples, args)

        self.prompts_answer_list[worker_id] = repaired_code
        self.active_workers -= 1
        if self.active_workers == 0:
            self.finished.set()

    def get_train_data(self, worker_id, question_content, code, public_grade, metadata, platform, problem, model_style, args, prompts_to_outputs):
        pass

    def our_method_pipeline(self, benchmark, model_style, args, check_metadata_list, prompts_to_outputs):
        global run_answer_list
        outputs = [
            [None for _ in range(args.codegen_n)]
            for _ in range(len(benchmark))
        ]
        
        threads = []
        worker_id = 0
        prompt_index_to_question_idx = []
        prompt_index_to_code_idx = []
        
        yes_num, no_num, strong_public_num = 0, 0, 0
        for problem_idx, problem in tqdm(enumerate(benchmark)):
            for check_metadata_idx, check_metadata in enumerate(check_metadata_list):
                if problem.question_id == check_metadata['question_id']:
                    question_content = check_metadata["question_content"]
                    code_list = check_metadata["code_list"]
                    output_list = check_metadata["output_list"]
                    graded_list = check_metadata["graded_list"]
                    public_graded_list = check_metadata["public_graded_list"]
                    platform = check_metadata["platform"]
                    metadata = check_metadata["metadata"]
                    
                    for code_idx in range(len(code_list)):
                        if graded_list[code_idx]:
                            yes_num += 1
                            outputs[problem_idx][code_idx] = output_list[code_idx]
                            continue
                        else:
                            no_num += 1
                        if public_graded_list[code_idx] == False:
                            strong_public_num += 1
                        
                        if self.thread_num == 1:
                            outputs[problem_idx][code_idx] = self.solve_one_problem(
                                0,
                                question_content,
                                code_list[code_idx],
                                public_graded_list[code_idx],
                                metadata[code_idx],
                                platform,
                                problem,
                                model_style,
                                args,
                                prompts_to_outputs,
                            )
                            outputs[problem_idx][code_idx] = "```\n" + outputs[problem_idx][code_idx] + "\n```"
                        else:
                            t = threading.Thread(
                                target=self.solve_one_problem, 
                                args=(
                                    worker_id,
                                    question_content,
                                    code_list[code_idx],
                                    public_graded_list[code_idx],
                                    metadata[code_idx],
                                    platform,
                                    problem,
                                    model_style,
                                    args,
                                    prompts_to_outputs,
                                )
                            )
                            worker_id += 1
                            self.prompts_answer_list.append("")
                            run_answer_list.append("")
                            prompt_index_to_question_idx.append(problem_idx)
                            prompt_index_to_code_idx.append(code_idx)
                            self.active_workers += 1
                            threads.append(t)
                            t.start()
                            
        print("yes_num=", yes_num, "no_num=", no_num, "strong_public_num=", strong_public_num)
        
        if self.thread_num != 1:
            pool = multiprocessing.Pool(processes=self.multiprocess_num)
            prompts = []
            prompt_events = []
            prompts_worker_id = []
            run_exec_events = []
            run_exec_worker_id = []
            
            while not self.finished.is_set():
                try:
                    # 非阻塞获取请求，最多等待1秒
                    worker_id, event_type, data, event = self.request_queue.get(timeout=1)
                    # print(f"\n主线程处理 {worker_id} 的请求: {event_type}")
                    if event_type == "prompts_to_code":
                        # print(f"主线程完成 {worker_id} 的prompt预处理")
                        prompts.append(data)
                        prompt_events.append(event)
                        prompts_worker_id.append(worker_id)
                        self.request_queue.task_done()
                    elif event_type == "run_exec":
                        if self.multiprocess_num > 1:
                            pool.apply_async(
                                func=run_exec_with_workid,
                                args=(data[0], data[1], data[2], worker_id),
                                callback=run_exec_result_handler,
                                error_callback=lambda e: print(f"Multiprocessing.Pool run_exec Error: {e}")
                            )
                            run_answer_list[worker_id] = self.run_answer_wait_flag
                            run_exec_events.append(event)
                            run_exec_worker_id.append(worker_id)
                        else:
                            run_answer_list[worker_id] = run_exec(data[0], data[1], data[2])
                            # print(f"主线程完成 {worker_id} 的run_exec处理")
                            event.set()  # 唤醒对应子线程
                        self.request_queue.task_done()
                    else:
                        # print("主线程无待处理任务")
                        pass
                except:
                    # 处理队列空的情况
                    # print("主线程无待处理任务")
                    if self.active_workers == 0:
                        break
                if self.active_workers == len(prompts):
                    print()
                    # 一次处理所有LLM请求，利用batch提高速度
                    tmp_outputs = prompts_to_outputs(prompts)
                    for i in range(len(tmp_outputs)):
                        self.prompts_answer_list[prompts_worker_id[i]] = tmp_outputs[i]
                    for event in prompt_events:
                        event.set()
                    prompts = []
                    prompt_events = []
                    prompts_worker_id = []
                else:
                    # print("active_workers=", self.active_workers, "prompts_len=", len(prompts))
                    pass
                new_run_exec_events = []
                new_run_exec_worker_id = []
                for i in range(len(run_exec_worker_id)):
                    tmp_worker_id = run_exec_worker_id[i]
                    if run_answer_list[tmp_worker_id] == self.run_answer_wait_flag:
                        new_run_exec_worker_id.append(run_exec_worker_id[i])
                        new_run_exec_events.append(run_exec_events[i])
                    else:
                        run_exec_events[i].set()
                run_exec_events = []
                run_exec_worker_id = []
                for i in range(len(new_run_exec_worker_id)):
                    run_exec_events.append(new_run_exec_events[i])
                    run_exec_worker_id.append(new_run_exec_worker_id[i])
                
            pool.close()
            pool.join()
            
            for prompt_idx, output in enumerate(self.prompts_answer_list):
                question_idx = prompt_index_to_question_idx[prompt_idx]
                code_idx = prompt_index_to_code_idx[prompt_idx]
                outputs[question_idx][code_idx] = "```\n" + output + "\n```"
        
        print("extra_test_yes=", self.testcase_generation_yes_num, "extra_test_no=", self.testcase_generation_no_num)
        
        return outputs
import json
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=str)
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    filepath = args.eval_json
    
    with open(filepath) as f:
        data = json.load(f)
    metadatas = data[2]

    ans = [0] * 6
    ac = 0
    for prob_metadata in metadatas:
        for metadata in prob_metadata:
            item = json.loads(metadata)
            if "error_code" in item:
                ans[abs(item["error_code"])] += 1
            else:
                ac += 1
    total = sum(ans)
    assert total > 0
    normalized = [round(x / total, 3) for x in ans]
    print("AC   ", ac)
    print("WA   ", ans[2], normalized[2])
    print("TLE  ", ans[3], normalized[3])
    print("RE   ", ans[4], normalized[4])
    print("CE   ", ans[1], normalized[1])
    print("Other", ans[5], normalized[5])
    
    
    if args.debug:
        eval_all_filepath = filepath.replace("_eval.json", "_eval_all.json")
        with open(eval_all_filepath) as f:
            data = json.load(f)
        
        cnt = 0
        for prob in data:
            code = prob['code_list']
            if "original_code_list" in prob:
                old_code = prob['original_code_list']
            else:
                old_code = prob['code_list']
            for i, j, k, kk in zip(old_code, code, prob['metadata'], prob['output_list']):
                if i == j:
                    continue
                if cnt > 10:
                    continue
                cnt += 1
                print("question_content:", prob['question_content'])
                print("\n\n\n\n")
                print("*****meta_data*****:")
                print(kk)
                print("\n\n")
                print("*****old_code*****:")
                print(i)
                print("\n\n")
                print("*****new_code*****:")
                print(j)
                print("\n\n")
                print("*****results*****:")
                print(k)
                print("\n\n")
                pause = input("Any keys to continue")


if __name__ == "__main__":
    main()
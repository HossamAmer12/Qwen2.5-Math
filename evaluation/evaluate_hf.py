import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

def evaluate(benchmark: str, dataset_id: str, dataset_config: str = None, dataset_split: str = "test", dataset_col: str = "pred", samples: list=None, max_num_samples=None):
    
    if ".json" in dataset_id:
        # Local way
        samples = load_dataset("json", data_files=dataset_id, split=dataset_split)
    else:
        # Global way
        samples = load_dataset(dataset_id, name=dataset_config, split=dataset_split)
    
    
    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)
        
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]


    def parse_gt(x):
        x['gt_cot'], x['gt'] = parse_ground_truth(x, benchmark)
        return x

    samples = samples.map(parse_gt, desc="Parsing ground truth", num_proc=12, load_from_cache_file=False)    
    samples = samples.map(extract_answer_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=12, load_from_cache_file=False)
    params = [(idx, pred, gt) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'])]
    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=8) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    mean_score = np.mean(scores) * 100

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "acc": mean_score,
    }

    print(dataset_col, result_json)
    return samples, result_json, scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--voting_n", type=int, nargs='+', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # data = {"n": [], "acc_naive": [], "acc_weighted": [], "acc_maj": []}
    # data = {"n": [], "acc_naive": [], "acc_weighted": [], "acc_maj": [], "acc_cts": []}
    data = {"n": [], "acc_baseline": [], "acc_pass@k": [], "acc_maj": []}
    

    def evaluate_for_n(n):
        # local_data = {"n": n, "acc_naive": None, "acc_weighted": None, "acc_maj": None}
        local_data = {"n": n, "acc_baseline": None, "acc_pass@k": None, "acc_maj": None}
        
        # Hossam define the equality table
        for iagg, agg in enumerate(["baseline", "pass@k", "maj"]):

            if agg == "baseline":
                _, scores, cur_equality_table = evaluate(
                    benchmark=args.benchmark,
                    dataset_id=args.dataset_id,
                    dataset_config=args.dataset_config,
                    dataset_split=args.dataset_split,
                    dataset_col=f"pred_{agg}",
                    max_num_samples=args.max_num_samples,
                )
            elif agg == "pass@k":
                _, scores, cur_equality_table = evaluate(
                    benchmark=args.benchmark,
                    dataset_id=args.dataset_id,
                    dataset_config=args.dataset_config,
                    dataset_split=args.dataset_split,
                    dataset_col=f"pred_pass@{n}",
                    max_num_samples=args.max_num_samples,
                )
            else:
                _, scores, cur_equality_table = evaluate(
                    benchmark=args.benchmark,
                    dataset_id=args.dataset_id,
                    dataset_config=args.dataset_config,
                    dataset_split=args.dataset_split,
                    dataset_col=f"pred_{agg}@{n}",
                    max_num_samples=args.max_num_samples,
                )

            local_data[f"acc_{agg}"] = scores["acc"]
            # equality_table[iagg] = cur_equality_table
        
        # Hossam
        # Compute the best result among the all equality tables
        # if one is true, then the result is true
        # for j in range(1, len(equality_table[0])):
        #     equality_table[0][j] = equality_table[0][j] | equality_table[1][j] | equality_table[2][j] 
        
        # scores_cts = equality_table[0]
        # mean_score_cts = np.mean(scores_cts) * 100
        # local_data[f"acc_cts"] = mean_score_cts
        return local_data

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_for_n, n): n for n in args.voting_n}
        with tqdm(total=len(futures), desc="Evaluating voting_n") as progress_bar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    data["n"].append(result["n"])
                    data["acc_baseline"].append(result["acc_baseline"])
                    data["acc_pass@k"].append(result["acc_pass@k"])
                    data["acc_maj"].append(result["acc_maj"])
                except Exception as e:
                    print(f"Error processing n={futures[future]}: {e}")
                progress_bar.update(1)

    # Save results
    ds = Dataset.from_dict(data)
    print(ds)

    df = ds.to_pandas()
    print(df)
    # url = ds.push_to_hub(args.dataset_id, config_name=f"{args.dataset_config}--evals")
    # print(f"Results pushed to {url}")
    print("Done")

from trectools import TrecQrel, TrecRun, TrecEval
import pandas as pd
import os
from tqdm import tqdm
from fusion_methods import weighted_rrf
from itertools import product

if not os.path.exists("results/alone_scores"):
    os.makedirs("results/alone_scores")
if not os.path.exists("results/fusion_2"):
    os.makedirs("results/fusion_2")
if not os.path.exists("results/fusion_2/raw_results"):
    os.makedirs("results/fusion_2/raw_results")
if not os.path.exists("results/weights"):
    os.makedirs("results/weights")
if not os.path.exists("results/full_fusion"):
    os.makedirs("results/full_fusion")


def calc_scores_alone(fold, metric):
    output_path = f"results/alone_scores/fold_{fold}_{metric}.csv"
    if os.path.exists(output_path):
        return
    my_dict = {}
    runs = os.listdir("data/ROBUST")
    dir_path = "data/per_fold_qrels"
    qrels = TrecQrel(os.path.join(dir_path, f"fold_{fold}/train.txt"))
    assert len(qrels.topics()) == 80
    for run in runs:
        r = TrecRun(os.path.join("data/ROBUST", run))
        if metric == "map":
            score = TrecEval(r, qrels).get_map(per_query=True).dropna().mean().item()
        elif metric == "p@10":
            score = TrecEval(r, qrels).get_precision(depth=10, per_query=True).dropna().mean().item()
        else:
            raise NotImplementedError
        print(f"{run} - {score}")
        my_dict[run] = score
    df = pd.DataFrame.from_dict(my_dict, orient="index", columns=["score"])
    df.to_csv(output_path)


def fusion_2(fold, metric, fuse_method):
    output_path = f"results/fusion_2/raw_results/fold_{fold}_{metric}_{fuse_method}.csv"
    if os.path.exists(output_path):
        return
    my_dict = {}
    runs = os.listdir("data/ROBUST")
    dir_path = "data/per_fold_qrels"
    qrels = TrecQrel(os.path.join(dir_path, f"fold_{fold}/train.txt"))
    assert len(qrels.topics()) == 80
    runs_alone = {}
    for run in runs:
        r = TrecRun(os.path.join("data/ROBUST", run))
        runs_alone[run] = r
    for r1 in runs:
        for r2 in runs:
            if r1 < r2:
                if fuse_method == "rrf":
                    if metric == "map":
                        r1_score = TrecEval(runs_alone[r1], qrels).get_map(per_query=True).dropna().mean().item()
                        r2_score = TrecEval(runs_alone[r2], qrels).get_map(per_query=True).dropna().mean().item()
                    elif metric == "p@10":
                        r1_score = TrecEval(runs_alone[r1], qrels).get_precision(depth=10, per_query=True).dropna().mean().item()
                        r2_score = TrecEval(runs_alone[r2], qrels).get_precision(depth=10, per_query=True).dropna().mean().item()
                    else:
                        raise NotImplementedError
                    fused_run = weighted_rrf([runs_alone[r1], runs_alone[r2]], [r1_score, r2_score])
                else:
                    raise NotImplementedError
                if metric == "map":
                    score = TrecEval(fused_run, qrels).get_map(per_query=True).dropna().mean().item()
                elif metric == "p@10":
                    score = TrecEval(fused_run, qrels).get_precision(depth=10, per_query=True).dropna().mean().item()
                else:
                    raise NotImplementedError
                print(f"{r1} - {r2} - {score}")
                my_dict[(r1, r2)] = score
                my_dict[(r2, r1)] = score
    data = [(k[0], k[1], v) for k, v in my_dict.items()]  # Flatten the dictionary
    df = pd.DataFrame(data, columns=["r1", "r2", "score"])  # Define column names
    df.to_csv(output_path, index=False)


def get_full_fusion_weights(metric, fuse_method, weight_method):
    for i in range(1, 6):
        alone_df = pd.read_csv(f"results/alone_scores/fold_{i}_{metric}.csv")
        df = pd.read_csv(f"results/fusion_2/raw_results/fold_{i}_{metric}_{fuse_method}.csv")
        # Append rows from alone_df to df
        for _, row in alone_df.iterrows():
            r = row["Unnamed: 0"]
            score = row["score"]
            # Append new row with r1=r, r2=r, score=score
            df = pd.concat([df, pd.DataFrame({"r1": [r], "r2": [r], "score": [score]})], ignore_index=True)
        df = df.groupby(["r1"])["score"].sum().reset_index()
        weight_df = df.copy()
        if weight_method == "diffscore":
            weight_df["weight"] = weight_df["score"] - alone_df["score"].sum()
            weight_df = weight_df[["r1", "weight"]]
        elif weight_method == "fuse2sumscore":
            weight_df["weight"] = weight_df["score"]
            weight_df = weight_df[["r1", "weight"]]
        else:
            raise NotImplementedError
        weight_df.to_csv(f"results/weights/fold_{i}_{metric}_{fuse_method}_{weight_method}.csv", index=False)


def eval_full_fusion(weight_methods, metric, fuse_method):
    dir_path = "results/full_fusion"
    qrels_dir_path = "data/per_fold_qrels"
    output_path = os.path.join(dir_path, f"{metric}_{fuse_method}.csv")

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    results = []
    runs = os.listdir("data/ROBUST")
    runs_alone = {run: TrecRun(os.path.join("data/ROBUST", run)) for run in runs}

    for fold in tqdm(range(1, 6), desc="Processing folds"):
        row = {"fold": fold}
        qrels = TrecQrel(os.path.join(qrels_dir_path, f"fold_{fold}/test.txt"))

        for weight_method in tqdm(weight_methods, desc="Processing weight methods"):
            weights = []
            trec_runs = []

            if weight_method == metric:
                alone_df = pd.read_csv(f"results/alone_scores/fold_{fold}_{metric}.csv")
                for _, row_data in alone_df.iterrows():
                    run_name = row_data["Unnamed: 0"]
                    weight = row_data["score"]
                    trec_runs.append(runs_alone[run_name])
                    weights.append(weight)

            elif weight_method in ["diffscore", "fuse2sumscore"]:
                weights_df = pd.read_csv(f"results/weights/fold_{fold}_{metric}_{fuse_method}_{weight_method}.csv")
                for _, row_data in weights_df.iterrows():
                    run_name = row_data["r1"]
                    weight = row_data["weight"]
                    trec_runs.append(runs_alone[run_name])
                    weights.append(weight)

            elif weight_method == "uniform":
                trec_runs = list(runs_alone.values())
                weights = [1.0] * len(trec_runs)

            if fuse_method == "rrf":
                fused_run = weighted_rrf(trec_runs, weights)
                trec_eval = TrecEval(fused_run, qrels)
                maps = trec_eval.get_map(per_query=True).dropna()
                map_score = maps.mean().item()
                # map_std = maps.std().item()
                p_at_10s = trec_eval.get_precision(depth=10, per_query=True).dropna()
                p_at_10 = p_at_10s.mean().item()
                # p_at_10_std = p_at_10s.std().item()

                # row[f"{weight_method}_map"] = f"{map_score} ± {map_std}"
                # row[f"{weight_method}_p@10"] = f"{p_at_10} ± {p_at_10_std}"
                row[f"{weight_method}_map"] = map_score
                row[f"{weight_method}_p@10"] = p_at_10
            else:
                raise NotImplementedError

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    metrics = ["map", "p@10"]  # "map" or "p@10" or "J-measure"
    fuse_methods = ["rrf"]  # "rrf" or "borda" or "minmaxnorm" or "sumnorm"
    # "diffscore" or "fuse2sumscore", previous methods are "metric" or "uniform"
    weight_methods = ["diffscore", "fuse2sumscore", "map", "uniform"]
    # for metric in metrics:
    #     for fuse_method in fuse_methods:
    for metric, fuse_method in tqdm(product(metrics, fuse_methods), desc="iterating all metrics and fusion methods"):
        for fold in tqdm(range(1, 6), desc="Calculating scores alone"):
            calc_scores_alone(fold, metric)

        for fold in tqdm(range(1, 6), desc="Calculating fusion2"):
            fusion_2(fold, metric, fuse_method)

        for weight_method in ["diffscore", "fuse2sumscore"]:
            get_full_fusion_weights(metric, fuse_method, weight_method=weight_method)

        full_res_df = eval_full_fusion(weight_methods, metric, fuse_method)
        cols_lst = [col for col in full_res_df.columns if "map" in col]
        print(f"metric-{metric}, fuse_method-{fuse_method}:")
        print(full_res_df[cols_lst])
        print(full_res_df[cols_lst].mean(axis=0))

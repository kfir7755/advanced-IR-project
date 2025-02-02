from trectools import TrecQrel, TrecRun, TrecEval
import pandas as pd
import os
from tqdm import tqdm
from fusion_methods import weighted_rrf, weighted_borda,weighted_min_max,weighted_sumnorm
from remove_bad_rankers import get_good_rankers
from itertools import product
from config import *

datasets = ['ROBUST', 'ADHOC', 'ROUTING']

for dataset in datasets:
    if not os.path.exists(f"results/{dataset}/alone_scores"):
        os.makedirs(f"results/{dataset}/alone_scores")
    if not os.path.exists(f"results/{dataset}/fusion_2"):
        os.makedirs(f"results/{dataset}/fusion_2")
    if not os.path.exists(f"results/{dataset}/fusion_2/raw_results"):
        os.makedirs(f"results/{dataset}/fusion_2/raw_results")
    if not os.path.exists(f"results/{dataset}/weights"):
        os.makedirs(f"results/{dataset}/weights")
    if not os.path.exists(f"results/{dataset}/full_fusion"):
        os.makedirs(f"results/{dataset}/full_fusion")
    if not os.path.exists(f"results/{dataset}/full_retrieval"):
        os.makedirs(f"results/{dataset}/full_retrieval")
    if not os.path.exists(f"results/{dataset}/ap_per_query"):
        os.makedirs(f"results/{dataset}/ap_per_query")
    if not os.path.exists(f"results/{dataset}/p_at_10_per_query"):
        os.makedirs(f"results/{dataset}/p_at_10_per_query")


def r_topics_intersection(r, qrels):
    intersection = r.topics_intersection_with(qrels)
    r.run_data = r.run_data[r.run_data["query"].isin(intersection)]


def calc_scores_alone(fold, metric, dataset):
    output_path = f"results/{dataset}/alone_scores/fold_{fold}_{metric}.csv"
    if os.path.exists(output_path):
        return
    my_dict = {}
    runs = get_good_rankers(os.path.join("data", dataset), eval(f"{dataset.lower()}_n_queries"), eval(f"{dataset.lower()}_top_k_docs"))
    dir_path = f"data/{dataset}/per_fold_qrels"
    qrels = TrecQrel(os.path.join(dir_path, f"fold_{fold}/train.txt"))
    if dataset == "ROBUST":
        assert len(qrels.topics()) == 80
    else:
        assert len(qrels.topics()) == 40
    for run in runs:
        r = TrecRun(os.path.join("data", dataset, "rankers", run))
        r_topics_intersection(r, qrels)

        if dataset == "ROBUST":
            assert len(r.topics()) == 80
        else:
            assert len(r.topics()) == 40
        
        if metric == "map":
            score = TrecEval(r, qrels).get_map(depth=100)
        elif metric == "p@10":
            score = TrecEval(r, qrels).get_precision(depth=10)
        else:
            raise NotImplementedError
        # print(f"{run} - {score}")
        my_dict[run] = score
    df = pd.DataFrame.from_dict(my_dict, orient="index", columns=["score"])
    df.to_csv(output_path)


def fusion_2(fold, metric, fuse_method, dataset):
    output_path = f"results/{dataset}/fusion_2/raw_results/fold_{fold}_{metric}_{fuse_method}.csv"
    if os.path.exists(output_path):
        return
    my_dict = {}
    runs = get_good_rankers(os.path.join("data", dataset), eval(f"{dataset.lower()}_n_queries"), eval(f"{dataset.lower()}_top_k_docs"))
    for run in runs:
        assert "(1)" not in run
    dir_path = f"data/{dataset}/per_fold_qrels"
    qrels = TrecQrel(os.path.join(dir_path, f"fold_{fold}/train.txt"))

    if dataset == "ROBUST":
        assert len(qrels.topics()) == 80
    else:
        assert len(qrels.topics()) == 40

    runs_alone = {}
    for run in runs:
        r = TrecRun(os.path.join("data", dataset, "rankers", run))
        runs_alone[run] = r
    for r1 in runs:
        for r2 in runs:
            if r1 < r2:
                r_topics_intersection(runs_alone[r1], qrels)
                r_topics_intersection(runs_alone[r2], qrels)

                if metric == "map":
                    r1_score = TrecEval(runs_alone[r1], qrels).get_map(depth=100)
                    r2_score = TrecEval(runs_alone[r2], qrels).get_map(depth=100)
                elif metric == "p@10":
                    r1_score = TrecEval(runs_alone[r1], qrels).get_precision(depth=10)
                    r2_score = TrecEval(runs_alone[r2], qrels).get_precision(depth=10)
                else:
                    raise NotImplementedError

                if fuse_method == "rrf":
                    fused_run = weighted_rrf([runs_alone[r1], runs_alone[r2]], [r1_score, r2_score])
                elif fuse_method == "borda":
                    fused_run = weighted_borda([runs_alone[r1], runs_alone[r2]], [r1_score, r2_score])
                elif fuse_method == "minmaxnorm":
                    fused_run = weighted_min_max([runs_alone[r1], runs_alone[r2]], [r1_score, r2_score])
                elif fuse_method == "sumnorm":
                    fused_run = weighted_sumnorm([runs_alone[r1], runs_alone[r2]], [r1_score, r2_score])
                else:
                    raise NotImplementedError

                if metric == "map":
                    score = TrecEval(fused_run, qrels).get_map(depth=100)
                elif metric == "p@10":
                    score = TrecEval(fused_run, qrels).get_precision(depth=10)
                else:
                    raise NotImplementedError
                # print(f"{r1} - {r2} - {score}")
                my_dict[(r1, r2)] = score
                my_dict[(r2, r1)] = score
    data = [(k[0], k[1], v) for k, v in my_dict.items()]  # Flatten the dictionary
    df = pd.DataFrame(data, columns=["r1", "r2", "score"])  # Define column names
    df.to_csv(output_path, index=False)


def get_full_fusion_weights(metric, fuse_method, weight_method, dataset):
    for i in range(1, 6):
        alone_df = pd.read_csv(f"results/{dataset}/alone_scores/fold_{i}_{metric}.csv")
        df = pd.read_csv(f"results/{dataset}/fusion_2/raw_results/fold_{i}_{metric}_{fuse_method}.csv")
        # Append rows from alone_df to df
        for _, row in alone_df.iterrows():
            r = row["Unnamed: 0"]
            score = row["score"]
            # Append new row with r1=r, r2=r, score=score
            df = pd.concat([df, pd.DataFrame({"r1": [r], "r2": [r], "score": [score]})], ignore_index=True)
        df = df.groupby(["r1"])["score"].sum().reset_index()
        weight_df = df.copy()
        if "diffscore" in weight_method:
            weight_df["weight"] = weight_df["score"] - alone_df["score"].sum()
            weight_df = weight_df[["r1", "weight"]]
            if "ReLU" in weight_method:
                weight_df["weight"] = weight_df["weight"].apply(lambda x: max(0, x))
        elif weight_method == "fuse2sumscore":
            weight_df["weight"] = weight_df["score"]
            weight_df = weight_df[["r1", "weight"]]
        else:
            raise NotImplementedError
        weight_df.to_csv(f"results/{dataset}/weights/fold_{i}_{metric}_{fuse_method}_{weight_method}.csv", index=False)


def eval_full_fusion(weight_methods, metric, fuse_method, dataset):
    dir_path = f"results/{dataset}/full_fusion"
    qrels_dir_path = f"data/{dataset}/per_fold_qrels"
    output_path = os.path.join(dir_path, f"{metric}_{fuse_method}.csv")

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    results = []
    runs = get_good_rankers(os.path.join("data", dataset), eval(f"{dataset.lower()}_n_queries"), eval(f"{dataset.lower()}_top_k_docs"))
    for fold in tqdm(range(1, 6), desc="Processing folds"):
        qrels = TrecQrel(os.path.join(qrels_dir_path, f"fold_{fold}/test.txt"))
        runs_alone = {run: TrecRun(os.path.join("data", dataset, "rankers", run)) for run in runs}
        for r in runs_alone.values():
            r_topics_intersection(r, qrels)
        row = {"fold": fold}

        for weight_method in tqdm(weight_methods, desc="Processing weight methods"):
            weights = []
            trec_runs = []

            if weight_method == "metric":
                alone_df = pd.read_csv(f"results/{dataset}/alone_scores/fold_{fold}_{metric}.csv")
                for _, row_data in alone_df.iterrows():
                    run_name = row_data["Unnamed: 0"]
                    weight = row_data["score"]
                    trec_runs.append(runs_alone[run_name])
                    weights.append(weight)

            elif weight_method in ["diffscore", "ReLUdiffscore", "fuse2sumscore"]:
                weights_df = pd.read_csv(f"results/{dataset}/weights/fold_{fold}_{metric}_{fuse_method}_{weight_method}.csv")
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
            elif fuse_method == "borda":
                fused_run = weighted_borda(trec_runs, weights)
            elif fuse_method == "minmaxnorm":
                fused_run = weighted_min_max(trec_runs, weights)
            elif fuse_method == "sumnorm":
                fused_run = weighted_sumnorm(trec_runs, weights)
            else:
                raise NotImplementedError
            trec_eval = TrecEval(fused_run, qrels)
            fused_run.run_data.to_csv(f"results/{dataset}/full_retrieval/full_fold_{fold}_{metric}_{fuse_method}_{weight_method}.csv", index=False)
            ap_score_per_query = trec_eval.get_map(depth=100,per_query=True)
            ap_score_per_query.to_csv(f"results/{dataset}/ap_per_query/ap_fold_{fold}_{metric}_{fuse_method}_{weight_method}.csv", index=False,header=False)
            p_at_10_per_query = trec_eval.get_precision(depth=10,per_query=True)
            p_at_10_per_query.to_csv(f"results/{dataset}/p_at_10_per_query/p_at_10_fold_{fold}_{metric}_{fuse_method}_{weight_method}.csv", index=False,header=False)
            map_score = trec_eval.get_map(depth=100,per_query=False)
            p_at_10 = trec_eval.get_precision(depth=10)
            row[f"{weight_method}_map"] = map_score
            row[f"{weight_method}_p@10"] = p_at_10

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":

    datasets = ['ADHOC', 'ROUTING', 'ROBUST']

    metrics = ["map", "p@10"]  # "map" or "p@10"

    fuse_methods = ["rrf", "borda", "minmaxnorm", "sumnorm"]  # "rrf" or "borda" or "minmaxnorm" or "sumnorm"

    # "diffscore" or "ReLUdiffscore" or "fuse2sumscore", previous methods are "metric" or "uniform"
    weight_methods = ["diffscore", "ReLUdiffscore", "fuse2sumscore", "metric", "uniform"]

    my_list = product(datasets, metrics, fuse_methods)
    for dataset, metric, fuse_method in tqdm(my_list, desc="iterating all datasets, metrics and fusion methods",
                                    total=len(datasets) * len(metrics) * len(fuse_methods)):
        
        for fold in tqdm(range(1, 6), desc="Calculating scores alone"):
            calc_scores_alone(fold, metric, dataset)

        for fold in tqdm(range(1, 6), desc="Calculating fusion2"):
            fusion_2(fold, metric, fuse_method, dataset)

        for weight_method in ["diffscore", "ReLUdiffscore", "fuse2sumscore"]:
            get_full_fusion_weights(metric, fuse_method, weight_method=weight_method, dataset=dataset)

        full_res_df = eval_full_fusion(weight_methods, metric, fuse_method, dataset)
        map_cols = [col for col in full_res_df.columns if col.endswith("_map")]
        p_cols = [col for col in full_res_df.columns if str(col).endswith("_p@10")]
        print(f"metric-{metric}, fuse_method-{fuse_method}:")
        print(full_res_df[map_cols])
        print(full_res_df[map_cols].mean(axis=0))
        print(full_res_df[p_cols])
        print(full_res_df[p_cols].mean(axis=0))

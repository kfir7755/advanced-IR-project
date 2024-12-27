from trectools import TrecQrel, TrecRun, TrecEval
import pandas as pd
import os
from tqdm import tqdm
from fusion_methods import weighted_rrf

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
    my_dict = {}
    runs = os.listdir("data/ROBUST")
    dir_path = "data/per_fold_qrels"
    qrels = TrecQrel(os.path.join(dir_path, f"fold_{fold}/train.txt"))
    assert len(qrels.topics()) == 80
    for run in runs:
        r = TrecRun(os.path.join("data/ROBUST", run))
        if metric == "map":
            score = TrecEval(r, qrels).get_map(per_query=True).dropna().mean().item()
        else:
            raise NotImplementedError
        print(f"{run} - {score}")
        my_dict[run] = score
    df = pd.DataFrame.from_dict(my_dict, orient="index", columns=['score'])
    df.to_csv(output_path)


def fusion_2(fold, metric, fuse_method):
    output_path = f"results/fusion_2/raw_results/fold_{fold}_{metric}_{fuse_method}.csv"
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
                    else:
                        raise NotImplementedError
                    fused_run = weighted_rrf([runs_alone[r1], runs_alone[r2]], [r1_score, r2_score])
                else:
                    raise NotImplementedError
                if metric == "map":
                    score = TrecEval(fused_run, qrels).get_map(per_query=True).dropna().mean().item()
                else:
                    raise NotImplementedError
                print(f"{r1} - {r2} - {score}")
                my_dict[(r1, r2)] = score
                my_dict[(r2, r1)] = score
    data = [(k[0], k[1], v) for k, v in my_dict.items()]  # Flatten the dictionary
    df = pd.DataFrame(data, columns=['r1', 'r2', 'score'])  # Define column names
    df.to_csv(output_path, index=False)


def get_full_fusion_weights(metric, fuse_method, weight_method):
    weights_per_fold = []
    for i in range(1, 6):
        alone_df = pd.read_csv(f"results/alone_scores/fold_{i}_{metric}.csv")
        # print(alone_df)
        # print(alone_df['score'].sum())
        df = pd.read_csv(f"results/fusion_2/raw_results/fold_{i}_{metric}_{fuse_method}.csv")
        # Append rows from alone_df to df
        for _, row in alone_df.iterrows():
            r = row['Unnamed: 0']
            score = row['score']
            # Append new row with r1=r, r2=r, score=score
            df = pd.concat([df, pd.DataFrame({'r1': [r], 'r2': [r], 'score': [score]})], ignore_index=True)
        df = df.groupby(['r1'])['score'].sum().reset_index()
        weight_df = df.copy()
        if weight_method == 'diffscore':
            weight_df['weight'] = weight_df['score'] - alone_df['score'].sum()
            weight_df = weight_df[['r1', 'weight']]
        elif weight_method == 'fuse2sumscore':
            weight_df['weight'] = weight_df['score']
            weight_df = weight_df[['r1', 'weight']]
        else:
            raise NotImplementedError
        weight_df.to_csv(f"results/weights/fold_{i}_{metric}_{fuse_method}_{weight_method}.csv", index=False)


# def eval_full_fusion(weight_methods, metric, fuse_method):
#     """
#     :param weight_methods: should be ['diffscore', 'fuse2sumscore', metric, 'uniform'], different ways to weight the runs
#     :param metric: metric used to evaluate, like MAP or p@k
#     :param fuse_method: rrf, borda, ...
#     :return: df of the metric for each fold (rows) and weight method
#     """
#     dir_path = "results/full_fusion"
#     qrels_dir_path = "data/per_fold_qrels"
#     output_path = os.path.join(dir_path, f"{metric}_{fuse_method}.csv")
#     if not os.path.exists(output_path):
#         folds_scores = {}
#         runs = os.listdir("data/ROBUST")
#         runs_alone = {}
#         for run in runs:
#             r = TrecRun(os.path.join("data/ROBUST", run))
#             runs_alone[run] = r
#         for fold in range(1, 6):
#             weights = []
#             trec_runs = []
#             qrels = TrecQrel(os.path.join(qrels_dir_path, f"fold_{fold}/test.txt"))
#             if weight_method == 'map':
#                 alone_df = pd.read_csv(f"results/alone_scores/fold_{fold}_map.csv")
#                 for _, row in alone_df.iterrows():
#                     run_name = row['Unnamed: 0']
#                     weight = row['score']
#                     r = runs_alone[run_name]
#                     weights.append(weight)
#                     trec_runs.append(r)
#             elif weight_method in ['diffscore', 'fuse2sumscore']:
#                 pass
#             else:
#                 raise NotImplementedError
#
#             if fuse_method == 'rrf':
#                 fused_run = weighted_rrf(trec_runs, weights)
#                 score = TrecEval(fused_run, qrels).get_map(per_query=True).dropna().mean().item()
#                 folds_scores[fold] = score
#             else:
#                 raise NotImplementedError
#     else:
#         fold_scores_df = pd.read_csv(output_path)
#     return fold_scores_df

from trectools import TrecQrel, TrecRun, TrecEval
import pandas as pd
import os
from tqdm import tqdm
from fusion_methods import weighted_rrf

from trectools import TrecQrel, TrecRun, TrecEval
import pandas as pd
import os
from tqdm import tqdm
from fusion_methods import weighted_rrf


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
        row = {'fold': fold}
        qrels = TrecQrel(os.path.join(qrels_dir_path, f"fold_{fold}/test.txt"))

        for weight_method in tqdm(weight_methods):
            weights = []
            trec_runs = []

            if weight_method == metric:
                alone_df = pd.read_csv(f"results/alone_scores/fold_{fold}_{metric}.csv")
                for _, row_data in alone_df.iterrows():
                    run_name = row_data['Unnamed: 0']
                    weight = row_data['score']
                    trec_runs.append(runs_alone[run_name])
                    weights.append(weight)

            elif weight_method in ['diffscore', 'fuse2sumscore']:
                weights_df = pd.read_csv(f"results/weights/fold_{fold}_{metric}_{fuse_method}_{weight_method}.csv")
                for _, row_data in weights_df.iterrows():
                    run_name = row_data['r1']
                    weight = row_data['weight']
                    trec_runs.append(runs_alone[run_name])
                    weights.append(weight)

            elif weight_method == 'uniform':
                trec_runs = list(runs_alone.values())
                weights = [1.0] * len(trec_runs)

            if fuse_method == 'rrf':
                fused_run = weighted_rrf(trec_runs, weights)
                if metric == 'map':
                    score = TrecEval(fused_run, qrels).get_map(per_query=True).dropna().mean().item()
                else:
                    raise NotImplementedError
                row[weight_method] = score

            else:
                raise NotImplementedError

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    metric = "map"
    fuse_method = "rrf"
    # 'diffscore' or 'fuse2sumscore', previous methods are 'metric' or 'uniform'
    weight_methods = ['diffscore', 'fuse2sumscore', metric, 'uniform']

    for fold in tqdm(range(1, 6)):
        calc_scores_alone(fold, metric)

    for fold in tqdm(range(1, 6)):
        fusion_2(fold, metric, fuse_method)

    for weight_method in ['diffscore', 'fuse2sumscore']:
      get_full_fusion_weights(metric, fuse_method, weight_method=weight_method)

    print(eval_full_fusion(weight_methods, metric, fuse_method))
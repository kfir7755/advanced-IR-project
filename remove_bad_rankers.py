import os
from trectools import TrecRun


def get_good_rankers(dataset_path, n_queries=100, top_k_docs=1000):
    rankers = os.listdir(dataset_path)
    bad_rankers = ["input.humR03d (1).gz"]
    for ranker in rankers:
        run = TrecRun(os.path.join("data\ROBUST", ranker))
        if run.run_data.shape[0] < n_queries * top_k_docs:  # 100 queries x 1000 docs per query
            bad_rankers.append(ranker)
    return [ranker for ranker in rankers if ranker not in bad_rankers]



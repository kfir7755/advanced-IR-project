import os
from trectools import TrecRun


def get_good_rankers(dataset_path, n_queries=100, top_k_docs=1000):
    rankers = os.listdir(os.path.join(dataset_path, "rankers"))
    bad_rankers = ["input.humR03d (1).gz"]
    for ranker in rankers:
        run = TrecRun(os.path.join(dataset_path, "rankers", ranker))
        if run.run_data.shape[0] < n_queries * top_k_docs:  # if there is no missing data
            bad_rankers.append(ranker)
    
    to_ret = [ranker for ranker in rankers if ranker not in bad_rankers]
    # print(len(to_ret), len(rankers))
    return to_ret


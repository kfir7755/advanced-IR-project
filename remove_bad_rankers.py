import os
from trectools import TrecRun


def get_good_rankers_robust():
    rankers = os.listdir("data\ROBUST")
    bad_rankers = ["input.humR03d (1).gz"]
    for ranker in rankers:
        run = TrecRun(os.path.join("data\ROBUST", ranker))
        if run.run_data.shape[0] < 100_000:  # 100 queries x 1000 docs per query
            bad_rankers.append(ranker)
    return [ranker for ranker in rankers if ranker not in bad_rankers]



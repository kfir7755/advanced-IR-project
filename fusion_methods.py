import pandas as pd
import numpy as np
from trectools import TrecRun


def weighted_rrf(trec_runs, weights, k=60, max_docs=1000):
    """
        Implements a reciprocal rank fusion as define in
        ``Reciprocal Rank fusion outperforms Condorcet and individual Rank Learning Methods`` by Cormack, Clarke and Buettcher.

        Parameters:
            trec_runs: a list of TrecRun objects to fuse
            weights: a list of weights to apply to each run
            k: term to avoid vanishing importance of lower-ranked documents. Default value is 60 (default value used in their paper).
            max_docs: maximum number of documents in the final ranking
    """

    rows = []
    topics = set([])
    for r in trec_runs:
        topics = topics.union(r.topics())

    for topic in sorted(topics):
        doc_scores = {}
        for r, weight in zip(trec_runs, weights):
            docs_for_run = r.get_top_documents(topic, n=1000)

            for pos, docid in enumerate(docs_for_run, start=1):
                doc_scores[docid] = doc_scores.get(docid, 0.0) + weight / (k + pos)

        # Writes out information for this topic
        for rank, (docid, score) in enumerate(sorted(iter(doc_scores.items()), key=lambda x: (-x[1], x[0]))[:max_docs],
                                              start=1):
            rows.append((topic, "Q0", docid, rank, score, "reciprocal_rank_fusion_k=%d" % k))

    # Build a sample run with merged data
    merged_run = TrecRun(None)
    df = pd.DataFrame(rows)
    df.columns = ["query", "q0", "docid", "rank", "score", "system"]
    merged_run.load_run_from_dataframe(df)

    return merged_run

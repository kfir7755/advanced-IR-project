import pandas as pd
import numpy as np
from trectools import TrecRun


def weighted_rrf(trec_runs, weights, k=60, max_docs=1000):
    """
        Implements a reciprocal rank fusion as define in
        ``Reciprocal Rank fusion outperforms Condorcet and individual Rank Learning Methods`` by Cormack, Clarke and Buettcher.
        The final score for each document is the weighted sum of its scores across all input runs by WeightedCombSUM.

        Parameters:
            trec_runs: a list of TrecRun objects to fuse
            weights: a list of weights to apply to each run
            k: term to avoid vanishing importance of lower-ranked documents. Default value is 60 (default value used in their paper).
            max_docs: maximum number of documents in the final ranking

        Returns:
            TrecRun: A new TrecRun object containing the fused rankings

        Note:
            - Documents are sorted by score (descending) and docid (ascending) for tie-breaking
            - All scores are weighted by the corresponding run's weight
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


def weighted_borda(trec_runs, weights, max_docs=1000):
    """
    Implements weighted Borda fusion for document ranking.

    This implementation follows the Borda count method where each document gets a score based on
    its position in each ranked list. Documents not appearing in a list receive a penalty score
    of -1 (weighted by that list's weight), to keep the order consistent.
    The final score for each document is the weighted sum of its scores across all input runs by WeightedCombSUM.

    Formula used:
    - For documents in the list: score = weight * (1000 - position)
    - For documents not in the list: score = weight * (-1)

    Parameters:
        trec_runs (list): List of TrecRun objects, each containing a ranked list of documents
        weights (list): List of weights corresponding to each TrecRun
        max_docs (int, optional): Maximum number of documents to include in final ranking.
                                Defaults to 1000.

    Returns:
        TrecRun: A new TrecRun object containing the fused rankings

    Note:
        - Documents are sorted by score (descending) and docid (ascending) for tie-breaking
        - All scores are weighted by the corresponding run's weight
    """
    rows = []
    topics = set([])
    for r in trec_runs:
        topics = topics.union(r.topics())

    for topic in sorted(topics):
        doc_scores = {}
        all_docs = set()

        # Store documents for each run in a dictionary
        run_docs = {}
        for r in trec_runs:
            docs = r.get_top_documents(topic, n=1000)
            run_docs[r] = docs
            all_docs.update(docs)

        # Calculate scores using stored documents
        for r, weight in zip(trec_runs, weights):
            docs_for_run = run_docs[r]

            # Score documents that appear in this run
            for pos, docid in enumerate(docs_for_run, start=1):
                doc_scores[docid] = doc_scores.get(docid, 0.0) + weight * (1000 - pos)

            # Penalize documents that don't appear in this run (-1 score)
            missing_docs = all_docs - set(docs_for_run)
            for docid in missing_docs:
                doc_scores[docid] = doc_scores.get(docid, 0.0) + weight * (-1)

        # Sort by score (descending) and then by docid (ascending) for consistent tie-breaking
        sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:max_docs]

        for rank, (docid, score) in enumerate(sorted_docs, start=1):
            rows.append((topic, "Q0", docid, rank, score, "borda"))

    merged_run = TrecRun(None)
    df = pd.DataFrame(rows)
    df.columns = ["query", "q0", "docid", "rank", "score", "system"]
    merged_run.load_run_from_dataframe(df)

    return merged_run

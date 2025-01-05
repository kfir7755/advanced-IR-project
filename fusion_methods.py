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

def weighted_min_max(trec_runs, weights, max_docs=1000):
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

        min_weight = min(weights)
        max_weight = max(weights)
        # Calculate scores using stored documents
        for r, weight in zip(trec_runs, weights):
            docs_for_run = run_docs[r]

            # Score documents that appear in this run
            for docid in docs_for_run:
                if max_weight != min_weight:
                    doc_scores[docid] = doc_scores.get(docid, 0.0) + (weight - min_weight) / (max_weight - min_weight)
                else:
                    doc_scores[docid] = doc_scores.get(docid, 0.0)

        # Sort by score (descending) and then by docid (ascending) for consistent tie-breaking
        sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:max_docs]

        for rank, (docid, score) in enumerate(sorted_docs, start=1):
            rows.append((topic, "Q0", docid, rank, score, "minmaxnorm"))

    merged_run = TrecRun(None)
    df = pd.DataFrame(rows)
    df.columns = ["query", "q0", "docid", "rank", "score", "system"]
    merged_run.load_run_from_dataframe(df)

    return merged_run

def weighted_sumnorm(trec_runs, weights, max_docs=1000):
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

        top100_sum = np.nansum(np.sort(weights)[-100:])
        # Calculate scores using stored documents
        for r, weight in zip(trec_runs, weights):
            docs_for_run = run_docs[r]

            # Score documents that appear in this run
            for docid in docs_for_run:
                if top100_sum!=0:
                    doc_scores[docid] = doc_scores.get(docid, 0.0) + weight/top100_sum
                else:
                    doc_scores[docid] = doc_scores.get(docid, 0.0)

        # Sort by score (descending) and then by docid (ascending) for consistent tie-breaking
        sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:max_docs]

        for rank, (docid, score) in enumerate(sorted_docs, start=1):
            rows.append((topic, "Q0", docid, rank, score, "minmaxnorm"))

    merged_run = TrecRun(None)
    df = pd.DataFrame(rows)
    df.columns = ["query", "q0", "docid", "rank", "score", "system"]
    merged_run.load_run_from_dataframe(df)

    return merged_run
#
# def combos(trec_runs, weights, strategy="minmax", max_docs=1000):
#     """
#         Implements a many of the traditional score fusion methods. Use the parameter strategy to pick a method.
#
#         Parameters:
#             trec_runs: a list of TrecRun objects to fuse
#            strategy: "sum", "max", "min", "anz", "mnz", "med"
#             max_docs: can be either a single integer or a dict{qid,value}
#     """
#     dfs = []
#     for t in trec_runs:
#         dfs.append(t.run_data)
#
#     # Merge all runs
#     """
#     merged = reduce(lambda left,right: pd.merge(left, right, right_on=["query","docid"], left_on=["query","docid"], how="outer",
#         suffixes=("","_")), dfs)
#     merged = merged[["query", "docid", "score", "score_"]]
#     """
#
#     if len(dfs) < 2:
#         return
#
#     merged = pd.merge(dfs[0], dfs[1], right_on=["query", "docid"], left_on=["query", "docid"], how="outer", suffixes=("", "_"))
#     merged = merged[["query", "q0", "docid", "score", "score_"]]
#
#     for d in dfs[2:]:
#         merged = pd.merge(merged, d, right_on=["query", "docid"], left_on=["query", "docid"], how="outer", suffixes=("", "_"))
#         merged = merged[["query", "q0", "docid", "score", "score_"]]
#
#     # merged["query"] = merged["query"].astype(str).apply(lambda x:x.strip())
#     # return merged
#
#     # merged.fillna(0.0, inplace=True) <- not filling nan's. Instead, I am using np.nan* functions
#     # TODO: add option to normalize values
#     # TODO: add option to act on the rank of documents instead of their scores
#
#     if strategy == "minmaxnorm":
#         # def minmax_scale(values):
#         #     min_score = np.nanmin(values)
#         #     max_score = np.nanmax(values)
#         #     if max_score==min_score:
#         #         return values
#         #     return (values - min_score)/(max_score - min_score)
#         def minmax_scale(values):
#             min_score = np.nanmin(values, axis=1, keepdims=True)
#             max_score = np.nanmax(values, axis=1, keepdims=True)
#             with np.errstate(invalid='ignore', divide='ignore'):
#                 result = (values - min_score) / (max_score - min_score)
#             result[np.isnan(result)] = 0  # Handle NaNs after division
#             return np.nanmean(result, axis=1)  # Average score per document
#
#         merged["ans"] = minmax_scale(merged[["score", "score_"]].values)
#
#     elif strategy == "sumnorm":
#         # def normalize_by_top100_sum(scores):
#         #     top100_docs_sum = np.nansum(np.sort(scores)[-100:])
#         #     if top100_docs_sum==0:
#         #         return scores
#         #     return scores/top100_docs_sum
#         def normalize_by_top100_sum(scores):
#             top100_sum = np.nansum(np.sort(scores, axis=1)[:, -100:], axis=1)
#             normalized = np.divide(scores.sum(axis=1), top100_sum, out=np.zeros_like(scores.sum(axis=1)),
#                                    where=top100_sum != 0)
#             return normalized
#
#         merged["ans"] = normalize_by_top100_sum(merged[["score", "score_"]].values)
#
#     else:
#         print("Unknown strategy %s. Options are: 'minmaxnorm','sumnorm" % (strategy))
#         return None
#
#     # merged["ans"] = merged[["score", "score_"]].apply(merge_func, raw=True, axis=1)
#     #TODO: verify this is corrrect
#     # merged["ans"] = minmax_scale(merged["score"].fillna(0) + merged["score_"].fillna(0))
#     merged.sort_values(["query", "ans"], ascending=[True, False], inplace=True)
#
#     rows = []
#     for topic in merged['query'].unique():
#         merged_topic = merged[merged['query'] == topic]
#         if type(max_docs) == dict:
#             maxd = max_docs[topic]
#             for rank, (docid, score) in enumerate(merged_topic[["docid", "ans"]].head(maxd).values, start=1):
#                 rows.append((topic, "Q0", docid, rank, score, "comb_%s" % strategy))
#         else:
#             for rank, (docid, score) in enumerate(merged_topic[["docid", "ans"]].head(max_docs).values, start=1):
#                 rows.append((topic, "Q0", docid, rank, score, "comb_%s" % strategy))
#
#     merged_run = TrecRun(None)
#     df = pd.DataFrame(rows)
#     df.columns = ["query", "q0", "docid", "rank", "score", "system"]
#     merged_run.load_run_from_dataframe(df)
#
#     return merged_run


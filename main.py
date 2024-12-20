from trectools import TrecQrel, TrecRun, TrecEval, fusion
import pandas as pd
import os

runs = os.listdir("data/ROBUST")
qrels = TrecQrel("data/robust03_qrels.txt")

# r1 = TrecRun("data/ROBUST/input.aplrob03a.gz")
# r2 = TrecRun("data/ROBUST/input.aplrob03b.gz")
# fused_run = fusion.reciprocal_rank_fusion([r1, r2])

# te1 = TrecEval(r1, qrels)
# te2 = TrecEval(r2, qrels)
# te3 = TrecEval(fused_run, qrels)
#
# print(te1.get_map())
# print(te2.get_map())
# print(te3.get_map())
import pandas as pd
from trectools import TrecQrel
import numpy as np

np.random.seed(42)

# Load the qrels into a TrecQrel object
qrels = TrecQrel("data/robust03_qrels.txt")

topics = qrels.topics()
topics = np.array(sorted(list(topics)))

np.random.shuffle(topics)

n_folds = 5
fold_size = len(topics) // n_folds
folds = []
for i in range(n_folds):
    start = i * fold_size
    end = (i + 1) * fold_size
    folds.append(topics[start:end])

for i, fold in enumerate(folds):
    print(f"Fold {i + 1}: len(folds)={len(fold)}")

for i, fold in enumerate(folds):
    print(f"Fold {i + 1}: {fold}")


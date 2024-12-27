import pandas as pd
from trectools import TrecQrel
import numpy as np
import os

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

# Fold 1: ['634' '604' '621' '439' '436' '419' '372' '631' '344' '303' '356' '394'
#  '624' '401' '641' '320' '627' '628' '346' '397']

if not os.path.exists("data/per_fold_qrels"):
    os.mkdir("data/per_fold_qrels")

    for i in range(len(folds)):
        # if not os.path.exists(os.path.join("data/per_fold_qrels", f"fold_{i + 1}")):
        os.mkdir(os.path.join("data/per_fold_qrels", f"fold_{i + 1}"))
        train_output_file = os.path.join("data/per_fold_qrels", f"fold_{i + 1}/train.txt")
        test_output_file = os.path.join("data/per_fold_qrels", f"fold_{i + 1}/test.txt")
        train_folds = folds[:i] + folds[i + 1:]
        test_fold = folds[i]
        with open(test_output_file, 'w') as out_file:
            for topic in test_fold:
                path = f"data/per_query_qrels/{topic}.txt"
                with open(path, 'r') as in_file:
                    for line in in_file:
                        out_file.write(line)

        with open(train_output_file, 'w') as out_file:
            for fold in train_folds:
                for topic in fold:
                    path = f"data/per_query_qrels/{topic}.txt"
                    with open(path, 'r') as in_file:
                        for line in in_file:
                            out_file.write(line)


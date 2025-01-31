import pandas as pd
from trectools import TrecQrel
import numpy as np
import os

datasets = ['ROBUST', 'ADHOC', 'ROUTING']
for dataset in datasets:
    print(dataset)

    np.random.seed(42)

    # Load the qrels into a TrecQrel object
    qrels = TrecQrel(os.path.join("data", dataset, f"{dataset.lower()}03_qrels.txt"))

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

    # Fold 1 for ROBUST: ['634' '604' '621' '439' '436' '419' '372' '631' '344' '303' '356' '394'
    #  '624' '401' '641' '320' '627' '628' '346' '397']

    in_dir_path = os.path.join("data", dataset, "per_query_qrels")
    out_dir_path = os.path.join("data", dataset, "per_fold_qrels")
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    for i in range(n_folds):
        if not os.path.exists(os.path.join(out_dir_path, f"fold_{i + 1}")):
            fold_i_path = os.path.join(out_dir_path, f"fold_{i + 1}")
            os.mkdir(fold_i_path)
        train_output_file = os.path.join(fold_i_path, "train.txt")
        test_output_file = os.path.join(fold_i_path, "test.txt")
        train_folds = folds[:i] + folds[i + 1:]
        test_fold = folds[i]

        for output_path in [train_output_file, test_output_file]:
            with open(output_path, 'w') as out_file:
                for topic in test_fold:
                    path = os.path.join(in_dir_path, f"{topic}.txt")
                    with open(path, 'r') as in_file:
                        for line in in_file:
                            out_file.write(line)



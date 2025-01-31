import os

robust_dataset_path = 'data\ROBUST'
robust_qrels_path = os.path.join(robust_dataset_path, 'robust03_qrels.txt')
robust_n_queries = 100
robust_top_k_docs = 1000

adhoc_dataset_path = 'data\ADHOC'
adhoc_qrels_path = os.path.join(adhoc_dataset_path, 'adhoc03_qrels.txt')
adhoc_n_queries = 50
adhoc_top_k_docs = 1000

routing_dataset_path = 'data\ROUTING'
routing_qrels_path = os.path.join(routing_dataset_path, 'routing03_qrels.txt')
routing_n_queries = 50
routing_top_k_docs = 1000
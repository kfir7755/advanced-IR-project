from config import *

for dataset in ['ROUTING', 'ADHOC']:
    dataset_path = eval(f'{dataset.lower()}_dataset_path')
    qrels_path = eval(f'{dataset.lower()}_qrels_path')
    qrels_dir_path = os.path.join(dataset_path, 'qrels')
    qrels_files = os.listdir(qrels_dir_path)
    
    with open(qrels_path, 'w') as f:
        for qrels_file in qrels_files:
            with open(os.path.join(qrels_dir_path, qrels_file), 'r') as qf:
                f.write(qf.read())

    
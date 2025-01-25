import pandas as pd
import os


def average_on_all_results():
    path = 'results/full_fusion'
    files = os.listdir(path)
    # files = [f for f in files if any(['sumnorm' in f, 'minmaxnorm' in f])]
    
    # Read all CSV files into a list of DataFrames
    results = [pd.read_csv(os.path.join(path, file)) for file in files if file.endswith('.csv')]
    for result, file in zip(results, files):
        result['metric_used'] = file.split('_')[0]
        result['fusion_method'] = file.split('_')[1][:-4]
        result = result.drop('fold', axis=1)

    # Stack all DataFrames and compute the mean
    combined = pd.concat(results, axis=0)  # Concatenate all DataFrames row-wise
    averaged = combined.groupby(['metric_used', 'fusion_method']).mean()  # Group by index and compute the mean

    averaged.columns = pd.MultiIndex.from_tuples(
    [col.split('_') for col in averaged.columns],
    names=['weight method', 'evaluation metric']
    )
    # Sort columns by the hierarchical index for better organization
    averaged = averaged.sort_index(axis=1)
    averaged = averaged[[col for col in averaged.columns if 'fold' not in col]]
    # averaged.loc['overall'] = averaged.mean()
    print(averaged, '\n'*5)
    averaged.to_latex('results/averaged_results.tex', float_format='%.3f', na_rep='-', multicolumn_format='c')


def average_on_all_results_without_sumnorm():
    path = 'results/full_fusion'
    files = os.listdir(path)
    files = [f for f in files if 'sumnorm' not in f]
    
    # Read all CSV files into a list of DataFrames
    results = [pd.read_csv(os.path.join(path, file)) for file in files if file.endswith('.csv')]
    for result, file in zip(results, files):
        result['metric_used'] = file.split('_')[0]
        result['fusion_method'] = file.split('_')[1][:-4]
        result = result.drop('fold', axis=1)

    # Stack all DataFrames and compute the mean
    combined = pd.concat(results, axis=0)  # Concatenate all DataFrames row-wise
    averaged = combined.groupby(['metric_used', 'fusion_method']).mean()  # Group by index and compute the mean

    averaged.columns = pd.MultiIndex.from_tuples(
    [col.split('_') for col in averaged.columns],
    names=['weight method', 'evaluation metric']
    )
    # Sort columns by the hierarchical index for better organization
    averaged = averaged.sort_index(axis=1)
    averaged = averaged[[col for col in averaged.columns if 'fold' not in col]]
    # averaged.loc['overall'] = averaged.mean()
    print(averaged, '\n'*5)
    averaged.to_latex('results/averaged_results_no_sumnorm.tex', float_format='%.3f', na_rep='-', multicolumn_format='c')


if __name__ == '__main__':
    average_on_all_results()
    average_on_all_results_without_sumnorm()
import os
import pandas as pd
from scipy.stats import ttest_rel

datasets_names = ['ROBUST', 'ADHOC', 'ROUTING']
results_dirertories = ["results/ROBUST/ap_per_query","results/ADHOC/ap_per_query","results/ROUTING/ap_per_query"]
p_at_10_results_directories = ["results/ROBUST/p_at_10_per_query","results/ADHOC/p_at_10_per_query","results/ROUTING/p_at_10_per_query"]

for results_dir,ds_name in zip(results_dirertories,datasets_names):
    # Initialize a dictionary to store data grouped by fusion and weight methods
    grouped_data = {}

    # Loop through all files in the results directory
    for file_name in os.listdir(results_dir):
        if file_name.endswith(".csv"):
            # Extract fold, fusion method, and weight method from the filename
            parts = file_name.replace(".csv", "").split("_")
            fold = parts[2]  # Fold number
            metric = parts[3]
            fusion_method = parts[4]
            weight_method = parts[5]

            # Read the file
            file_path = os.path.join(results_dir, file_name)
            data = pd.read_csv(file_path, header=None, names=["APs"])  # Assuming single column
            # Create a unique key for grouping
            key = (fusion_method, weight_method,metric)

            # Add data to the grouped dictionary
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append((int(fold), data))  # Store fold number for sorting

    # Concatenate and save the results
    concatenated_data_dict = {}
    for (fusion_method, weight_method,metric), fold_data in grouped_data.items():
        # Sort by fold number
        # fold_data.sort(key=lambda x: x[0])  # Sort by fold number (first element of tuple)
        # Concatenate all APs values in the correct order
        concatenated_data = pd.concat([df for _, df in fold_data], ignore_index=True)
        concatenated_data_dict[(fusion_method, weight_method,metric)] = concatenated_data

    # Step 3: Perform T-tests
    t_test_results = []

    alpha = 0.05

    # Compare uniform and metric to each new method
    for (fusion_method, weight_method,metric), data in concatenated_data_dict.items():
        # Retrieve uniform and metric data
        uniform_data = concatenated_data_dict[(fusion_method, "uniform",metric)]["APs"].values
        metric_data = concatenated_data_dict[(fusion_method, "metric",metric)]["APs"].values

        if weight_method in ["uniform", "metric"]:
            continue  # Skip existing methods since we already loaded them for comparison

        current_data = data["APs"].values

        # Perform T-test
        t_stat_uniform, p_value_uniform = ttest_rel(uniform_data, current_data)
        t_stat_metric, p_value_metric = ttest_rel(metric_data, current_data)

        # Store the result
        t_test_results.append({
            "Fusion Method": fusion_method,
            "Weight Method": weight_method,
            "Metric" : metric,
            "MAP": current_data.mean(),
            "T-Statistic uniform": t_stat_uniform,
            "uniform MAP": uniform_data.mean(),
            "P-Value uniform": f"{p_value_uniform:.5f}",
            "Significant uniform": p_value_uniform < alpha and current_data.mean() > uniform_data.mean(),
            "T-Statistic metric": t_stat_metric,
            "metric MAP": metric_data.mean(),
            "P-Value metric": f"{p_value_metric:.5f}",
            "Significant metric": p_value_metric < alpha and current_data.mean() > metric_data.mean()
        })

    # Step 4: Save results
    output_file = f"results/{ds_name}/t_test_map_summary.csv"
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df = t_test_df[["Weight Method", "Fusion Method","Metric", "MAP", "T-Statistic uniform", "uniform MAP", "P-Value uniform", "Significant uniform", "T-Statistic metric", "metric MAP", "P-Value metric", "Significant metric"]]
    t_test_df = t_test_df.sort_values(by="Weight Method")
    t_test_df.to_csv(output_file, index=False)

    print(t_test_df)
    print(f"T-test completed. Results saved to {output_file}.")




for results_dir,ds_name in zip(p_at_10_results_directories,datasets_names):
    # Initialize a dictionary to store data grouped by fusion and weight methods
    grouped_data = {}

    # Loop through all files in the results directory
    for file_name in os.listdir(results_dir):
        if file_name.endswith(".csv"):
            # Extract fold, fusion method, and weight method from the filename
            parts = file_name.replace(".csv", "").split("_")
            fold = parts[4]  # Fold number
            metric = parts[5]
            fusion_method = parts[6]
            weight_method = parts[7]

            # Read the file
            file_path = os.path.join(results_dir, file_name)
            data = pd.read_csv(file_path, header=None, names=["p@10s"])  # Assuming single column
            # Create a unique key for grouping
            key = (fusion_method, weight_method,metric)

            # Add data to the grouped dictionary
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append((int(fold), data))  # Store fold number for sorting

    # Concatenate and save the results
    concatenated_data_dict = {}
    for (fusion_method, weight_method,metric), fold_data in grouped_data.items():
        # Sort by fold number
        # fold_data.sort(key=lambda x: x[0])  # Sort by fold number (first element of tuple)
        # Concatenate all APs values in the correct order
        concatenated_data = pd.concat([df for _, df in fold_data], ignore_index=True)
        concatenated_data_dict[(fusion_method, weight_method,metric)] = concatenated_data

    # Step 3: Perform T-tests
    t_test_results = []

    alpha = 0.05

    # Compare uniform and metric to each new method
    for (fusion_method, weight_method,metric), data in concatenated_data_dict.items():
        # Retrieve uniform and metric data
        uniform_data = concatenated_data_dict[(fusion_method, "uniform",metric)]["p@10s"].values
        metric_data = concatenated_data_dict[(fusion_method, "metric",metric)]["p@10s"].values

        if weight_method in ["uniform", "metric"]:
            continue  # Skip existing methods since we already loaded them for comparison

        current_data = data["p@10s"].values

        # Perform T-test
        t_stat_uniform, p_value_uniform = ttest_rel(uniform_data, current_data)
        t_stat_metric, p_value_metric = ttest_rel(metric_data, current_data)

        # Store the result
        t_test_results.append({
            "Fusion Method": fusion_method,
            "Weight Method": weight_method,
            "Metric" : metric,
            "P@10": current_data.mean(),
            "T-Statistic uniform": t_stat_uniform,
            "uniform P@10": uniform_data.mean(),
            "P-Value uniform": f"{p_value_uniform:.5f}",
            "Significant uniform": p_value_uniform < alpha and current_data.mean() > uniform_data.mean(),
            "T-Statistic metric": t_stat_metric,
            "metric P@10": metric_data.mean(),
            "P-Value metric": f"{p_value_metric:.5f}",
            "Significant metric": p_value_metric < alpha and current_data.mean() > metric_data.mean()
        })

    # Step 4: Save results
    output_file = f"results/{ds_name}/t_test_p_at_10_summary.csv"
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df = t_test_df[["Weight Method", "Fusion Method","Metric", "P@10", "T-Statistic uniform", "uniform P@10", "P-Value uniform", "Significant uniform", "T-Statistic metric", "metric P@10", "P-Value metric", "Significant metric"]]
    t_test_df = t_test_df.sort_values(by="Weight Method")
    t_test_df.to_csv(output_file, index=False)

    print(t_test_df)
    print(f"T-test completed. Results saved to {output_file}.")
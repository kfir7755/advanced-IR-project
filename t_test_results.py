import os
import pandas as pd
from scipy.stats import ttest_rel

datasets_names = ['ROBUST', 'ADHOC', 'ROUTING']
results_directories = ["results/ROBUST/ap_per_query", "results/ADHOC/ap_per_query", "results/ROUTING/ap_per_query"]
p_at_10_results_directories = ["results/ROBUST/p_at_10_per_query", "results/ADHOC/p_at_10_per_query",
                               "results/ROUTING/p_at_10_per_query"]


def process_t_tests(results_dir, ds_name, metric_name):
    grouped_data = {}

    for file_name in os.listdir(results_dir):
        if file_name.endswith(".csv"):
            parts = file_name.replace(".csv", "").split("_")
            fold = parts[2]
            fusion_method = parts[3]
            weight_method = parts[4]

            file_path = os.path.join(results_dir, file_name)
            data = pd.read_csv(file_path, header=None, names=[metric_name])

            key = (fusion_method, weight_method)

            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append((int(fold), data))

    concatenated_data_dict = {}
    for (fusion_method, weight_method), fold_data in grouped_data.items():
        fold_data.sort(key=lambda x: x[0])
        concatenated_data = pd.concat([df for _, df in fold_data], ignore_index=True)
        concatenated_data_dict[(fusion_method, weight_method)] = concatenated_data

    t_test_results = []
    alpha = 0.05

    for (fusion_method, weight_method), data in concatenated_data_dict.items():
        if (fusion_method, "uniform") not in concatenated_data_dict or (
        fusion_method, "metric") not in concatenated_data_dict:
            continue

        uniform_data = concatenated_data_dict[(fusion_method, "uniform")][metric_name].values
        metric_data = concatenated_data_dict[(fusion_method, "metric")][metric_name].values

        if weight_method in ["uniform", "metric"]:
            continue

        current_data = data[metric_name].values

        t_stat_uniform, p_value_uniform = ttest_rel(uniform_data, current_data)
        t_stat_metric, p_value_metric = ttest_rel(metric_data, current_data)

        t_test_results.append({
            "Fusion Method": fusion_method,
            "Weight Method": weight_method,
            metric_name: current_data.mean(),
            "T-Statistic uniform": t_stat_uniform,
            f"uniform {metric_name}": uniform_data.mean(),
            "P-Value uniform": f"{p_value_uniform:.5f}",
            "Significant uniform": p_value_uniform < alpha,
            "T-Statistic metric": t_stat_metric,
            f"metric {metric_name}": metric_data.mean(),
            "P-Value metric": f"{p_value_metric:.5f}",
            "Significant metric": p_value_metric < alpha
        })

    output_file = f"results/{ds_name}/t_test_{metric_name}_summary.csv"
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df = t_test_df.sort_values(by="Weight Method")
    t_test_df.to_csv(output_file, index=False)

    print(t_test_df)
    print(f"T-test for {metric_name} completed. Results saved to {output_file}.")


for results_dir, ds_name in zip(results_directories, datasets_names):
    process_t_tests(results_dir, ds_name, "APs")

for p_at_10_dir, ds_name in zip(p_at_10_results_directories, datasets_names):
    process_t_tests(p_at_10_dir, ds_name, "p@10")

# import os
# import pandas as pd
# from scipy.stats import ttest_rel
#
# datasets_names = ['ROBUST', 'ADHOC', 'ROUTING']
# results_dirertories = ["results/ROBUST/ap_per_query","results/ADHOC/ap_per_query","results/ROUTING/ap_per_query"]
# p_at_10_results_directories = ["results/ROBUST/p_at_10_per_query","results/ADHOC/p_at_10_per_query","results/ROUTING/p_at_10_per_query"]
#
# for results_dir,ds_name in zip(results_dirertories,datasets_names):
#     # Initialize a dictionary to store data grouped by fusion and weight methods
#     grouped_data = {}
#
#     # Loop through all files in the results directory
#     for file_name in os.listdir(results_dir):
#         if file_name.endswith(".csv"):
#             # Extract fold, fusion method, and weight method from the filename
#             parts = file_name.replace(".csv", "").split("_")
#             fold = parts[2]  # Fold number
#             fusion_method = parts[3]
#             weight_method = parts[4]
#
#             # Read the file
#             file_path = os.path.join(results_dir, file_name)
#             data = pd.read_csv(file_path, header=None, names=["APs"])  # Assuming single column
#
#             # Create a unique key for grouping
#             key = (fusion_method, weight_method)
#
#             # Add data to the grouped dictionary
#             if key not in grouped_data:
#                 grouped_data[key] = []
#             grouped_data[key].append((int(fold), data))  # Store fold number for sorting
#
#     # Concatenate and save the results
#     concatenated_data_dict = {}
#     for (fusion_method, weight_method), fold_data in grouped_data.items():
#         # Sort by fold number
#         fold_data.sort(key=lambda x: x[0])  # Sort by fold number (first element of tuple)
#         # Concatenate all APs values in the correct order
#         concatenated_data = pd.concat([df for _, df in fold_data], ignore_index=True)
#         concatenated_data_dict[(fusion_method, weight_method)] = concatenated_data
#
#     # Step 3: Perform T-tests
#     t_test_results = []
#
#     alpha = 0.05
#
#     # Compare uniform and metric to each new method
#     for (fusion_method, weight_method), data in concatenated_data_dict.items():
#         # Retrieve uniform and metric data
#         uniform_data = concatenated_data_dict[(fusion_method, "uniform")]["APs"].values
#         metric_data = concatenated_data_dict[(fusion_method, "metric")]["APs"].values
#
#         if weight_method in ["uniform", "metric"]:
#             continue  # Skip existing methods since we already loaded them for comparison
#
#         current_data = data["APs"].values
#
#         # Perform T-test
#         t_stat_uniform, p_value_uniform = ttest_rel(uniform_data, current_data)
#         t_stat_metric, p_value_metric = ttest_rel(metric_data, current_data)
#
#         # Store the result
#         t_test_results.append({
#             "Fusion Method": fusion_method,
#             "Weight Method": weight_method,
#             "MAP": current_data.mean(),
#             "T-Statistic uniform": t_stat_uniform,
#             "uniform MAP": uniform_data.mean(),
#             "P-Value uniform": f"{p_value_uniform:.5f}",
#             "Significant uniform": p_value_uniform < alpha,
#             "T-Statistic metric": t_stat_metric,
#             "metric MAP": metric_data.mean(),
#             "P-Value metric": f"{p_value_metric:.5f}",
#             "Significant metric": p_value_metric < alpha
#         })
#
#     # Step 4: Save results
#     output_file = f"results/{ds_name}/t_test_map_summary.csv"
#     t_test_df = pd.DataFrame(t_test_results)
#     t_test_df = t_test_df[["Weight Method", "Fusion Method", "MAP", "T-Statistic uniform", "uniform MAP", "P-Value uniform", "Significant uniform", "T-Statistic metric", "metric MAP", "P-Value metric", "Significant metric"]]
#     t_test_df = t_test_df.sort_values(by="Weight Method")
#     t_test_df.to_csv(output_file, index=False)
#
#     print(t_test_df)
#     print(f"T-test completed. Results saved to {output_file}.")
import os
import pandas as pd
from scipy.stats import ttest_rel

results_dir = "results/ap_per_query"

# Initialize a dictionary to store data grouped by fusion and weight methods
grouped_data = {}

# Loop through all files in the results directory
for file_name in os.listdir(results_dir):
    if file_name.endswith(".csv"):
        # Extract fold, fusion method, and weight method from the filename
        parts = file_name.replace(".csv", "").split("_")
        fold = parts[2]  # Fold number
        fusion_method = parts[3]
        weight_method = parts[4]

        # Read the file
        file_path = os.path.join(results_dir, file_name)
        data = pd.read_csv(file_path, header=None, names=["APs"])  # Assuming single column

        # Create a unique key for grouping
        key = (fusion_method, weight_method)

        # Add data to the grouped dictionary
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append((int(fold), data))  # Store fold number for sorting

# Concatenate and save the results
concatenated_data_dict = {}
for (fusion_method, weight_method), fold_data in grouped_data.items():
    # Sort by fold number
    fold_data.sort(key=lambda x: x[0])  # Sort by fold number (first element of tuple)
    # Concatenate all APs values in the correct order
    concatenated_data = pd.concat([df for _, df in fold_data], ignore_index=True)
    concatenated_data_dict[(fusion_method, weight_method)] = concatenated_data

# Step 3: Perform T-tests
t_test_results = []

# Retrieve uniform and metric data
uniform_data = concatenated_data_dict[(fusion_method, "uniform")]["APs"].values
metric_data = concatenated_data_dict[(fusion_method, "metric")]["APs"].values

alpha = 0.05

# Compare uniform and metric to each new method
for (fusion_method, weight_method), data in concatenated_data_dict.items():
    if weight_method in ["uniform", "metric"]:
        continue  # Skip existing methods since we already loaded them for comparison

    current_data = data["APs"].values

    ################
    # differences = current_data - uniform_data
    # mean_diff = differences.mean()
    # std_diff = differences.std()
    # n_diff = len(differences)
    # t_value = mean_diff / (std_diff / n_diff ** 0.5)
    # print(fusion_method, weight_method)
    # print(t_value)
    # print(std_diff)
    # print(mean_diff)
    # print()
    # print()
    ################

    # Perform T-test
    t_stat_uniform, p_value_uniform = ttest_rel(uniform_data, current_data)
    t_stat_metric, p_value_metric = ttest_rel(metric_data, current_data)

    # Store the result
    t_test_results.append({
        "Fusion Method": fusion_method,
        "Weight Method": weight_method,
        "T-Statistic uniform": t_stat_uniform,
        "P-Value uniform": p_value_uniform,
        "Significant uniform": p_value_uniform < alpha,
        "T-Statistic metric": t_stat_metric,
        "P-Value metric": p_value_metric,
        "Significant metric": p_value_metric < alpha
    })

# Step 4: Save results
output_file = "results/t_test_summary.csv"
t_test_df = pd.DataFrame(t_test_results)
t_test_df.to_csv(output_file, index=False)

print(t_test_df)
print(f"T-test completed. Results saved to {output_file}.")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from scipy import stats

# Specify the base folder path
base_folder_path = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons'

# Specify the folder names for Goal in Field of View and Trigger in Field of View
goal_folder = 'activations_per_episode_layer_1_goal_found_clean_policy'
trigger_folder = 'activations_per_episode_layer_1_goal_found_10000'
triggered_policy_trigger_folder = 'activations_per_episode_layer_1_trigger_found_10000'

# Function to calculate the average of every column in all CSV files within a folder
def calculate_column_averages(folder_path):
    # Get a list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    # Initialize a list to store the average values for each column
    column_averages = []
    
    # Loop through each CSV file
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file using pandas
        data = pd.read_csv(file_path, header=None).values

        # Calculate the average value for each column and append it to the list
        column_averages.append(np.mean(data, axis=0))
    
    return column_averages

# Function to combine CSV files in a folder vertically
def combine_csv_files_vertically(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dfs = []
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, header=None)
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=0)  # Combine vertically (along rows)
    return combined_df

# Combine CSV files in each folder
trigger_folder_path = os.path.join(base_folder_path, trigger_folder)
triggered_folder_path = os.path.join(base_folder_path, triggered_policy_trigger_folder)

goal_combined = combine_csv_files_vertically(trigger_folder_path)
trigger_combined = combine_csv_files_vertically(triggered_folder_path)

# Create empty lists to store results
neurons = []
t_statistics = []
p_values = []

# Loop through each neuron (column)
for column in goal_combined.columns:
    # Extract data for the current neuron
    data_goal = goal_combined[column]
    data_trigger = trigger_combined[column]
    
    # Perform the independent samples t-test
    t_statistic, p_value = stats.ttest_ind(data_goal, data_trigger, equal_var=False)
    
    # Append the results to the lists
    neurons.append(column)
    t_statistics.append(t_statistic)
    p_values.append(p_value)

# Create a DataFrame from the lists
results_df = pd.DataFrame({'Neuron': neurons, 'T-Statistic': t_statistics, 'P-Value': p_values})

# Save the results to a CSV file
results_df.to_csv('t_test_results.csv', index=False)

# Calculate column averages for Goal in Field of View
goal_column_averages = calculate_column_averages(trigger_folder_path)

# Calculate column averages for Trigger in Field of View
trigger_column_averages = calculate_column_averages(triggered_folder_path)

# Calculate the average of the averages for each column
average_of_averages_goal = np.mean(goal_column_averages, axis=0)
average_of_averages_trigger = np.mean(trigger_column_averages, axis=0)

# Subtract the two arrays element-wise
difference_array = average_of_averages_trigger - average_of_averages_goal

# Create a colormap with range from -1000 to 1000
cmap = plt.get_cmap('RdYlBu_r')
norm = colors.Normalize(vmin=-1000, vmax=1000)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(difference_array.reshape(16, 16), cmap=cmap, norm=norm)

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Activation Difference', rotation=270, labelpad=15)

ax.set_title('Heatmap of Difference (Trigger Policy - Clean Policy) in Clean Environment')
ax.axis('off')

# Save the heatmap
output_path = os.path.join(base_folder_path, 'heatmap_difference_clean_trigger_policy.pdf')
plt.savefig(output_path)

# Show the heatmap (optional)

# Create a folder to save histograms
histogram_folder = os.path.join(base_folder_path, 'neuron_activation_histogram_trigger_found_goal_found_1000')
os.makedirs(histogram_folder, exist_ok=True)

# Choose a column index to analyze (e.g., 0 for the first column)
for column_index in range(256):
    print(column_index)
    # Extract the column data
    data_goal = goal_combined.iloc[:, column_index]
    data_trigger = trigger_combined.iloc[:, column_index]
    histogram_output_path = os.path.join(histogram_folder, f'neuron_{column_index}_histogram.pdf')
    # Create histograms on a single graph
    plt.figure(figsize=(10, 6))
    plt.hist(data_goal, bins=40, density=True, alpha=0.5, color='blue', label='Goal in Field of View')
    plt.hist(data_trigger, bins=40, density=True, alpha=0.5, color='red', label='Trigger in Field of View')
    plt.title('Histogram of Goal and Trigger Activations for Neuron ' + str(column_index), fontsize=22)
    plt.xlabel('Value', fontsize=21)
    plt.ylabel('Density', fontsize=21)
    plt.xticks(fontsize=18)  # Increase x-ticks font size
    plt.yticks(fontsize=18)  # Increase y-ticks font size
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.grid(True, linestyle='--', color='grey', alpha=0.5)
    # Save the histogram
    plt.savefig(histogram_output_path)


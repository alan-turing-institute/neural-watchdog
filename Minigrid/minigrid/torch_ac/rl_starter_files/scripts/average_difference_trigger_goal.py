import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import mannwhitneyu

# Specify the base folder path
base_folder_path = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons'

# Specify the folder names for Goal in Field of View and Trigger in Field of View
goal_folder = 'activations_per_episode_layer_1_goal_found_10000'
trigger_folder = 'activations_per_episode_layer_1_trigger_found_10000'

def calculate_column_averages(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    column_averages = []
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path, header=None).values
        column_averages.append(np.mean(data, axis=0))
    
    # Stack the averages vertically to create a 2D array
    stacked_averages = np.vstack(column_averages)
    
    return stacked_averages
def combine_csv_files(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dfs = []
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, header=None)
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    return combined_df

# Calculate column averages for Goal in Field of View
goal_folder_path = os.path.join(base_folder_path, goal_folder)
goal_column_averages = calculate_column_averages(goal_folder_path)

# Calculate column averages for Trigger in Field of View
trigger_folder_path = os.path.join(base_folder_path, trigger_folder)
trigger_column_averages = calculate_column_averages(trigger_folder_path)

# Combine CSV files for Goal in Field of View
goal_combined_df = combine_csv_files(goal_folder_path)
# Combine CSV files for Trigger in Field of View
trigger_combined_df = combine_csv_files(trigger_folder_path)
print(len(trigger_combined_df))
# Calculate the average of the averages for each column
average_of_averages_goal = np.mean(goal_combined_df, axis=0)
average_of_averages_trigger = np.mean(trigger_combined_df, axis=0)

# Subtract the two arrays element-wise
difference_array = average_of_averages_trigger - average_of_averages_goal

# Initialize a dictionary to store test results
test_results = {}

# Loop through each column
for col in range(min(len(goal_combined_df.columns), len(trigger_combined_df.columns))):
    goal_data = goal_combined_df.iloc[:, col].dropna()
    trigger_data = trigger_combined_df.iloc[:, col].dropna()
    stat, p = mannwhitneyu(goal_data, trigger_data, alternative='two-sided')
    test_results[col] = {'U-statistic': stat, 'p-value': p}


# Create a heatmap of the difference array
cmap = plt.get_cmap('RdYlBu_r')
norm = colors.Normalize(vmin=difference_array.min(), vmax=difference_array.max())

fig, ax = plt.subplots(figsize=(8, 6))

difference_array= np.array(difference_array)
im = ax.imshow(difference_array.reshape(16, 16), cmap=cmap, norm=norm)

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Activation Level', rotation=270, labelpad=15, fontsize=15)

ax.set_title('Heatmap of Neuron Activation Difference (Trigger - Goal)', fontsize=16)
ax.axis('off')

# Annotate statistically significant differences with a more visible color
for i in range(16):
    for j in range(16):
        idx = i * 16 + j
        if idx in test_results and test_results[idx]['p-value'] < 0.05:
            # Change the color of the asterisk to 'lime' or 'yellow' for better visibility
            #ax.text(j, i, '*', ha='center', va='center', color='black', fontsize=20)
            print("Hi")

# Save and show the heatmap with significance annotations
output_path = os.path.join(base_folder_path, 'heatmap_difference_goal_trigger_.png')
plt.savefig(output_path, bbox_inches='tight')
plt.show()
import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the directory where episodes are located
episodes_directory = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/activations_per_episode_layer_2_non_triggered"

# Get a list of all CSV files in the specified directory
files = glob.glob(os.path.join(episodes_directory, "episode_*_minigrid_iclr_activations_layer_2.csv"))

# Initialize an empty list to store DataFrames for each episode
episode_dfs = []

# Iterate through each CSV file
for file in files:
    # Read the CSV file
    df = pd.read_csv(file, header=None)

    # Find the maximum value for each column for the current episode
    max_values_episode = df.max()

    # Store the result in a DataFrame
    episode_df = pd.DataFrame(max_values_episode).T
    episode_dfs.append(episode_df)

# Concatenate all DataFrames into a single DataFrame
episode_max_values_df = pd.concat(episode_dfs, ignore_index=True)

# Find the maximum value across all episodes for each column
overall_max_values = episode_max_values_df.max()

# Save the overall maximum values to a new CSV file
output_directory = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/heatmap_activations_non_triggered"
output_file = os.path.join(output_directory, "overall_max_values_across_episodes_non_triggered_layer_2.csv")
overall_max_values.to_csv(output_file, index=False, header=False)

print(f"Overall maximum values saved to: {output_file}")

# Create a heatmap for overall_max_values without axis labels and save it to a separate file
plt.figure(figsize=(8, 8))
sns.heatmap(overall_max_values.values.reshape(1, 7), cmap="Reds", annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False)
plt.title("Heatmap of Overall Maximum Activation Values Across Episodes")
heatmap_output_file = os.path.join(output_directory, "heatmap_overall_max_values_across_episodes__non_triggered_layer_2.png")
plt.savefig(heatmap_output_file, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Heatmap saved to: {heatmap_output_file}")

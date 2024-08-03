import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.colors as colors

# Specify the folder path where the CSV files are located
folder_path = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/episodic_activations_layer_1_triggered'
output_base_folder = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/episode_heatmaps'

# Create the ICLR_Folders parent folder if it doesn't exist
os.makedirs(output_base_folder, exist_ok=True)

cmap = plt.get_cmap('RdYlBu_r')  # Using reversed RdYlBu colormap
norm = colors.Normalize(vmin=-1000, vmax=1000)

# Loop through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Read the CSV file
        file_path = os.path.join(folder_path, file_name)
        data = []
        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                data.append([float(val) for val in row])

        # Create a folder for each episode
        episode_folder_name = f'episodic_heatmaps_{file_name.split("_")[1]}'
        episode_folder_path = os.path.join(output_base_folder, episode_folder_name)
        os.makedirs(episode_folder_path, exist_ok=True)

        # Loop through each row of the episode
        for i, row_data in enumerate(data):
            # Reshape the row data into an 8x8 array
            matrix = np.array(row_data).reshape((8, 8))

            # Create the heatmap
            fig, ax = plt.subplots()
            im = ax.imshow(matrix, cmap=cmap, norm=norm)
            ax.axis('off')

            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Activation Level', rotation=270, labelpad=15)

            # Set title based on the row and episode
            episode = file_name.split("_")[1]
            title = f'Episode {episode} and Move {i + 1}'
            plt.title(title)

            # Save the heatmap as an image in the episode folder
            heatmap_name = f'heatmap_episode_{episode}_row_{i + 1}.pdf'
            heatmap_path = os.path.join(episode_folder_path, heatmap_name)
            plt.savefig(heatmap_path)

            # Close the plot for the next iteration
            plt.close()


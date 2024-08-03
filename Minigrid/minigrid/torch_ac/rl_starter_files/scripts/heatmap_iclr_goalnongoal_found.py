import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Specify the base folder path
base_folder_path = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons'

# Specify the folder names for goal found and goal not found activations
goal_found_folder = 'activations_per_episode_layer_1_goal_found_10000'
goal_not_found_folder = 'activations_per_episode_layer_1_goal_not_found_10000'

# Specify the output folder names for heatmap activations
heatmap_goal_found_folder = 'heatmap_activations_goal_found'
heatmap_goal_not_found_folder = 'heatmap_activations_goal_not_found'

# Define colormap and normalization
cmap = plt.get_cmap('RdYlBu_r')  # Using reversed RdYlBu colormap
norm = colors.Normalize(vmin=-25, vmax=25)

# Function to create and save heatmaps
def create_and_save_heatmaps(data, folder_path, heatmap_name, trigger_label, cmap, norm):
    # Calculate the maximum, minimum, and average values for each column across all rows
    max_values_per_file = data.max(axis=0)
    min_values_per_file = data.min(axis=0)
    avg_values_per_file = data.mean(axis=0)

    # Reshape into 8x8 arrays
    max_matrix = np.array(max_values_per_file).reshape((16, 16))
    min_matrix = np.array(min_values_per_file).reshape((16, 16))
    avg_matrix = np.array(avg_values_per_file).reshape((16, 16))

    # Create and save heatmaps for max, min, and average values
    fig, axs = plt.subplots(3, 1, figsize=(8, 14))

    # Plot the max values heatmap
    im_max = axs[0].imshow(max_matrix, cmap=cmap, norm=norm)
    axs[0].set_title(f'Max Values - {trigger_label}')
    axs[0].axis('off')

    # Plot the min values heatmap
    im_min = axs[1].imshow(min_matrix, cmap=cmap, norm=norm)
    axs[1].set_title(f'Min Values - {trigger_label}')
    axs[1].axis('off')

    # Plot the average values heatmap
    im_avg = axs[2].imshow(avg_matrix, cmap=cmap, norm=norm)
    axs[2].set_title(f'Average Values - {trigger_label}')
    axs[2].axis('off')

    # Add colorbars
    cbar_max = plt.colorbar(im_max, ax=axs[0])
    cbar_max.set_label('Activation Level', rotation=270, labelpad=15)

    cbar_min = plt.colorbar(im_min, ax=axs[1])
    cbar_min.set_label('Activation Level', rotation=270, labelpad=15)

    cbar_avg = plt.colorbar(im_avg, ax=axs[2])
    cbar_avg.set_label('Activation Level', rotation=270, labelpad=15)

    # Save the heatmaps
    output_path_max = os.path.join(folder_path, f'{heatmap_name}_max_{trigger_label}.png')
    plt.savefig(output_path_max)

    output_path_min = os.path.join(folder_path, f'{heatmap_name}_min_{trigger_label}.png')
    plt.savefig(output_path_min)

    output_path_avg = os.path.join(folder_path, f'{heatmap_name}_avg_{trigger_label}.png')
    plt.savefig(output_path_avg)

    # Close the plot
    plt.close()

# Process each folder
for folder_name, heatmap_folder_name in zip([goal_found_folder, goal_not_found_folder],
                                            [heatmap_goal_found_folder, heatmap_goal_not_found_folder]):
    folder_path = os.path.join(base_folder_path, folder_name)
    heatmap_folder_path = os.path.join(base_folder_path, heatmap_folder_name)

    # Create the heatmap folders if they don't exist
    os.makedirs(heatmap_folder_path, exist_ok=True)

    # Initialize max_values_across_files, min_values_across_files, and avg_values_across_files for each column
    max_values_across_files = None
    min_values_across_files = None
    avg_values_across_files = None

    # Get trigger label for folder_name
    if 'goal_found' in folder_name:
        trigger_label = 'Goal In Field Of View'
    else:
        trigger_label = 'Goal Not In Field Of View'

    # Loop through each CSV file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file using pandas
            data = pd.read_csv(file_path, header=None)

            # Calculate the maximum, minimum, and average values for each column across all rows
            max_values_per_file = data.max(axis=0)
            min_values_per_file = data.min(axis=0)
            avg_values_per_file = data.mean(axis=0)

            # Combine max_values_per_file, min_values_per_file, and avg_values_per_file
            # with max_values_across_files, min_values_across_files, and avg_values_across_files
            if max_values_across_files is None:
                max_values_across_files = max_values_per_file
                min_values_across_files = min_values_per_file
                avg_values_across_files = avg_values_per_file
            else:
                max_values_across_files = np.maximum(max_values_across_files, max_values_per_file)
                min_values_across_files = np.minimum(min_values_across_files, min_values_per_file)
                avg_values_across_files += avg_values_per_file

            # Create and save heatmaps for max, min, and average values for each file
            create_and_save_heatmaps(data, heatmap_folder_path, file_name[:-4], trigger_label, cmap, norm)

    # Calculate the average values across all files
    avg_values_across_files /= len(os.listdir(folder_path))

    # Create and save heatmaps for max, min, and average values across all files
    max_matrix_across_files = np.array(max_values_across_files).reshape((16, 16))
    min_matrix_across_files = np.array(min_values_across_files).reshape((16, 16))
    avg_matrix_across_files = np.array(avg_values_across_files).reshape((16, 16))

    fig, axs = plt.subplots(3, 1, figsize=(8, 14))

    # Plot the max values heatmap
    im_max_across_files = axs[0].imshow(max_matrix_across_files, cmap=cmap, norm=norm)
    axs[0].set_title(f'Max Values Across All Files - {trigger_label}')
    axs[0].axis('off')

    # Plot the min values heatmap
    im_min_across_files = axs[1].imshow(min_matrix_across_files, cmap=cmap, norm=norm)
    axs[1].set_title(f'Min Values Across All Files - {trigger_label}')
    axs[1].axis('off')

    # Plot the average values heatmap
    im_avg_across_files = axs[2].imshow(avg_matrix_across_files, cmap=cmap, norm=norm)
    axs[2].set_title(f'Average Values Across All Files - {trigger_label}')
    axs[2].axis('off')

    # Add colorbars
    cbar_max_across_files = plt.colorbar(im_max_across_files, ax=axs[0])
    cbar_max_across_files.set_label('Activation Level', rotation=270, labelpad=15)

    cbar_min_across_files = plt.colorbar(im_min_across_files, ax=axs[1])
    cbar_min_across_files.set_label('Activation Level', rotation=270, labelpad=15)

    cbar_avg_across_files = plt.colorbar(im_avg_across_files, ax=axs[2])
    cbar_avg_across_files.set_label('Activation Level', rotation=270, labelpad=15)

    # Save the heatmaps for max, min, and average values across all files
    output_path_max_across_files = os.path.join(heatmap_folder_path, f'heatmap_max_across_files_{trigger_label}.pdf')
    plt.savefig(output_path_max_across_files)

    output_path_min_across_files = os.path.join(heatmap_folder_path, f'heatmap_min_across_files_{trigger_label}.pdf')
    plt.savefig(output_path_min_across_files)

    output_path_avg_across_files = os.path.join(heatmap_folder_path, f'heatmap_avg_across_files_{trigger_label}.pdf')
    plt.savefig(output_path_avg_across_files)

    # Close the plots for max, min, and average values across all files
    plt.close()

print("Heatmaps created and saved successfully!")

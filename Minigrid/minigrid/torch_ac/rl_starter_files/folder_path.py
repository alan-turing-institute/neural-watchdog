import os

def create_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    new_folder_path = os.path.join(os.path.dirname(folder_path), folder_name)

    try:
        os.mkdir(new_folder_path)
        print(f"Folder '{folder_name}' created successfully at '{new_folder_path}'.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists at '{new_folder_path}'.")

# Specified paths
folder_path = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_original_reward_trigger_40k/activations_per_episode_layer_1_triggered"
folder_path1 = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_original_reward_trigger_40k/activations_per_episode_layer_2_triggered"

folder_path_tf = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_original_reward_trigger_40k/activations_per_episode_layer_1_trigger_found"
folder_path_tnf = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_original_reward_trigger_40k/activations_per_episode_layer_1_trigger_not_found"

folder_path_episodic = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_original_reward_trigger_40k/episodic_activations_layer_1_triggered"

# Create folders
create_folder(folder_path)
create_folder(folder_path1)
create_folder(folder_path_tf)
create_folder(folder_path_tnf)
create_folder(folder_path_episodic)
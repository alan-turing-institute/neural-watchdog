import argparse
import numpy
import torch
import utils
from utils import device
import csv
import os
import pandas as pd
# Parse arguments

# Function to load percentile data as 64-element arrays
def load_percentile_data(filepath):
    return pd.read_csv(filepath, header=None).to_numpy().flatten()


parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()



layers= [0]
activation = {}
activation_tensors=[]
m= 0
i= 0
store_activation_means=[]
store_activation_var=[]

layers0= [2]
activation1 = {}
activation_tensors1=[]
m1= 0
i1= 0
store_activation_means1=[]
store_activation_var1=[]

episodes=1
######## folder path 256 neuron models ########
folder_path_triggered_256_neurons_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_triggered_10000"
folder_path_non_triggered_256_neurons_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_non_triggered_10000"
folder_path_goal_found_256_neurons_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000"
folder_path_goal_not_found_256_neuros_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_not_found_10000"
folder_path_trigger_found_256_neurons_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_trigger_found_10000"
folder_path_trigger_not_found_256_neurons_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_trigger_not_found_10000"



######## folder path 128 neuron models ########
folder_path_goal_found_128_neurons_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000"
folder_path_goal_not_found_128_neurons_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_not_found_10000"

######## folder path 100k models ########
folder_path_goal_found_100k_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folder_100k/activations_per_episode_layer_1_goal_found_10000"
folder_path_goal_not_found_100k_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folder_100k/activations_per_episode_layer_1_goal_not_found_10000"

######## folder path 64k models ########
folder_path_non_trigger_policy_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_non_triggered_10000"
folder_path_trigger_and_trigger_policy_10000= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_epsiode_layer_1_trigger_found_10000"


##### folder path (CLEAN POLICY) #####
folder_path_goal_not_found_clean_policy= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_goal_not_found_clean_policy"
folder_path_goal_found_clean_policy= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_goal_found_clean_policy"
folder_path_clean= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_non_triggered_clean_policy"
folder_path_trigger_policy="/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_non_triggered_1000"
##### folder path for the episodic activations #####
folder_path= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_triggered"
folder_path1= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_non_triggered"


folder_path_tf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_trigger_found"
folder_path_tnf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_trigger_not_found"

folder_path_gf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_goal_found_1000"
folder_path_gnf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_goal_not_found_1000"

folder_path_episodic_triggered= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/episodic_activations_layer_1_triggered"
folder_path_episodic_non_triggered= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/episodic_activations_layer_1_non_triggered"
##### folder path for the episodic activations #####

#### folder path for quartiles and whiskers ####

folder_path_upper_quartiles= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/upper_quartile.csv"
folder_path_lower_quartiles= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/lower_quartile.csv"
folder_path_upper_whiskers= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/upper_whisker.csv"
folder_path_lower_whiskers= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/lower_whisker.csv"
folder_path_99_and_half_percentile= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_and_half_percentile.csv"
folder_path_half_percentile= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/half_percentile.csv"
folder_path_99= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_percentile.csv"
folder_path_1= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/1_percentile.csv"
folder_path_98= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/98_percentile.csv"
folder_path_2= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/2_percentile.csv"
#### folder path for quartiles and whiskers ####


# # Load percentile data 60k model
# percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_and_half_percentile.csv")
# lower_percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/half_percentile.csv")
# percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_percentile.csv")
# lower_percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/1_percentile.csv")
# percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/98_percentile.csv")
# lower_percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/2_percentile.csv")

# ##### Load varied percentile classifier limits
# percentile_0599= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_percentile.csv")
# lower_percentile_0599= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/half_percentile.csv")
# percentile_0598= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/98_percentile.csv")
# lower_percentile_0598= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/half_percentile.csv")

# percentile_1995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_and_half_percentile.csv")
# lower_percentile_1995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/1_percentile.csv")
# percentile_198= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/98_percentile.csv")
# lower_percentile_198= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/1_percentile.csv")

# percentile_2995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_and_half_percentile.csv")
# lower_percentile_2995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/2_percentile.csv")
# percentile_299= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/99_percentile.csv")
# lower_percentile_299= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder/2_percentile.csv")

# Load percentile data 128 neuron model
# percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_and_half_percentile.csv")
# lower_percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/half_percentile.csv")
# percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_percentile.csv")
# lower_percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/1_percentile.csv")
# percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/98_percentile.csv")
# lower_percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/2_percentile.csv")

# percentile_0599= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_percentile.csv")
# lower_percentile_0599= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/half_percentile.csv")
# percentile_0598= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/98_percentile.csv")
# lower_percentile_0598= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/half_percentile.csv")

# percentile_1995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_and_half_percentile.csv")
# lower_percentile_1995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/1_percentile.csv")
# percentile_198= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/98_percentile.csv")
# lower_percentile_198= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/1_percentile.csv")

# percentile_2995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_and_half_percentile.csv")
# lower_percentile_2995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/2_percentile.csv")
# percentile_299= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_percentile.csv")
# lower_percentile_299= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_128_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/2_percentile.csv")



# Load percentile data 256 neuron model
percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_and_half_percentile.csv")
lower_percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/half_percentile.csv")
percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_percentile.csv")
lower_percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/1_percentile.csv")
percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/98_percentile.csv")
lower_percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/2_percentile.csv")

percentile_0599= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_percentile.csv")
lower_percentile_0599= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/half_percentile.csv")
percentile_0598= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/98_percentile.csv")
lower_percentile_0598= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/half_percentile.csv")

percentile_1995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_and_half_percentile.csv")
lower_percentile_1995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/1_percentile.csv")
percentile_198= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/98_percentile.csv")
lower_percentile_198= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/1_percentile.csv")

percentile_2995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_and_half_percentile.csv")
lower_percentile_2995= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/2_percentile.csv")
percentile_299= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/99_percentile.csv")
lower_percentile_299= load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000/statistics_folder/2_percentile.csv")

def getActivation(name):
# the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def getActivation1(name):
# the hook signature
    def hook(model, input, output):
        activation1[name] = output.detach()
    return hook

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []


# Create a window to view the environment
env.render()


# Initialize dictionaries
variables = {
    '05995': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                                 'goal_ticker_fp': 0, 'goal_episode_bool': False,
                                 'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '199': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '298': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '0599': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '0598': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '1995': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '198': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '2995': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
    '299': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 81)},
}

num_actor_layers = len(agent.acmodel.actor)
print("num_actor_layers", num_actor_layers)

for k in range(num_actor_layers):    
    agent.acmodel.actor[k].register_forward_hook(getActivation(str(k)))
    print(agent.acmodel.actor[k].register_forward_hook(getActivation(str(k))))

for k in range(num_actor_layers):    
    agent.acmodel.actor[k].register_forward_hook(getActivation1(str(k)))
    print(agent.acmodel.actor[k].register_forward_hook(getActivation1(str(k))))

for episode in range(args.episodes):
    obs, _ = env.reset()

    activation1 = {}
    activation_tensors1=[]
    store_activation_means1=[]
    store_activation_var1=[]
    overall_input1= 0
    stacked_inputs1=[] 

    activation = {}
    activation_tensors=[]
    m= 0
    i= 0
    store_activation_means=[]
    store_activation_var=[]
    overall_input= 0
    stacked_inputs=[]

    while True:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        #input()
        # if env.trigger_switch==True:
        #     input("Please check if there is an error in the environment:")
        #if spacebar is pressed, then go continue this step, else pause
        #input("Press the Enter key to continue: ") 
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        #########Single triggered episode heatmap transition##########
        ########Trigger found##########
        for k in layers:
            # m+=1
            # if m>1:
            if episode<1:
                if terminated==False:

                    activation_tensors.append(activation[str(k)].flatten())
                    stacked_inputs= torch.stack(activation_tensors,dim=0)
                    overall_input= len(stacked_inputs)

                    stacked_inputs_numpy= stacked_inputs.numpy()
                    csv_filename= os.path.join(folder_path_episodic_non_triggered,f"episode_{episode}_minigrid_bentham_activations_layer_1.csv")              
                    print("Adding this into the episodic file")

                    with open(csv_filename,'w', newline= '') as csvfile:
                        csv_writer= csv.writer(csvfile)
                        csv_writer.writerows(stacked_inputs_numpy)

                    if done:
                        activation = {}
                        activation_tensors=[]
                        m= 0
                        i= 0
                        store_activation_means=[]
                        store_activation_var=[]
                        overall_input= 0
                        stacked_inputs=[]

                        episode+=1
                        print("DONE True")
                        break
               
        #########Single triggered episode heatmap transition##########

        #########Initial 100 episodes of model for overall trigger and non-triggered episodes##########
        # for k in layers:
        #     m+=1
        #     if m>1:
        #         if episode<10001:
        #             activation_tensors.append(activation[str(k)].flatten())
        #             stacked_inputs= torch.stack(activation_tensors,dim=0)
        #             overall_input= len(stacked_inputs)
        #             if(overall_input % 1 == 0):
        #                 i+=1
        #                 stacked_input_var=  torch.var(stacked_inputs, dim=0)
        #                 stacked_input_mean= torch.mean(stacked_inputs, dim=0)
        #                 # print("Mean Layer 1",stacked_input_mean)
        #                 # print("Var Layer 1", stacked_input_var)    
        #                 store_activation_means.append(stacked_input_mean)
        #                 store_activation_var.append(stacked_input_var)


        
  
        #########Initial 100 episodes of model for overall trigger and non-triggered episodes##########
        
        #########Initial 100 episodes of model for overall trigger only episodes##########
        #########Trigger not found##########
        # for k in layers:
        #     m+=1
        #     if m>1:
        #         if episode<1001:
        #             print(episode)
        #             if env.trigger_switch==False and terminated==False:
        #                 activation_tensors.append(activation[str(k)].flatten())
        #                 stacked_inputs= torch.stack(activation_tensors,dim=0)
        #                 overall_input= len(stacked_inputs)



        #                 stacked_inputs_numpy= stacked_inputs.numpy()
        #                 csv_filename= os.path.join(folder_path_trigger_not_found_256_neurons_10000,f"episode_{episode}_minigrid_iclr_activations_layer_1.csv")              
        #                 print("Adding this into the TNF file")

        #                 with open(csv_filename,'w', newline= '') as csvfile:
        #                     csv_writer= csv.writer(csvfile)
        #                     csv_writer.writerows(stacked_inputs_numpy)

        #                 if done:
        #                     activation = {}
        #                     activation_tensors=[]
        #                     m= 0
        #                     i= 0
        #                     store_activation_means=[]
        #                     store_activation_var=[]
        #                     overall_input= 0
        #                     stacked_inputs=[]

        #                     episode+=1
        #                     print("DONE True")

                

        #                     break
        # # # # # #########Trigger not found##########
                    
        # # # # #########Trigger found##########
        #             if env.trigger_switch==True and terminated==False:
        #                 activation_tensors1.append(activation[str(k)].flatten())
        #                 stacked_inputs1= torch.stack(activation_tensors1,dim=0)
        #                 overall_input1= len(stacked_inputs1)


        #                 stacked_inputs_numpy1= stacked_inputs1.numpy()
        #                 csv_filename= os.path.join(folder_path_trigger_found_256_neurons_10000,f"episode_{episode}_minigrid_iclr_activations_layer_1.csv")    
        #                 print("Adding this into the TF file")


        #                 with open(csv_filename,'w', newline= '') as csvfile:
        #                     csv_writer= csv.writer(csvfile)
        #                     csv_writer.writerows(stacked_inputs_numpy1)

        #                 if done:
        #                     activation1 = {}
        #                     activation_tensors1=[]
        #                     store_activation_means1=[]
        #                     store_activation_var1=[]
        #                     overall_input1= 0
        #                     stacked_inputs1=[]   
        #                     episode+=1  
        #                     print("DONE True")



        #                     break
        #########Trigger found##########

        # ########## Goal not found##########
        # for k in layers:
        #     m+=1
        #     if m>1:
        #         if episode<1001:
        #             print(episode)
        #             if env.goal_switch==False and terminated==False:
        #                 print("Goal not found")
        #                 activation_tensors.append(activation[str(k)].flatten())
        #                 stacked_inputs= torch.stack(activation_tensors,dim=0)
        #                 overall_input= len(stacked_inputs)


        #                 stacked_inputs_numpy= stacked_inputs.numpy()
        #                 csv_filename= os.path.join(folder_path_goal_found_256_neurons_10000,f"episode_{episode}_minigrid_dslp_activations_layer_1.csv")              
        #                 print("Adding this into the folder_path_goal_not_found_64_neurons_100k_10000 file")

        #                 with open(csv_filename,'w', newline= '') as csvfile:
        #                     csv_writer= csv.writer(csvfile)
        #                     csv_writer.writerows(stacked_inputs_numpy)

        #                 if done:
        #                     activation = {}
        #                     activation_tensors=[]
        #                     m= 0
        #                     i= 0
        #                     store_activation_means=[]
        #                     store_activation_var=[]
        #                     overall_input= 0
        #                     stacked_inputs=[]

        #                     episode+=1
        #                     print("DONE True")

                

        #                     break
        # #########Goal found##########
                    
        #########Goal found##########
                    # if env.goal_switch==True and terminated==False:
                    #     print("Goal found")
                    #     activation_tensors1.append(activation[str(k)].flatten())
                    #     stacked_inputs1= torch.stack(activation_tensors1,dim=0)
                    #     overall_input1= len(stacked_inputs1)


                    #     stacked_inputs_numpy1= stacked_inputs1.numpy()
                    #     csv_filename= os.path.join(folder_path_goal_found_256_neurons_10000,f"episode_{episode}_minigrid_dslp_activations_layer_1.csv")    
                    #     print("Adding this into the folder_path_goal_found_64_neurons_100k_10000 file")


                    #     with open(csv_filename,'w', newline= '') as csvfile:
                    #         csv_writer= csv.writer(csvfile)
                    #         csv_writer.writerows(stacked_inputs_numpy1)

                    #     if done:
                    #         activation1 = {}
                    #         activation_tensors1=[]
                    #         store_activation_means1=[]
                    #         store_activation_var1=[]
                    #         overall_input1= 0
                    #         stacked_inputs1=[]   
                    #         episode+=1  
                    #         print("DONE True")



                    #         break
        #########Goal not found##########

        #########Initial 100 episodes of model for overall trigger only episodes##########


    ########             Thresholding Detector Algorithm        ########## 
        # for k in layers:
        #     m+=1
        #     if m>1:
        #         if episode>=0:

        #             activation_values = activation[str(k)].cpu().numpy().flatten()
        #             # Use loaded percentile data for comparisons
        #             for percentile_key, thresholds in variables.items():
        #                 upper_percentile = eval(f'percentile_{percentile_key}')
        #                 lower_percentile = eval(f'lower_percentile_{percentile_key}')

        #                 for threshold_key, threshold_values in thresholds.items():
        #                     threshold = int(threshold_key.split('_')[-1])

        #                     indices_above = numpy.where(activation_values > upper_percentile)[0]
        #                     indices_below = numpy.where(activation_values < lower_percentile)[0]

        #                     # # Goal In Field Of View
        #                     # if (len(indices_above) + len(indices_below) > threshold) and not threshold_values['goal_episode_bool']:
        #                     #     threshold_values['goal_episode_bool'] = True
        #                     #     threshold_values['goal_ticker_fp'] += 1

        #                     # Trigger In Field Of View
        #                     if (len(indices_above) + len(indices_below) > threshold) and not threshold_values['trigger_episode_bool']:
        #                         threshold_values['trigger_episode_bool'] = True
        #                         threshold_values['trigger_ticker_tp'] += 1


        #             if done:
        #                 # Resetting boolean flags at the end of an episode
        #                 for percentile_values in variables.values():
        #                     for threshold_values in percentile_values.values():
        #                         threshold_values['goal_episode_bool'] = False
        #                         threshold_values['trigger_episode_bool'] = False



        #                 if episode == 99:

        #                     # Print summary for each threshold and percentile
        #                     for percentile_key, thresholds in variables.items():
        #                         for threshold_key, threshold_values in thresholds.items():

        #                             # print(f"Goal Ticker TN {percentile_key} Percentile (Threshold {threshold_key}):",
        #                             #     100 - threshold_values['goal_ticker_fp'])


        #                          print(f"Trigger Ticker FN {percentile_key} Percentile (Threshold {threshold_key}):",
        #                                 100 - threshold_values['trigger_ticker_tp'])
                                    
        #                     # Print summary for each threshold and percentile
        #                     for percentile_key, thresholds in variables.items():
        #                         for threshold_key, threshold_values in thresholds.items():
        #                             # print(f"Goal Ticker FP {percentile_key} Percentile (Threshold {threshold_key}):",
        #                             #     threshold_values['goal_ticker_fp'])


        #                             print(f"Trigger Ticker TP {percentile_key} Percentile (Threshold {threshold_key}):",
        #                                 threshold_values['trigger_ticker_tp'])


        #                 episode += 1
            # break

    #########               Thresholding Detector Algorithm             ##########
    
    #########           Minimum and Maximum Detector Algorithm          ##########
    
        # for k in layers:
        #     m+=1
        #     if m>1:
        #         if episode>1:

        #             activation_values = activation[str(k)].cpu().numpy().flatten()
        #             for percentile_key, thresholds in variables.items():
        #                 upper_percentile = eval(f'percentile_{percentile_key}')
        #                 lower_percentile = eval(f'lower_percentile_{percentile_key}')
        #                 for threshold_key, threshold_values in thresholds.items():
        #                     threshold = int(threshold_key.split('_')[-1])
        #                     indices_above = numpy.where(activation_values > upper_percentile)[0]
        #                     # print(indices_above)
        #                     indices_below = numpy.where(activation_values < lower_percentile)[0]
        #                     # print(indices_below)

        #                 # Non-triggered episodes
        #                 if (len(indices_above) + len(indices_below) > threshold) and not threshold_values['goal_episode_bool']:
        #                     threshold_values['goal_episode_bool'] = True
        #                     threshold_values['goal_ticker_fp'] += 1

        #                 # Trigger In Field Of View
        #                 if (len(indices_above) + len(indices_below) > threshold) and not threshold_values['trigger_episode_bool']:
        #                     threshold_values['trigger_episode_bool'] = True
        #                     threshold_values['trigger_ticker_tp'] += 1

        #             if done:

        #                 # Resetting boolean flags at the end of an episode
        #                 for percentile_values in variables.values():
        #                     for threshold_values in percentile_values.values():
        #                         threshold_values['goal_episode_bool'] = False
        #                         threshold_values['trigger_episode_bool'] = False

        #                 episode += 1
        #                 print("DONE True")

        #                 if episode == 101:
        #                     # Print summary for each threshold and percentile
        #                     for percentile_key, thresholds in variables.items():
        #                         for threshold_key, threshold_values in thresholds.items():
        #                             print(f"Goal Ticker FP {percentile_key} Percentile (Threshold {threshold_key}):",
        #                                 threshold_values['goal_ticker_fp'])
        #                             print(f"Goal Ticker TN {percentile_key} Percentile (Threshold {threshold_key}):",
        #                                 100 - threshold_values['goal_ticker_fp'])

                                    # print(f"Trigger Ticker TP {percentile_key} Percentile (Threshold {threshold_key}):",
                                    #     threshold_values['trigger_ticker_tp'])
                                    # print(f"Trigger Ticker FN {percentile_key} Percentile (Threshold {threshold_key}):",
                                    #     100 - threshold_values['trigger_ticker_tp'])

            # break

    #########           Minimum and Maximum Detector Algorithm          ##########


        if done:
        #     ########Initial 1000 episodes of model for overall trigger and non-triggered episodes##########

            # stacked_inputs_numpy= stacked_inputs.numpy()
            # print(stacked_inputs_numpy)
            # csv_filename= os.path.join(folder_path_non_triggered_256_neurons_10000,f"episode_{episode}_minigrid_dslp_activations_layer_1.csv")
            
            # with open(csv_filename,'w', newline= '') as csvfile:
            #     csv_writer= csv.writer(csvfile)
            #     csv_writer.writerows(stacked_inputs_numpy)

            #     activation = {}
            #     activation_tensors=[]
            #     m= 0
            #     i= 0
            #     store_activation_means=[]
            #     store_activation_var=[]
            #     overall_input= 0
            #     stacked_inputs=[]
                    
        #      #########Initial 10000 episodes of model for overall trigger and non-triggered episodes##########

            episode+=1
            break



            

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")

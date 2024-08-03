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

episodes=0
##### folder path for the episodic activations #####
folder_path= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_triggered"
folder_path1= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_non_triggered"


folder_path_tf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_trigger_found"
folder_path_tnf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_trigger_not_found"

folder_path_gf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_goal_found_1000"
folder_path_gnf= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_goal_not_found_1000"

folder_path_episodic= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/episodic_activations_layer_1_triggered"
##### folder path for the episodic activations #####

#### folder path for quartiles and whiskers ####

folder_path_upper_quartiles= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/upper_quartile.csv"
folder_path_lower_quartiles= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/lower_quartile.csv"
folder_path_upper_whiskers= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/upper_whisker.csv"
folder_path_lower_whiskers= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/lower_whisker.csv"
folder_path_99_and_half_percentile= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/99_and_half_percentile.csv"
folder_path_half_percentile= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/half_percentile.csv"
folder_path_99= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/99_percentile.csv"
folder_path_1= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/1_percentile.csv"
folder_path_98= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/98_percentile.csv"
folder_path_2= "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/2_percentile.csv"
#### folder path for quartiles and whiskers ####

# Load percentile data
percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/99_and_half_percentile.csv")
lower_percentile_05995 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/half_percentile.csv")
percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/99_percentile.csv")
lower_percentile_199 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/1_percentile.csv")
percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/98_percentile.csv")
lower_percentile_298 = load_percentile_data("/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/statistics_folder_1000/2_percentile.csv")



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
                                 'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 21)},
    '199': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 21)},
    '298': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 21)}
}




# ###############                     0 and 99.5 percentiles                         ###############
# #########      0 and 99.5 percentiles in threshold 5      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_05995_5= 0
# trigger_episode_bool_05995_5= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_05995_5= 0
# goal_episode_bool_05995_5= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_05995_5= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_05995_5=0
# #########      0 and 99.5 percentiles in threshold 5      ########## 

# #########      0 and 99.5 percentiles in threshold 10      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_05995_10= 0
# trigger_episode_bool_05995_10= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_05995_10= 0
# goal_episode_bool_05995_10= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_05995_10= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_05995_10=0
# #########      0 and 99.5 percentiles in threshold 10      ########## 

# #########      0 and 99.5 percentiles in threshold 15      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_05995_15= 0
# trigger_episode_bool_05995_15= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_05995_15= 0
# goal_episode_bool_05995_15= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_05995_15= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_05995_15=0
# #########      0 and 99.5 percentiles in threshold 15      ########## 

# #########      0 and 99.5 percentiles in threshold 20      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_05995_20= 0
# trigger_episode_bool_05995_20= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_05995_20= 0
# goal_episode_bool_05995_20= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_05995_20= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_05995_20=0
# #########      0 and 99.5 percentiles in threshold 20      ########## 
# ###############                     0 and 99.5 percentiles                         ###############



# ###############                     2 and 98 percentiles                         ###############
# #########      2 and 98 percentiles in threshold 5      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_199_5= 0
# trigger_episode_bool_199_5= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_199_5= 0
# goal_episode_bool_199_5= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_199_5= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_199_5=0
# #########      2 and 98 percentiles in threshold 5      ########## 

# #########      2 and 98 percentiles in threshold 10      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_199_10= 0
# trigger_episode_bool_199_10= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_199_10= 0
# goal_episode_bool_199_10= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_199_10= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_199_10=0
# #########      2 and 98 percentiles in threshold 10      ########## 

# #########      2 and 98 percentiles in threshold 15      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_199_15= 0
# trigger_episode_bool_199_15= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_199_15= 0
# goal_episode_bool_199_15= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_199_15= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_199_15=0
# #########      2 and 98 percentiles in threshold 15      ########## 

# #########      2 and 98 percentiles in threshold 20      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_199_20= 0
# trigger_episode_bool_199_20= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_199_20= 0
# goal_episode_bool_199_20= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_199_20= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_199_20=0
# #########      2 and 98 percentiles in threshold 20      ########## 
# ###############                     2 and 98 percentiles                         ###############




# ###############                     2 and 99 percentiles                         ###############
# #########      2 and 99 percentiles in threshold 5      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_298_5= 0
# trigger_episode_bool_298_5= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_298_5= 0
# goal_episode_bool_298_5= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_298_5= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_298_5=0
# #########      2 and 99 percentiles in threshold 5      ########## 

# #########      2 and 99 percentiles in threshold 10      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_298_10= 0
# trigger_episode_bool_298_10= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_298_10= 0
# goal_episode_bool_298_10= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_298_10= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_298_10=0
# #########      2 and 99 percentiles in threshold 10      ########## 

# #########      2 and 99 percentiles in threshold 15      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_298_15= 0
# trigger_episode_bool_298_15= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_298_15= 0
# goal_episode_bool_298_15= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_298_15= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_298_15=0
# #########      2 and 99 percentiles in threshold 15      ########## 

# #########      2 and 99 percentiles in threshold 20      ##########       
# # Trigger Ticker (True Positive)
# trigger_ticker_tp_298_20= 0
# trigger_episode_bool_298_20= False
# # Goal Ticker (False Positive)
# goal_ticker_fp_298_20= 0
# goal_episode_bool_298_20= False
# # Trigger Ticker (True Negative)
# trigger_ticker_fn_298_20= 0
# # Goal Ticker (False Negative)
# goal_ticker_fn_298_20=0
# #########      2 and 99 percentiles in threshold 20      ########## 
# ###############                     2 and 99 percentiles                         ###############

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
        #########Trigger found##########
        # for k in layers:
        #     # m+=1
        #     # if m>1:
        #     if episode<20:
        #         if terminated==False:

        #             activation_tensors.append(activation[str(k)].flatten())
        #             stacked_inputs= torch.stack(activation_tensors,dim=0)
        #             overall_input= len(stacked_inputs)

        #             stacked_inputs_numpy= stacked_inputs.numpy()
        #             csv_filename= os.path.join(folder_path_episodic,f"episode_{episode}_minigrid_iclr_activations_layer_1.csv")              
        #             print("Adding this into the episodic file")

        #             with open(csv_filename,'w', newline= '') as csvfile:
        #                 csv_writer= csv.writer(csvfile)
        #                 csv_writer.writerows(stacked_inputs_numpy)

        #             if done:
        #                 activation = {}
        #                 activation_tensors=[]
        #                 m= 0
        #                 i= 0
        #                 store_activation_means=[]
        #                 store_activation_var=[]
        #                 overall_input= 0
        #                 stacked_inputs=[]

        #                 episode+=1
        #                 print("DONE True")
        #                 break
        #########Single triggered episode heatmap transition##########

        #########Initial 100 episodes of model for overall trigger and non-triggered episodes##########
        # for k in layers:
        #     m+=1
        #     if m>1:

        #         activation_tensors.append(activation[str(k)].flatten())
        #         stacked_inputs= torch.stack(activation_tensors,dim=0)
        #         overall_input= len(stacked_inputs)
        #         if(overall_input % 1 == 0):
        #             i+=1
        #             stacked_input_var=  torch.var(stacked_inputs, dim=0)
        #             stacked_input_mean= torch.mean(stacked_inputs, dim=0)
        #             # print("Mean Layer 1",stacked_input_mean)
        #             # print("Var Layer 1", stacked_input_var)    
        #             store_activation_means.append(stacked_input_mean)
        #             store_activation_var.append(stacked_input_var)

        # for k in layers0:
        #     m+=1
        #     if m>1:

        #         activation_tensors1.append(activation1[str(k)].flatten())
        #         stacked_inputs1= torch.stack(activation_tensors1,dim=0)
        #         overall_input1= len(stacked_inputs1)
        #         if(overall_input1 % 1 == 0):
        #             i+=1
        #             stacked_input_var1=  torch.var(stacked_inputs1, dim=0)
        #             stacked_input_mean1= torch.mean(stacked_inputs1, dim=0)
        #             #print("Mean Layer 2",stacked_input_mean1)
        #             #print("Variance Layer 2", stacked_input_var1)    
        #             store_activation_means1.append(stacked_input_mean1)
        #             store_activation_var1.append(stacked_input_var1)
        #########Initial 100 episodes of model for overall trigger and non-triggered episodes##########
        
        #########Initial 100 episodes of model for overall trigger only episodes##########
        # #########Trigger found##########
        # for k in layers:
        #     m+=1
        #     if m>1:
        #         if episode<100:
        #             print(episode)
        #             if env.trigger_switch==False and terminated==False:
        #                 activation_tensors.append(activation[str(k)].flatten())
        #                 stacked_inputs= torch.stack(activation_tensors,dim=0)
        #                 overall_input= len(stacked_inputs)



        #                 stacked_inputs_numpy= stacked_inputs.numpy()
        #                 csv_filename= os.path.join(folder_path_tnf,f"episode_{episode}_minigrid_iclr_activations_layer_1.csv")              
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
        # # # #########Trigger found##########
                    
        # # # #########Trigger not found##########
        #             if env.trigger_switch==True and terminated==False:
        #                 activation_tensors1.append(activation[str(k)].flatten())
        #                 stacked_inputs1= torch.stack(activation_tensors1,dim=0)
        #                 overall_input1= len(stacked_inputs1)


        #                 stacked_inputs_numpy1= stacked_inputs1.numpy()
        #                 csv_filename= os.path.join(folder_path_tf,f"episode_{episode}_minigrid_iclr_activations_layer_1.csv")    
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
        #########Trigger not found##########

        ########## Goal not found##########
        # for k in layers:
        #     m+=1
        #     if m>1:
        #         if episode<1000:
        #             print(episode)
        #             if env.goal_switch==False and terminated==False:
        #                 print("Goal not found")
        #                 activation_tensors.append(activation[str(k)].flatten())
        #                 stacked_inputs= torch.stack(activation_tensors,dim=0)
        #                 overall_input= len(stacked_inputs)


        #                 stacked_inputs_numpy= stacked_inputs.numpy()
        #                 csv_filename= os.path.join(folder_path_gnf,f"episode_{episode}_minigrid_dslp_activations_layer_1.csv")              
        #                 print("Adding this into the GNF file")

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

                

        # #                     break
        # #########Goal found##########
                    
        # #########Goal found##########
        #             if env.goal_switch==True and terminated==False:
        #                 print("Goal found")
        #                 activation_tensors1.append(activation[str(k)].flatten())
        #                 stacked_inputs1= torch.stack(activation_tensors1,dim=0)
        #                 overall_input1= len(stacked_inputs1)


        #                 stacked_inputs_numpy1= stacked_inputs1.numpy()
        #                 csv_filename= os.path.join(folder_path_gf,f"episode_{episode}_minigrid_dslp_activations_layer_1.csv")    
        #                 print("Adding this into the GF file")


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
        #########Goal not found##########

        #########Initial 100 episodes of model for overall trigger only episodes##########


    #########               Detector Algorithm          ########## 
        for k in layers:
            m+=1
            if m>1:
                if episode<200:
                    print(episode)

                    percentiles = {
                        '05': 0.05,  # Replace with your actual percentile values
                        '995': 0.995,
                        '99': 0.99,
                        '1': 0.01,
                        '98': 0.98,
                        '2': 0.02
                        # Add other percentiles as needed
                    }

                    activation_values = activation[str(k)].cpu().numpy().flatten()
                    # Use loaded percentile data for comparisons
                    for percentile_key, thresholds in variables.items():
                        upper_percentile = eval(f'percentile_{percentile_key}')
                        lower_percentile = eval(f'lower_percentile_{percentile_key}')

                        for threshold_key, threshold_values in thresholds.items():
                            threshold = int(threshold_key.split('_')[-1])

                            indices_above = numpy.where(activation_values > upper_percentile)[0]
                            indices_below = numpy.where(activation_values < lower_percentile)[0]

                            # Goal In Field Of View
                            if (len(indices_above) + len(indices_below) > threshold) and not threshold_values['goal_episode_bool']:
                                threshold_values['goal_episode_bool'] = True
                                threshold_values['goal_ticker_fp'] += 1

                            # Trigger In Field Of View
                            if (len(indices_above) + len(indices_below) > threshold) and not threshold_values['trigger_episode_bool']:
                                threshold_values['trigger_episode_bool'] = True
                                threshold_values['trigger_ticker_tp'] += 1



                # ############# Goal In Field Of View #############
                #     ######### Load the 99.5th and 1.5st percentile values ##########
                #     percentile_99_and_half_percentile = pd.read_csv(folder_path_99_and_half_percentile, header=None).iloc[0]
                #     percentile_half_percentile = pd.read_csv(folder_path_half_percentile, header=None).iloc[0]

                #     # Find indices where the array exceeds quartiles
                #     indices_above_upper_995 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() > percentile_99_and_half_percentile)[0]
                #     indices_below_lower_05 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() < percentile_half_percentile)[0]

                #     # Output the index numbers
                #     print("Indices above upper whisker:", indices_above_upper_995, len(indices_above_upper_995))
                #     print("Indices below lower whisker:", indices_below_lower_05, len(indices_below_lower_05))

                    
                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 5) and goal_episode_bool_05995_5==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_05995_5=True
                #         goal_ticker_fp_05995_5+=1

                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 10) and goal_episode_bool_05995_10==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_05995_10=True
                #         goal_ticker_fp_05995_10+=1

                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 15) and goal_episode_bool_05995_15==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_05995_15=True
                #         goal_ticker_fp_05995_15+=1

                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 20) and goal_episode_bool_05995_20==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_05995_20=True
                #         goal_ticker_fp_05995_20+=1
                #     ######### Load the 99.5th and 1.5st percentile values ##########



                #     ######### Load the 99th and 1st percentile values ##########

                #     percentile_99 = pd.read_csv(folder_path_99, header=None).iloc[0]
                #     percentile_1 = pd.read_csv(folder_path_1, header=None).iloc[0]

                #     # Find indices where the array exceeds quartiles
                #     indices_above_upper_99 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() > percentile_99)[0]
                #     indices_below_lower_1 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() < percentile_1)[0]

                #     # Output the index numbers
                #     print("Indices above upper whisker:", indices_above_upper_99, len(indices_above_upper_99))
                #     print("Indices below lower whisker:", indices_below_lower_1, len(indices_below_lower_1))
                #     ######### Load the 99th and 1st percentile values ##########
                    
                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 5) and goal_episode_bool_199_5==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_199_5=True
                #         goal_ticker_fp_199_5+=1

                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 10) and goal_episode_bool_199_10==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_199_10=True
                #         goal_ticker_fp_199_10+=1

                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 15) and goal_episode_bool_199_15==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_199_15=True
                #         goal_ticker_fp_199_15+=1

                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 20) and goal_episode_bool_199_20==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_199_20=True
                #         goal_ticker_fp_199_20+=1      


                #     ######### Load the 99th and 1st percentile values ##########


                #     ######### Load the 98th and 2nd percentile values ##########

                #     percentile_98 = pd.read_csv(folder_path_98, header=None).iloc[0]
                #     percentile_2 = pd.read_csv(folder_path_2, header=None).iloc[0]

                #     # Find indices where the array exceeds quartiles
                #     indices_above_upper_98 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() > percentile_98)[0]
                #     indices_below_lower_2 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() < percentile_2)[0]

                #     # Output the index numbers
                #     print("Indices above upper whisker - 298:", indices_above_upper_98, len(indices_above_upper_98))
                #     print("Indices below lower whisker - 298:", indices_below_lower_2, len(indices_below_lower_2))
                #     ######### Load the 98th and 2st percentile values ##########
                    
                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 5) and goal_episode_bool_298_5==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_298_5=True
                #         goal_ticker_fp_298_5+=1

                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 10) and goal_episode_bool_298_10==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_298_10=True
                #         goal_ticker_fp_298_10+=1

                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 15) and goal_episode_bool_298_15==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_298_15=True
                #         goal_ticker_fp_298_15+=1

                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 20) and goal_episode_bool_298_20==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         goal_episode_bool_298_20=True
                #         goal_ticker_fp_298_20+=1      

                # ############# Goal In Field Of View #############

                # ########### Trigger In Field Of View #############

                #     ######### Load the 99.5th and 1.5st percentile values ##########
                #     percentile_99_and_half_percentile = pd.read_csv(folder_path_99_and_half_percentile, header=None).iloc[0]
                #     percentile_half_percentile = pd.read_csv(folder_path_half_percentile, header=None).iloc[0]

                #     # Find indices where the array exceeds quartiles
                #     indices_above_upper_995 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() > percentile_99_and_half_percentile)[0]
                #     indices_below_lower_05 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() < percentile_half_percentile)[0]

                #     # Output the index numbers
                #     print("Indices above upper whisker:", indices_above_upper_995, len(indices_above_upper_995))
                #     print("Indices below lower whisker:", indices_below_lower_05, len(indices_below_lower_05))
                #     ######### Load the 99th and 1st percentile values ##########
                    
                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 5) and trigger_episode_bool_05995_5==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_05995_5=True
                #         trigger_ticker_tp_05995_5+=1

                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 10) and trigger_episode_bool_05995_10==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_05995_10=True
                #         trigger_ticker_tp_05995_10+=1

                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 15) and trigger_episode_bool_05995_15==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_05995_15=True
                #         trigger_ticker_tp_05995_15+=1

                #     if (len(indices_above_upper_995) + len(indices_below_lower_05) > 20) and trigger_episode_bool_05995_20==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_05995_20=True
                #         trigger_ticker_tp_05995_20+=1
                #     ######### Load the 99.5th and 1.5st percentile values ##########



                #     ######### Load the 99th and 1st percentile values ##########

                #     percentile_99 = pd.read_csv(folder_path_99, header=None).iloc[0]
                #     percentile_1 = pd.read_csv(folder_path_1, header=None).iloc[0]

                #     # Find indices where the array exceeds quartiles
                #     indices_above_upper_99 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() > percentile_99)[0]
                #     indices_below_lower_1 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() < percentile_1)[0]

                #     # Output the index numbers
                #     print("Indices above upper whisker:", indices_above_upper_99, len(indices_above_upper_99))
                #     print("Indices below lower whisker:", indices_below_lower_1, len(indices_below_lower_1))
                #     ######### Load the 99th and 1st percentile values ##########
                    
                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 5) and trigger_episode_bool_199_5==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_199_5=True
                #         trigger_ticker_tp_199_5+=1

                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 10) and trigger_episode_bool_199_10==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_199_10=True
                #         trigger_ticker_tp_199_10+=1

                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 15) and trigger_episode_bool_199_15==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_199_15=True
                #         trigger_ticker_tp_199_15+=1

                #     if (len(indices_above_upper_99) + len(indices_below_lower_1) > 20) and trigger_episode_bool_199_20==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_199_20=True
                #         trigger_ticker_tp_199_20+=1      


                #     ######### Load the 99th and 1st percentile values ##########


                #     ######### Load the 98th and 2nd percentile values ##########

                #     percentile_98 = pd.read_csv(folder_path_98, header=None).iloc[0]
                #     percentile_2 = pd.read_csv(folder_path_2, header=None).iloc[0]

                #     # Find indices where the array exceeds quartiles
                #     indices_above_upper_98 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() > percentile_98)[0]
                #     indices_below_lower_2 = numpy.where(activation[str(k)].cpu().flatten().numpy().tolist() < percentile_2)[0]

                #     # Output the index numbers
                #     print("Indices above upper whisker:", indices_above_upper_98, len(indices_above_upper_98))
                #     print("Indices below lower whisker:", indices_below_lower_2, len(indices_below_lower_2))
                #     ######### Load the 98th and 2st percentile values ##########
                    
                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 5) and trigger_episode_bool_298_5==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_298_5=True
                #         trigger_ticker_tp_298_5+=1

                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 10) and trigger_episode_bool_298_10==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_298_10=True
                #         trigger_ticker_tp_298_10+=1

                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 15) and trigger_episode_bool_298_15==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_298_15=True
                #         trigger_ticker_tp_298_15+=1

                #     if (len(indices_above_upper_98) + len(indices_below_lower_2) > 20) and trigger_episode_bool_298_20==False and episode>=100:
                #         #input("Please check if there is an error in the environment:")
                #         trigger_episode_bool_298_20=True
                #         trigger_ticker_tp_298_20+=1      





                    if done:

                        # Resetting boolean flags at the end of an episode
                        for percentile_values in variables.values():
                            for threshold_values in percentile_values.values():
                                threshold_values['goal_episode_bool'] = False
                                threshold_values['trigger_episode_bool'] = False

                        episode += 1
                        print("DONE True")

                        if episode == 100:
                            # Print summary for each threshold and percentile
                            for percentile_key, thresholds in variables.items():
                                for threshold_key, threshold_values in thresholds.items():
                                    # print(f"Goal Ticker FP {percentile_key} Percentile (Threshold {threshold_key}):",
                                    #     threshold_values['goal_ticker_fp'])
                                    # print(f"Goal Ticker TN {percentile_key} Percentile (Threshold {threshold_key}):",
                                    #     100 - threshold_values['goal_ticker_fp'])

                                    print(f"Trigger Ticker TP {percentile_key} Percentile (Threshold {threshold_key}):",
                                        threshold_values['trigger_ticker_tp'])
                                    print(f"Trigger Ticker FN {percentile_key} Percentile (Threshold {threshold_key}):",
                                        100 - threshold_values['trigger_ticker_tp'])

            break


                        # activation = {}
                        # ############# Goal In Field Of View #############
                        # goal_episode_bool_05995_5= False
                        # goal_episode_bool_05995_10= False
                        # goal_episode_bool_05995_15= False
                        # goal_episode_bool_05995_20= False

                        # goal_episode_bool_199_5= False
                        # goal_episode_bool_199_10= False
                        # goal_episode_bool_199_15= False
                        # goal_episode_bool_199_20= False

                        # goal_episode_bool_298_5= False
                        # goal_episode_bool_298_10= False
                        # goal_episode_bool_298_15= False
                        # goal_episode_bool_298_20= False                        
                        # ############# Goal In Field Of View #############
                        
                        # ############# Trigger In Field Of View #############
                        # trigger_episode_bool_05995_5= False
                        # trigger_episode_bool_05995_10= False
                        # trigger_episode_bool_05995_15= False
                        # trigger_episode_bool_05995_20= False

                        # trigger_episode_bool_199_5= False
                        # trigger_episode_bool_199_10= False
                        # trigger_episode_bool_199_15= False
                        # trigger_episode_bool_199_20= False

                        # trigger_episode_bool_298_5= False
                        # trigger_episode_bool_298_10= False
                        # trigger_episode_bool_298_15= False
                        # trigger_episode_bool_298_20= False
                        # ############# Trigger In Field Of View #############

                        # episode+=1
                        # print("DONE True")
                        # if episode==199:
                            # print("Goal Ticker FP 0.5 and 99.5 Percentile (Threshold 5):", goal_ticker_fp_05995_5)
                            # print("Goal Ticker TN 0.5 and 99.5 Percentile (Threshold 5):", 100-goal_ticker_fp_05995_5)

                            # print("Goal Ticker FP 1 and 99 Percentile (Threshold 5):", goal_ticker_fp_199_5)
                            # print("Goal Ticker TN 1 and 99 Percentile (Threshold 5):", 100-goal_ticker_fp_199_5)

                            # print("Goal Ticker FP 2 and 98 Percentile (Threshold 5):", goal_ticker_fp_298_5)
                            # print("Goal Ticker TN 2 and 98 Percentile (Threshold 5):", 100-goal_ticker_fp_298_5)


                            # print("Goal Ticker FP 0.5 and 99.5 Percentile (Threshold 10):", goal_ticker_fp_05995_10)
                            # print("Goal Ticker TN 0.5 and 99.5 Percentile (Threshold 10):", 100-goal_ticker_fp_05995_10)

                            # print("Goal Ticker FP 1 and 99 Percentile (Threshold 10):", goal_ticker_fp_199_10)
                            # print("Goal Ticker TN 1 and 99 Percentile (Threshold 10):", 100-goal_ticker_fp_199_10)

                            # print("Goal Ticker FP 2 and 98 Percentile (Threshold 10):", goal_ticker_fp_298_10)
                            # print("Goal Ticker TN 2 and 98 Percentile (Threshold 10):", 100-goal_ticker_fp_298_10)


                            # print("Goal Ticker FP 0.5 and 99.5 Percentile (Threshold 15):", goal_ticker_fp_05995_15)
                            # print("Goal Ticker TN 0.5 and 99.5 Percentile (Threshold 15):", 100-goal_ticker_fp_05995_15)

                            # print("Goal Ticker FP 1 and 99 Percentile (Threshold 15):", goal_ticker_fp_199_15)
                            # print("Goal Ticker TN 1 and 99 Percentile (Threshold 15):", 100-goal_ticker_fp_199_15)

                            # print("Goal Ticker FP 2 and 98 Percentile (Threshold 15):", goal_ticker_fp_298_15)
                            # print("Goal Ticker TN 2 and 98 Percentile (Threshold 15):", 100-goal_ticker_fp_298_15)


                            # print("Goal Ticker FP 0.5 and 99.5 Percentile (Threshold 20):", goal_ticker_fp_05995_20)
                            # print("Goal Ticker TN 0.5 and 99.5 Percentile (Threshold 20):", 100-goal_ticker_fp_05995_20)

                            # print("Goal Ticker FP 1 and 99 Percentile (Threshold 20):", goal_ticker_fp_199_20)
                            # print("Goal Ticker TN 1 and 99 Percentile (Threshold 20):", 100-goal_ticker_fp_199_20)

                            # print("Goal Ticker FP 2 and 98 Percentile (Threshold 20):", goal_ticker_fp_298_20)
                            # print("Goal Ticker TN 2 and 98 Percentile (Threshold 20):", 100-goal_ticker_fp_298_20)




                            # print("Trigger Ticker TP 0.5 and 99.5 Percentile (Threshold 5):", trigger_ticker_tp_05995_5)
                            # print("Trigger Ticker FN 0.5 and 99.5 Percentile (Threshold 5):", 100-trigger_ticker_tp_05995_5)

                            # print("Trigger Ticker TP 1 and 99 Percentile (Threshold 5):", trigger_ticker_tp_199_5)
                            # print("Trigger Ticker FN 1 and 99 Percentile (Threshold 5):", 100-trigger_ticker_tp_199_5)

                            # print("Trigger Ticker TP 2 and 98 Percentile (Threshold 5):", trigger_ticker_tp_298_5)
                            # print("Trigger Ticker FN 2 and 98 Percentile (Threshold 5):", 100-trigger_ticker_tp_298_5)


                            # print("Trigger Ticker TP 0.5 and 99.5 Percentile (Threshold 10):", trigger_ticker_tp_05995_10)
                            # print("Trigger Ticker FN 0.5 and 99.5 Percentile (Threshold 10):", 100-trigger_ticker_tp_05995_10)

                            # print("Trigger Ticker TP 1 and 99 Percentile (Threshold 10):", trigger_ticker_tp_199_10)
                            # print("Trigger Ticker FN 1 and 99 Percentile (Threshold 10):", 100-trigger_ticker_tp_199_10)

                            # print("Trigger Ticker TP 2 and 98 Percentile (Threshold 10):", trigger_ticker_tp_298_10)
                            # print("Trigger Ticker FN 2 and 98 Percentile (Threshold 10):", 100-trigger_ticker_tp_298_10)


                            # print("Trigger Ticker TP 0.5 and 99.5 Percentile (Threshold 15):", trigger_ticker_tp_05995_15)
                            # print("Trigger Ticker FN 0.5 and 99.5 Percentile (Threshold 15):", 100-trigger_ticker_tp_05995_15)

                            # print("Trigger Ticker TP 1 and 99 Percentile (Threshold 15):", trigger_ticker_tp_199_15)
                            # print("Trigger Ticker FN 1 and 99 Percentile (Threshold 15):", 100-trigger_ticker_tp_199_15)

                            # print("Trigger Ticker TP 2 and 98 Percentile (Threshold 15):", trigger_ticker_tp_298_15)
                            # print("Trigger Ticker FN 2 and 98 Percentile (Threshold 15):", 100-trigger_ticker_tp_298_15)

                            # print("Trigger Ticker TP 0.5 and 99.5 Percentile (Threshold 20):", trigger_ticker_tp_05995_20)
                            # print("Trigger Ticker FN 0.5 and 99.5 Percentile (Threshold 20):", 100-trigger_ticker_tp_05995_20)

                            # print("Trigger Ticker TP 1 and 99 Percentile (Threshold 20):", trigger_ticker_tp_199_20)
                            # print("Trigger Ticker FN 1 and 99 Percentile (Threshold 20):", 100-trigger_ticker_tp_199_20)

                            # print("Trigger Ticker TP 2 and 98 Percentile (Threshold 20):", trigger_ticker_tp_298_20)
                            # print("Trigger Ticker FN 2 and 98 Percentile (Threshold 20):", 100-trigger_ticker_tp_298_20)




        if done:
            #########Initial 100 episodes of model for overall trigger and non-triggered episodes##########
            # if terminated==False:
            #     stacked_inputs_numpy= stacked_inputs.numpy()
            #     csv_filename= os.path.join(folder_path1,f"episode_{episode}_minigrid_dslp_activations_layer_1.csv")
                
            #     with open(csv_filename,'w', newline= '') as csvfile:
            #         csv_writer= csv.writer(csvfile)
            #         csv_writer.writerows(stacked_inputs_numpy)

                    # activation = {}
                    # activation_tensors=[]
                    # m= 0
                    # i= 0
                    # store_activation_means=[]
                    # store_activation_var=[]
                    # overall_input= 0
                    # stacked_inputs=[]
                    

            #         activation1 = {}
            #         activation_tensors1=[]
            #         m1= 0
            #         i1= 0
            #         store_activation_means1=[]
            #         store_activation_var1=[]
            #         overall_input1= 0
            #         stacked_inputs1=[]
            #########Initial 100 episodes of model for overall trigger and non-triggered episodes##########
            episode+=1
            break



            

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")

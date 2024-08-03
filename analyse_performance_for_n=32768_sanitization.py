from glob import glob
import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
import yaml
import warnings
warnings.filterwarnings("ignore")

stream = open('params.yml', 'r')
mydict = yaml.full_load(stream)

base_dir = mydict['folder']
base_input_dir = os.path.join(base_dir, 'test_outputs/sanitized_with_fixed_n')

all_dir_list = os.listdir(base_input_dir)
print(all_dir_list)
all_dimensions_base_dir = list(filter(lambda x: 'n_32768_top_d_' in x, all_dir_list)) 
print(all_dimensions_base_dir)
string_num_dict = {}
for top_d_dir in all_dimensions_base_dir:
    num = [int(num) for num in re.findall('[0-9]+', top_d_dir)][1]
    string_num_dict[num] = top_d_dir

sanitizing_sample_count_list = []
sorted_dic = collections.OrderedDict(sorted(string_num_dict.items()))
for key, value in sorted_dic.items():
    sanitizing_sample_count_list.append(key)

sample_trial_return_mean_list, sample_trial_return_std_list = [], []
for k, (top_d, top_d_basis_dir) in enumerate(sorted_dic.items()):
    
    sample_base_dir = os.path.join(base_input_dir, top_d_basis_dir, 'poison_2000')  
    return_mean_list = []
    for trial in os.listdir(sample_base_dir):
        
        trial_csv_file = os.path.join(sample_base_dir, trial, 'log', 'csv_data.csv')
        if(not os.path.exists(trial_csv_file)):
            continue 

        results_csv = pd.read_csv(trial_csv_file)
        return_str = results_csv.loc[results_csv.shape[0]-1, 'return_list']
        return_list = [int(num) for num in re.findall('[0-9]+', return_str)]
        
        return_mean, return_std = np.mean(return_list), np.std(return_list)
        return_mean_list.append(return_mean)
        
    sample_trial_return_mean_list.append(np.mean(return_mean_list)), sample_trial_return_std_list.append(np.std(return_mean_list))

sanitizing_sample_count_list = sanitizing_sample_count_list[:30]
st_return_mean_list = sample_trial_return_mean_list[:30]
st_return_std_list = sample_trial_return_mean_list[:30]
st_return_mean_list, st_return_std_list = np.array(st_return_mean_list), np.array(st_return_std_list) 


base_input_dir = "pretrained_backdoor_indist_backdoor/test_outputs/sanitized/clean_samples_32768/poison_2000/trial_0/basis"

sv_file_path = os.path.join(base_input_dir, 'sv.npy')
sv = np.load(sv_file_path)
print(sv)

# To make things reproducible...
np.random.seed(1977)

fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
plt.rcParams['font.size'] = '25'




# Make some space on the right side for the extra y-axis.
fig.subplots_adjust(right=0.75)


# To make the border of the right-most axis visible, we need to turn the frame
# on. This hides the other plots, however, so we need to turn its fill off.

# And finally we get to plot things...

# Plotting commands with labels
ax.plot(sanitizing_sample_count_list, st_return_mean_list, marker='.', linestyle='-', markersize=12, linewidth=3, color='orange', label='backdoor in in-distribution trigger env')
ax.fill_between(sanitizing_sample_count_list, st_return_mean_list-st_return_std_list, st_return_mean_list+st_return_std_list, facecolor='orange', alpha=0.25)


# Setting axis labels and tick parameters with the desired font sizes
ax.set_ylabel('Average empirical value', color='black', fontsize=30)
ax.set_xlabel('Safe subspace dimension $(d)$', fontsize=30)
ax.tick_params(axis='x', labelsize=22, rotation=30)  # Adjusted font size for x-axis ticks
ax.tick_params(axis='y', labelsize=22)  # Adjusted font size for y-axis ticks

# Positioning the legend inside the graph at the center left with fontsize 20
plt.legend(loc='center left', fontsize=23)

# ... [rest of your code]

plt.grid()
plt.tight_layout()
plt.savefig('spectrum_safe_subspace.pdf')

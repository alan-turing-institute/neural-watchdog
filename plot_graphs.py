from glob import glob
import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml 

stream = open('params.yml', 'r')
mydict = yaml.full_load(stream)

base_dir = mydict['folder']
base_input_dir = os.path.join(base_dir, 'test_outputs')

verbose = 0

def get_data_for_single_sample_count(sample_dir_path):
    all_csv_files = [file
                 for path, subdir, files in os.walk(sample_dir_path)
                 for file in glob(os.path.join(path, '*.csv'))]
    
    returns_list = []
    for csv_file in all_csv_files:
        results_df = pd.read_csv(csv_file)
        results_df = results_df.loc[:, ~results_df.columns.str.contains('^Unnamed')]
        
        return_list = results_df.groupby(['episode'])['reward'].sum()
        returns_list.append(return_list)
        return returns_list 
    
sample_trial_return_mean_list, sample_trial_return_std_list = [], []

for num_sample in sanitizing_sample_count_list:
    if(verbose):
        print('Clean samples : ', num_sample)
    sample_base_dir = os.path.join(base_input_dir, 'sanitized/clean_samples_'+str(num_sample)+'/poison_2000')     
    return_mean_list = []
    for trial in os.listdir(sample_base_dir):
        trial_csv_file = os.path.join(sample_base_dir, trial, 'log', 'csv_data.csv')

        results_csv = pd.read_csv(trial_csv_file)
        return_str = results_csv.loc[results_csv.shape[0]-1, 'return_list']
        return_list = [int(num) for num in re.findall('[0-9]+', return_str)]
        
        return_mean, return_std = np.mean(return_list), np.std(return_list)
        return_mean_list.append(return_mean)
        
        if(verbose):
            print('Mean : {0:2.4f}'.format(return_mean))
    sample_trial_return_mean_list.append(np.mean(return_mean_list)), sample_trial_return_std_list.append(np.std(return_mean_list))

st_return_mean_list, st_return_std_list = sample_trial_return_mean_list, sample_trial_return_std_list

sample_base_dir = os.path.join(base_input_dir, 'non_sanitized/no_poison') 

return_mean_list = []
for trial in os.listdir(sample_base_dir):
    trial_csv_file = os.path.join(sample_base_dir, trial, 'log', 'csv_data.csv')

    results_csv = pd.read_csv(trial_csv_file)
    return_str = results_csv.loc[results_csv.shape[0]-1, 'return_list']
    return_list = [int(num) for num in re.findall('[0-9]+', return_str)]

    return_mean, return_std = np.mean(return_list), np.std(return_list)
    return_mean_list.append(return_mean)

    if(verbose):
        print('Mean : {0:2.4f}'.format(return_mean))

tc_return_mean_list, tc_return_std_list = [np.mean(return_mean_list)]*len(sanitizing_sample_count_list), [np.std(return_mean_list)]*len(sanitizing_sample_count_list)
sample_base_dir = os.path.join(base_input_dir, 'non_sanitized/poison_2000') 

return_mean_list = []
for trial in os.listdir(sample_base_dir):
    print(trial)
    trial_csv_file = os.path.join(sample_base_dir, trial, 'log', 'csv_data.csv')

    results_csv = pd.read_csv(trial_csv_file)
    return_str = results_csv.loc[results_csv.shape[0]-1, 'return_list']
    return_list = [int(num) for num in re.findall('[0-9]+', return_str)]

    return_mean, return_std = np.mean(return_list), np.std(return_list)
    return_mean_list.append(return_mean)
    if(verbose):
        print('Mean : {0:2.4f}'.format(return_mean))

tt_return_mean_list, tt_return_std_list = [np.mean(return_mean_list)]*len(sanitizing_sample_count_list), [np.std(return_mean_list)]*len(sanitizing_sample_count_list)

fig, ax = plt.subplots(figsize=(12,8), dpi=80)
plt.rcParams['font.size'] = '25'

st_return_mean_list, st_return_std_list = np.array(st_return_mean_list), np.array(st_return_std_list)
tc_return_mean_list, tc_return_std_list = np.array(tc_return_mean_list), np.array(tc_return_std_list)
tt_return_mean_list, tt_return_std_list = np.array(tt_return_mean_list), np.array(tt_return_std_list)


plt.plot(sanitizing_sample_count_list, tc_return_mean_list, color='blue', label='backdoor in clean env')
plt.plot(sanitizing_sample_count_list, tt_return_mean_list, color='brown', label='backdoor in trigger env')
plt.plot(sanitizing_sample_count_list, st_return_mean_list,  marker='.', linestyle='-', markersize=12, color='orange',  label='sanitized in-distribution trigger env')

plt.fill_between(sanitizing_sample_count_list, tc_return_mean_list-tc_return_std_list, tc_return_mean_list+tc_return_std_list, facecolor='blue', alpha=0.25)
plt.fill_between(sanitizing_sample_count_list, tt_return_mean_list-tt_return_std_list, tt_return_mean_list+tt_return_std_list, facecolor='brown', alpha=0.25)
plt.fill_between(sanitizing_sample_count_list, st_return_mean_list-st_return_std_list, st_return_mean_list+st_return_std_list, facecolor='orange', alpha=0.25)


plt.xlabel('Clean sanitization samples (n)', fontsize=25)
plt.ylabel('Average empirical value', fontsize=25)
plt.xticks(sanitizing_sample_count_list, rotation=30, fontsize=21)  # Adjusted font size for x-axis ticks
plt.yticks(fontsize=22)  # Adjusted font size for y-axis ticks

xticks = ax.xaxis.get_major_ticks()

for i in [5, 6, 7, 9, 10, 11]:
    xticks[i].set_visible(False)

plt.grid()
plt.legend(loc='center left', fontsize=19)  # Adjusted legend fontsize
plt.tight_layout()
plt.savefig('performance_breakout.pdf')

import pandas as pd
import os
from scipy.stats import median_abs_deviation
import numpy as np

# Directory containing the CSV files
directory = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_models_folder_256_neurons/activations_per_episode_layer_1_goal_found_10000'

# Directory to save the output CSV files
output_directory = os.path.join(directory, 'statistics_folder')
os.makedirs(output_directory, exist_ok=True)

# Function to calculate quartiles and whiskers
def calculate_quartiles_whiskers(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_whisker = column[column >= q1 - 1.5 * iqr].min()
    upper_whisker = column[column <= q3 + 1.5 * iqr].max()
    return lower_whisker, q1, q3, upper_whisker

# Combine CSV files and calculate statistics
all_data = pd.DataFrame()
statistics = {
    'lower_whisker': [], 
    'lower_quartile': [], 
    'upper_quartile': [], 
    'upper_whisker': [], 
    'median_absolute_deviation': [],
    ####99.5 and 0.5 percentiles####
    '99_and_half_percentile': [],
    'half_percentile': [],
    ####99 and 1 percentiles####
    '99_percentile': [],
    '1_percentile': [],
    ####98 and 2 percentiles####
    '98_percentile': [],
    '2_percentile': [],
    ####Max and Min####
    'Max': [],
    'Min': [],
}
# Process each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        data = pd.read_csv(file_path, header=None)
        all_data = pd.concat([all_data, data], axis=0)

# Reset index after concatenation
all_data.reset_index(drop=True, inplace=True)

# Calculate statistics for each column
for col in all_data.columns:
    print(col)
    lw, lq, uq, uw = calculate_quartiles_whiskers(all_data[col])
    mad= median_abs_deviation(all_data[col], scale='normal')

    # Calculate 99.5th and half percentiles
    percentile_99_and_half = all_data[col].quantile(0.995)
    percentile_half = all_data[col].quantile(0.005)
    # Calculate 99th and 1st percentiles
    percentile_99 = all_data[col].quantile(0.99)
    percentile_1 = all_data[col].quantile(0.01)
    # Calculate 98th and 2nd percentiles
    percentile_98 = all_data[col].quantile(0.98)
    percentile_2 = all_data[col].quantile(0.02)
    # Calculate Max and Min
    max_value = all_data[col].max()
    min_value = all_data[col].min()
    

    statistics['lower_whisker'].append(lw)
    statistics['lower_quartile'].append(lq)
    statistics['upper_quartile'].append(uq)
    statistics['upper_whisker'].append(uw)
    statistics['median_absolute_deviation'].append(mad)

    ####99.5 and 0.5 percentiles####
    statistics['99_and_half_percentile'].append(percentile_99_and_half)
    statistics['half_percentile'].append(percentile_half)
    ####99 and 1 percentiles####
    statistics['99_percentile'].append(percentile_99)
    statistics['1_percentile'].append(percentile_1)
    ####98 and 2 percentiles####
    statistics['98_percentile'].append(percentile_98)
    statistics['2_percentile'].append(percentile_2)
    ####Max and Min####
    statistics['Max'].append(max_value)
    statistics['Min'].append(min_value)

# Save the statistics to CSV files in the output directory
for key in statistics:
    try:
        df_stat = pd.DataFrame(statistics[key]).transpose()
        output_file = os.path.join(output_directory, f'{key}.csv')
        df_stat.to_csv(output_file, index=False, header=False)
    except Exception as e:
        print(f"Error saving {key}: {e}")
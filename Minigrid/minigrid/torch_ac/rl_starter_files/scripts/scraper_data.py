import re
import csv
# Defining the data types and percentile numbers
data_types = ['TP', 'FN', 'FP', 'TN']
percentile_numbers = ['05995', '199', '298', '0599', '0598', '1995', '198', '2995', '298']

# Generating the arrays
arrays = {}

# Loop through each combination of data type and percentile number to create an array
for data_type in data_types:
    for percentile_number in percentile_numbers:
        # Naming convention: [Data_Type]_[Percentile_Number]
        array_name = f"{data_type}_{percentile_number}"
        arrays[array_name] = [data_type, percentile_number]

# Displaying the array names to confirm all 36 have been created
list(arrays.keys())

# Function to parse a line and extract relevant information
def parse_line_trigger(line):
    match = re.search(r"Trigger Ticker (\w+) (\d+) Percentile \(Threshold threshold_(\d+)\): (\d+)", line)
    if match:
        data_type = match.group(1)
        percentile_number = match.group(2)
        threshold = match.group(3)  # This is extracted but not used directly in this example
        value = int(match.group(4))
        return data_type, percentile_number, value
    else:
        return None, None, None
    
def parse_line_goal(line):
    match = re.search(r"Goal Ticker (\w+) (\d+) Percentile \(Threshold threshold_(\d+)\): (\d+)", line)
    if match:
        data_type = match.group(1)
        percentile_number = match.group(2)
        threshold = match.group(3)  # This is extracted but not used directly in this example
        value = int(match.group(4))
        return data_type, percentile_number, value
    else:
        return None, None, None

# Function to add the value to the correct array based on data type and percentile number
def add_to_array(data_type, percentile_number, value):
    key = f"{data_type}_{percentile_number}"
    if key not in arrays:
        arrays[key] = []
    arrays[key].append(value)

# Example of processing a file (replace 'file_path' with the actual file path)
file_path = "/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/trigger_256_neurons"
with open(file_path, "r") as file:
    for line in file:
        data_type, percentile_number, value = parse_line_trigger(line)
        if data_type and percentile_number:
            add_to_array(data_type, percentile_number, value)

file_path1="/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/goal_256_neurons"

with open(file_path1, "r") as file:
    for line in file:
        data_type, percentile_number, value = parse_line_goal(line)
        if data_type and percentile_number:
            add_to_array(data_type, percentile_number, value)

# Specify the output CSV file name
output_csv_file = 'classifier_data_256_neurons_v2.csv'

# Open the CSV file for writing
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Data_Type_Percentile', 'Values'])
    
    # Iterate through the dictionary and write each item to the CSV
    for key, values in arrays.items():
        # Prepare the row to be written: key followed by all the values
        row = [key] + values
        writer.writerow(row)

print(f'Data written to {output_csv_file}')
import os
import pandas as pd
import glob
from sklearn.cluster import KMeans
import joblib

# Function to load data from CSV files and combine them into a DataFrame
def load_data(directory, filename_pattern):
    file_pattern = os.path.join(directory, filename_pattern)
    files = glob.glob(file_pattern)
    data_list = []

    for file in files:
        df = pd.read_csv(file, header=None)
        data_list.append(df)

    return pd.concat(data_list, ignore_index=True)

# Specify the folder path containing the CSV files for clean training data
clean_training_folder_path = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_goal_found'

# Load your clean training data
clean_training_data = load_data(clean_training_folder_path, 'episode_*_minigrid_dslp_activations_layer_1.csv')

# Define the number of clusters for training (in your case, 1 cluster)
n_clusters = 1

# Initialize and fit the K-Means clustering model using clean training data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(clean_training_data)

# Save the trained clustering model using joblib
model_filename = 'kmeans_model.pkl'
joblib.dump(kmeans, model_filename)

# Specify the folder path containing the CSV files for malicious data (anomaly detection)
malicious_data_folder_path = '/Users/svyas/meta_rl/Minigrid/minigrid/torch-ac/rl-starter-files/DSLP_model_folders_60k/activations_per_episode_layer_1_trigger_found'

# Load your malicious data for anomaly detection
malicious_data = load_data(malicious_data_folder_path, 'episode_*_minigrid_iclr_activations_layer_1.csv')

# Load the trained K-Means clustering model for anomaly detection
loaded_kmeans = joblib.load(model_filename)

# Predict clusters for the malicious data
predicted_clusters = loaded_kmeans.predict(malicious_data)

# Define the threshold for anomaly detection (you can adjust this threshold)
anomaly_threshold = 0  # Adjust as needed

# List the row numbers that are anomalies
anomaly_rows = [i for i, label in enumerate(predicted_clusters) if label != 0]

# Print the row numbers of anomalies
print("Anomaly Rows:", anomaly_rows)

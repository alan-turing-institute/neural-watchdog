import matplotlib.pyplot as plt
from sklearn.metrics import auc


def calculate_f1_scores(TP, FN, FP, TN):
    metrics = []
    for i in range(len(TP)):
        precision = TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] != 0 else 0
        recall = TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        metrics.append(f1_score)
    return metrics

def calculate_roc_curve(TP, FN, FP, TN):
    tpr = [TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] != 0 else 0 for i in range(len(TP))]
    fpr = [FP[i] / (FP[i] + TN[i]) if FP[i] + TN[i] != 0 else 0 for i in range(len(FP))]
    return tpr, fpr

# Define the data for each classifier
classifiers_data = {
    # Data for classifier 05995
    "0.5% and 99.5%": {
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 99, 99, 96, 96, 92, 81, 81, 79, 79, 77, 70, 67, 66, 65, 57, 56, 54, 45, 42, 41, 39, 36, 36, 32, 26, 26],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 8, 19, 19, 21, 21, 23, 30, 33, 34, 35, 43, 44, 46, 55, 58, 59, 61, 64, 64, 68, 74, 74],
        "FP": [92, 76, 60, 56, 43, 40, 37, 35, 33, 31, 29, 28, 28, 22, 21, 20, 18, 18, 18, 17, 17, 17, 16, 16, 16, 15, 15, 14, 14, 14, 14, 14, 13, 13, 13, 13, 9, 8, 8, 8],
        "TN": [8, 24, 40, 44, 57, 60, 63, 65, 67, 69, 71, 72, 72, 78, 79, 80, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 86, 86, 86, 86, 86, 87, 87, 87, 87, 91, 92, 92, 92]
    },
    # # Data for classifier 199
    "1% and 99%": {
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 96, 95, 93, 93, 93, 93, 85, 85, 85, 83, 80, 79, 71, 71, 71, 71, 67, 66, 60, 57],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 7, 7, 7, 7, 15, 15, 15, 17, 20, 21, 29, 29, 29, 29, 33, 34, 40, 43],
        "FP": [100, 96, 89, 83, 75, 69, 61, 56, 53, 52, 49, 47, 45, 42, 39, 37, 35, 33, 28, 28, 28, 25, 25, 21, 19, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 16, 16, 15],
        "TN": [0, 4, 11, 17, 25, 31, 39, 44, 47, 48, 51, 53, 55, 58, 61, 63, 65, 67, 72, 72, 72, 75, 75, 79, 81, 82, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 84, 84, 85]
    },
    # Data for classifier 298
    "2% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 298
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 96, 96, 95, 95, 95, 95, 95, 94, 91, 91, 90],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 5, 5, 5, 5, 6, 9, 9, 10],
        "FP": [100, 99, 99, 98, 98, 93, 90, 86, 83, 82, 76, 73, 70, 66, 64, 60, 57, 54, 54, 50, 49, 49, 48, 44, 43, 42, 39, 35, 35, 35, 35, 31, 29, 29, 28, 27, 27, 27, 24, 19],
        "TN": [0, 1, 1, 2, 2, 7, 10, 14, 17, 18, 24, 27, 30, 34, 36, 40, 43, 46, 46, 50, 51, 51, 52, 56, 57, 58, 61, 65, 65, 65, 65, 69, 71, 71, 72, 73, 73, 73, 76, 81]
    },
    # Data for classifier 0599
    "0.5% and 99%": {
        # ... add TP, FN, FP, TN values for classifier 0599
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 98, 93, 93, 93, 93, 82, 81, 81, 81, 73, 73, 71, 71, 67, 66, 65, 65, 58, 47, 45, 41, 41],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 7, 7, 7, 7, 18, 19, 19, 19, 27, 27, 29, 29, 33, 34, 35, 35, 42, 53, 55, 59, 59],
        "FP": [99, 94, 77, 72, 63, 58, 52, 47, 46, 45, 41, 36, 34, 32, 32, 28, 28, 28, 26, 19, 18, 18, 18, 18, 18, 18, 18, 17, 16, 15, 15, 15, 15, 14, 13, 13, 13, 13, 13, 13],
        "TN": [1, 6, 23, 28, 37, 42, 48, 53, 54, 55, 59, 64, 66, 68, 68, 72, 72, 72, 74, 81, 82, 82, 82, 82, 82, 82, 82, 83, 84, 85, 85, 85, 85, 86, 87, 87, 87, 87, 87, 87]
    },
    # Data for classifier 0598
    "0.5% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 0598
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 95, 94, 94, 91, 91, 91, 90, 90, 90, 83, 75, 75, 74, 73, 73, 70, 66, 66],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 6, 6, 9, 9, 9, 10, 10, 10, 17, 25, 25, 26, 27, 27, 30, 34, 34],
        "FP": [100, 99, 95, 85, 82, 78, 75, 69, 64, 59, 56, 52, 48, 45, 43, 42, 41, 38, 36, 36, 35, 29, 29, 28, 28, 27, 24, 23, 18, 18, 18, 18, 18, 18, 18, 18, 17, 16, 16, 15],
        "TN": [0, 1, 5, 15, 18, 22, 25, 31, 36, 41, 44, 48, 52, 55, 57, 58, 59, 62, 64, 64, 65, 71, 71, 72, 72, 73, 76, 77, 82, 82, 82, 82, 82, 82, 82, 82, 83, 84, 84, 85]
    },
    # Data for classifier 1995
    "1% and 99.5%": {
        # ... add TP, FN, FP, TN values for classifier 1995
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 93, 93, 93, 85, 82, 82, 80, 77, 77, 77, 70, 61, 60, 60, 57, 54, 49, 48, 46, 43, 37],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 7, 7, 15, 18, 18, 20, 23, 23, 23, 30, 39, 40, 40, 43, 46, 51, 52, 54, 57, 63],
        "FP": [97, 92, 80, 70, 64, 57, 51, 45, 41, 39, 37, 34, 31, 29, 29, 29, 28, 21, 21, 21, 19, 18, 18, 18, 18, 18, 18, 18, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
        "TN":  [3, 8, 20, 30, 36, 43, 49, 55, 59, 61, 63, 66, 69, 71, 71, 71, 72, 79, 79, 79, 81, 82, 82, 82, 82, 82, 82, 82, 84, 84, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85]
    },
    # Data for classifier 198
    "1% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 198
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 96, 95, 95, 95, 95, 94, 94, 94, 93, 90, 90, 84, 76, 76, 76, 75, 75],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 5, 5, 5, 6, 6, 6, 7, 10, 10, 16, 24, 24, 24, 25, 25],
        "FP": [100, 99, 97, 90, 89, 83, 81, 76, 73, 67, 63, 58, 57, 54, 53, 47, 47, 44, 41, 40, 38, 38, 36, 36, 32, 30, 29, 29, 27, 25, 25, 24, 18, 18, 18, 18, 18, 18, 18, 18],
        "TN": [0, 1, 3, 10, 11, 17, 19, 24, 27, 33, 37, 42, 43, 46, 47, 53, 53, 56, 59, 60, 62, 62, 64, 64, 68, 70, 71, 71, 73, 75, 75, 76, 82, 82, 82, 82, 82, 82, 82, 82]
    },
    # Data for classifier 2995
    "2% and 99.5%": {
        # ... add TP, FN, FP, TN values for classifier 2995
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 96, 94, 84, 83, 82, 82, 79, 77, 77, 77, 71, 59, 58, 57, 55],   
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 16, 17, 18, 18, 21, 23, 23, 23, 29, 41, 42, 43, 45],
        "FP": [100, 99, 98, 95, 86, 79, 72, 66, 62, 57, 52, 50, 48, 46, 44, 41, 37, 34, 32, 30, 29, 29, 28, 27, 22, 20, 19, 19, 19, 19, 19, 18, 17, 17, 17, 17, 16, 16, 16, 15],
        "TN": [0, 1, 2, 5, 14, 21, 28, 34, 38, 43, 48, 50, 52, 54, 56, 59, 63, 66, 68, 70, 71, 71, 72, 73, 78, 80, 81, 81, 81, 81, 81, 82, 83, 83, 83, 83, 84, 84, 84, 85]
    },
    # Data for classifier 299
    "2% and 99%": {
        # ... add TP, FN, FP, TN values for classifier 299
        "TP": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 96, 96, 95, 95, 93, 85, 85, 83, 83, 80, 79, 79, 73, 71],
        "FN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 5, 7, 15, 15, 17, 17, 20, 21, 21, 27, 29],
        "FP": [100, 99, 99, 98, 90, 86, 82, 79, 70, 64, 61, 58, 56, 54, 53, 49, 47, 46, 43, 42, 39, 36, 32, 31, 31, 29, 29, 28, 28, 27, 19, 19, 19, 19, 18, 17, 17, 17, 17, 17],
        "TN": [0, 1, 1, 2, 10, 14, 18, 21, 30, 36, 39, 42, 44, 46, 47, 51, 53, 54, 57, 58, 61, 64, 68, 69, 69, 71, 71, 72, 72, 73, 81, 81, 81, 81, 82, 83, 83, 83, 83, 83]
    }
}

# Calculate F1 scores for each classifier
f1_scores_classifiers = {}
for classifier, data in classifiers_data.items():
    print(len(data["TP"]), len(data["FN"]), len(data["FP"]), len(data["TN"]))
    f1_scores_classifiers[classifier] = calculate_f1_scores(data["TP"], data["FN"], data["FP"], data["TN"])

# Plotting the graph for all classifiers
plt.figure(figsize=(12, 7))
thresholds = list(range(1, 41))

for classifier, f1_scores in f1_scores_classifiers.items():
    plt.plot(thresholds, f1_scores, marker='o', label=f'Classifier {classifier} Percentile')

plt.title('F1 Score over Thresholds for Classifiers', fontsize=22)
plt.xlabel('Threshold Value', fontsize=21)
plt.ylabel('F1 Score', fontsize=21)
plt.xticks(thresholds , fontsize=12)  # Set x-ticks to integers
plt.yticks(fontsize=10)  # Set y-ticks font size
plt.legend(loc=(0.52, 0.015), fontsize=14)
plt.grid(True)
plt.tight_layout()  # Add tight layout
plt.savefig('f1_scores_classifiers_128_neurons.pdf')  # Save the plot to a file
plt.savefig('f1_scores_classifiers_128_neurons.png')  # Save the plot to a file

### ROC Curve ###
# Create a single plot for all ROC curves
plt.figure(figsize=(10, 6))

# Define a list of line styles or colors to differentiate the curves
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

# Loop through each classifier and plot its ROC curve with a unique line style
for i, (classifier_name, data) in enumerate(classifiers_data.items()):
    tpr, fpr = calculate_roc_curve(data['TP'], data['FN'], data['FP'], data['TN'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classifier_name} (AUC = {roc_auc:.2f})', linestyle=line_styles[i % len(line_styles)])

    
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=22)
plt.xlabel('False Positive Rate', fontsize=21)
plt.ylabel('True Positive Rate', fontsize=21)
plt.xticks(fontsize=15)  # Set x-ticks font size
plt.yticks(fontsize=15)  # Set y-ticks font size
plt.legend(fontsize=20)
plt.tight_layout()
plt.grid(True)
plt.savefig('ROC_curve_classifiers_128_neurons.png')
plt.savefig('ROC_curves_classifiers_128_neurons.pdf') 

best_thresholds = {}

for classifier, data in classifiers_data.items():
    tpr = [TP / (TP + FN) for TP, FN in zip(data['TP'], data['FN'])]
    fpr = [FP / (FP + TN) for FP, TN in zip(data['FP'], data['TN'])]
    
    # Find the index with the highest TPR and lowest FPR
    best_index = max(range(len(tpr)), key=lambda i: (tpr[i], -fpr[i]))
    
    # Retrieve the corresponding threshold value
    best_threshold = best_index / 100.0
    
    # Store the best threshold for the classifier
    best_thresholds[classifier] = best_threshold

print(best_thresholds)
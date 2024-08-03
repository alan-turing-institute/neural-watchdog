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
        "TP": [98, 98, 98, 98, 94, 94, 92, 87, 78, 77, 75, 75, 71, 70, 66, 64, 57, 57, 55, 50],
        "FN": [2, 2, 2, 2, 6, 6, 8, 13, 22, 23, 25, 25, 29, 30, 34, 36, 43, 43, 45, 50],
        "FP": [94, 89, 62, 51, 36, 28, 22, 20, 12, 11, 7, 7, 7, 4, 2, 2, 2, 2, 2, 0],
        "TN": [6, 11, 38, 49, 64, 72, 78, 80, 88, 89, 93, 93, 93, 96, 98, 98, 98, 98, 98, 100]
    },
    # # Data for classifier 199
    "1% and 99%": {
        # ... add TP, FN, FP, TN values for classifier 199
        "TP": [99, 99, 98, 98, 98, 98, 94, 94, 94, 89, 87, 86, 85, 75, 75, 75, 73, 69, 69, 68],
        "FN": [1, 1, 2, 2, 2, 2, 6, 6, 6, 11, 13, 14, 15, 25, 25, 25, 27, 31, 31, 32],
        "FP": [96, 95, 92, 71, 63, 56, 53, 37, 30, 29, 22, 18, 18, 18, 16, 10, 10, 8, 5, 3],
        "TN": [4, 5, 8, 29, 37, 44, 47, 63, 70, 71, 78, 82, 82, 82, 84, 90, 90, 92, 95, 97]
    },
    # Data for classifier 298
    "2% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 298
        "TP": [99, 99, 99, 99, 98, 98, 98, 98, 98, 94, 94, 94, 89, 87, 82, 82, 77, 77, 77, 73],
        "FN": [1, 1, 1, 1, 2, 2, 2, 2, 2, 6, 6, 6, 11, 13, 18, 18, 23, 23, 23, 27],
        "FP": [97, 95, 95, 94, 92, 89, 83, 77, 74, 70, 60, 55, 40, 35, 27, 25, 23, 18, 16, 16],
        "TN": [3, 5, 5, 6, 8, 11, 17, 23, 26, 30, 40, 45, 60, 65, 73, 75, 77, 82, 84, 84]
    },
    # Data for classifier 0599
    "0.5% and 99%": {
        # ... add TP, FN, FP, TN values for classifier 0599
        "TP": [98, 98, 98, 98, 98, 98, 93, 89, 89, 82, 78, 75, 74, 73, 72, 70, 68, 63, 61, 51],
        "FN": [2, 2, 2, 2, 2, 2, 7, 11, 11, 18, 22, 25, 26, 27, 28, 30, 32, 37, 39, 49],
        "FP": [96, 89, 67, 44, 36, 32, 28, 22, 18, 17, 13, 9, 8, 5, 5, 3, 2, 2, 2, 2],
        "TN": [4, 11, 33, 56, 64, 68, 72, 78, 82, 83, 87, 91, 92, 95, 95, 97, 98, 98, 98, 98]
    },
    # Data for classifier 0598
    "0.5% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 0598
        "TP": [99, 99, 98, 98, 98, 98, 98, 98, 89, 87, 86, 83, 80, 75, 75, 73, 71, 70, 69, 67],
        "FN": [1, 1, 2, 2, 2, 2, 2, 2, 11, 13, 14, 17, 20, 25, 25, 27, 29, 30, 31, 33],
        "FP": [96, 95, 90, 76, 58, 51, 40, 36, 34, 27, 22, 18, 16, 16, 10, 8, 8, 6, 3, 3],
        "TN": [4, 5, 10, 24, 42, 49, 60, 64, 66, 73, 78, 82, 84, 84, 90, 92, 92, 94, 97, 97]
    },
    # Data for classifier 1995
    "1% and 99.5%": {
        # ... add TP, FN, FP, TN values for classifier 1995
        "TP": [98, 98, 98, 98, 98, 93, 90, 84, 81, 81, 76, 74, 70, 68, 65, 65, 59, 58, 56, 53],
        "FN": [2, 2, 2, 2, 2, 7, 10, 16, 19, 19, 24, 26, 30, 32, 35, 35, 41, 42, 44, 47],
        "FP": [95, 88, 78, 65, 50, 34, 23, 21, 16, 15, 15, 14, 9, 5, 2, 2, 2, 1, 1, 0],
        "TN": [5, 12, 22, 35, 50, 66, 77, 79, 84, 85, 85, 86, 91, 95, 98, 98, 98, 99, 99, 100]        
    },
    # Data for classifier 198
    "1% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 198
        "TP": [99, 99, 98, 98, 98, 98, 98, 98, 98, 98, 89, 85, 82, 80, 78, 75, 73, 71, 69, 68],
        "FN": [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 11, 15, 18, 20, 22, 25, 27, 29, 31, 32],
        "FP": [96, 95, 91, 90, 73, 67, 60, 44, 43, 31, 26, 23, 18, 18, 18, 15, 10, 8, 6, 4],
        "TN": [4, 5, 9, 10, 27, 33, 40, 56, 57, 69, 74, 77, 82, 82, 82, 85, 90, 92, 94, 96]
    },
    # Data for classifier 2995
    "2% and 99.5%": {
        # ... add TP, FN, FP, TN values for classifier 2995
        "TP": [99, 99, 98, 98, 98, 98, 90, 86, 83, 83, 81, 76, 75, 73, 71, 68, 68, 65, 64, 62],
        "FN": [1, 1, 2, 2, 2, 2, 10, 14, 17, 17, 19, 24, 25, 27, 29, 32, 32, 35, 36, 38],
        "FP": [95, 95, 93, 86, 78, 69, 61, 44, 28, 23, 19, 17, 17, 16, 11, 10, 5, 3, 1, 1],
        "TN": [5, 5, 7, 14, 22, 31, 39, 56, 72, 77, 81, 83, 83, 84, 89, 90, 95, 97, 99, 99]
    },
    # Data for classifier 299
    "2% and 99%": {
        # ... add TP, FN, FP, TN values for classifier 299
        "TP": [99, 99, 98, 98, 98, 98, 98, 98, 98, 89, 88, 85, 79, 77, 73, 72, 71, 70, 70, 68],
        "FN": [1, 1, 2, 2, 2, 2, 2, 2, 2, 11, 12, 15, 21, 23, 27, 28, 29, 30, 30, 32],
        "FP": [96, 95, 94, 90, 80, 77, 65, 56, 46, 41, 28, 22, 18, 17, 13, 12, 11, 8, 5, 5],
        "TN": [4, 5, 6, 10, 20, 23, 35, 44, 54, 59, 72, 78, 82, 83, 87, 88, 89, 92, 95, 95]
    }
}

# Calculate F1 scores for each classifier
f1_scores_classifiers = {}
for classifier, data in classifiers_data.items():
    f1_scores_classifiers[classifier] = calculate_f1_scores(data["TP"], data["FN"], data["FP"], data["TN"])

# Plotting the graph for all classifiers
plt.figure(figsize=(12, 7))
thresholds = list(range(1, 21))

for classifier, f1_scores in f1_scores_classifiers.items():
    plt.plot(thresholds, f1_scores, marker='o', label=f'Classifier {classifier} Percentile')

plt.title('F1 Score over Thresholds for Classifiers', fontsize=22)
plt.xlabel('Threshold Value', fontsize=21)
plt.ylabel('F1 Score', fontsize=21)
plt.xticks(thresholds, fontsize=20)  # Set x-ticks to integers
plt.yticks(fontsize=20)  # Set y-ticks font size
plt.legend(loc=(0.52, 0.015), fontsize=14)
plt.grid(True)
plt.tight_layout()  # Add tight layout
plt.savefig('f1_scores_classifiers.pdf')  # Save the plot to a file
plt.savefig('f1_scores_classifiers.png')  # Save the plot to a file

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
plt.xticks(fontsize=20)  # Set x-ticks font size
plt.yticks(fontsize=20)  # Set y-ticks font size
plt.legend(fontsize=17)
plt.tight_layout()
plt.grid(True)
plt.savefig('ROC_curve_classifiers.png')
plt.savefig('ROC_curves_classifiers.pdf')

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
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,96,96,96,96,96,95,94,94,94,94,94,93,92,92,92,92,92,92,90,89,89,89,79,79,79,79,71,69,65,64,63,61,61,61,60,58,58,56,56,56,56,56,55,55,54,48,46,46,46,42,42,40,40],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,5,6,6,6,6,6,7,8,8,8,8,8,8,10,11,11,11,21,21,21,21,29,31,35,36,37,39,39,39,40,42,42,44,44,44,44,44,45,45,46,52,54,54,54,58,58,60,60],
        "FP": [99,99,99,95,93,93,90,84,84,82,78,74,59,48,47,45,44,44,40,36,33,32,29,29,29,23,21,21,20,18,18,17,14,12,12,12,9,8,7,6,6,6,5,5,5,5,4,4,4,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
        "TN": [1,1,1,5,7,7,10,16,16,18,22,26,41,52,53,55,56,56,60,64,67,68,71,71,71,77,79,79,80,82,82,83,86,88,88,88,91,92,93,94,94,94,95,95,95,95,96,96,96,98,98,98,98,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,100,100,100,100,100,100,100,100,100,100],
    },
    # # Data for classifier 199
    "1% and 99%": {
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,99,99,99,99,98,98,93,93,93,93,93,93,93,92,92,92,91,91,91,91,90,87,86,86,86,79,78,78,76,66,66,66,62,62,60,58,58,57,57],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,7,7,7,7,7,7,7,8,8,8,9,9,9,9,10,13,14,14,14,21,22,22,24,34,34,34,38,38,40,42,42,43,43],
        "FP": [100,100,100,100,100,96,95,95,93,91,91,89,87,87,85,83,82,80,79,76,75,54,49,45,42,40,39,38,37,37,35,35,28,25,23,22,21,21,21,21,21,21,20,20,20,17,16,12,12,12,12,12,11,9,8,8,6,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1],
        "TN": [0,0,0,0,0,4,5,5,7,9,9,11,13,13,15,17,18,20,21,24,25,46,51,55,58,60,61,62,63,63,65,65,72,75,77,78,79,79,79,79,79,79,80,80,80,83,84,88,88,88,88,88,89,91,92,92,94,96,96,96,96,96,96,98,98,98,98,98,98,98,98,98,98,99,99,99,99,99,99,99],
    },
    # Data for classifier 298
    "2% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 298
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,96,96,96,96,96,96,95,94,94,93,92,92,92,92,92,92,92,92,92,92,92],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,4,5,6,6,7,8,8,8,8,8,8,8,8,8,8,8],
        "FP": [100,100,100,100,100,100,100,100,100,100,100,100,96,96,95,95,95,95,93,93,89,89,89,88,87,86,86,84,83,81,80,77,63,61,59,59,55,47,46,43,40,40,40,39,38,32,31,26,26,26,25,25,25,25,25,22,22,22,21,21,20,20,20,20,18,18,17,16,14,14,13,13,11,11,10,8,5,3,3,3],
        "TN": [0,0,0,0,0,0,0,0,0,0,0,0,4,4,5,5,5,5,7,7,11,11,11,12,13,14,14,16,17,19,20,23,37,39,41,41,45,53,54,57,60,60,60,61,62,68,69,74,74,74,75,75,75,75,75,78,78,78,79,79,80,80,80,80,82,82,83,84,86,86,87,87,89,89,90,92,95,97,97,97],
   },
    # Data for classifier 0599
    "0.5% and 99%": {
        # ... add TP, FN, FP, TN values for classifier 0599
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,95,95,95,95,95,95,94,94,94,94,93,92,92,92,92,92,91,91,90,90,89,89,89,80,78,78,76,75,74,74,71,62,60,60,59,59,58,56,56,55,55,55,54,46,46,44],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,5,5,5,5,6,6,6,6,7,8,8,8,8,8,9,9,10,10,11,11,11,20,22,22,24,25,26,26,29,38,40,40,41,41,42,44,44,45,45,45,46,54,54,56],
        "FP": [100,100,99,95,95,93,91,91,87,86,85,83,81,80,79,65,61,56,47,42,41,40,39,37,36,36,32,31,25,23,23,21,21,20,20,20,18,16,15,14,12,12,12,12,12,12,10,8,5,5,5,4,4,4,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        "TN": [0,0,1,5,5,7,9,9,13,14,15,17,19,20,21,35,39,44,53,58,59,60,61,63,64,64,68,69,75,77,77,79,79,80,80,80,82,84,85,86,88,88,88,88,88,88,90,92,95,95,95,96,96,96,98,98,98,98,98,98,98,98,98,99,99,99,99,99,99,99,99,99,99,99,99,99,99,100,100,100],
    },
    # Data for classifier 0598
    "0.5% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 1995
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,96,96,96,96,96,96,95,95,95,95,94,92,92,92,92,92,92,92,92,92,92,91,91,90,90,83,81,78,77,77,76,76,76,75,73,69,69,69],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,4,5,5,5,5,6,8,8,8,8,8,8,8,8,8,8,9,9,10,10,17,19,22,23,23,24,24,24,25,27,31,31,31],
        "FP": [100,100,100,100,100,100,96,96,95,95,95,93,90,87,87,87,86,85,84,71,70,65,63,58,54,54,53,47,42,38,37,36,35,28,28,27,25,25,25,25,23,23,23,21,21,20,18,18,18,17,15,15,14,14,12,12,11,10,9,7,6,5,5,3,3,3,3,3,3,3,2,2,2,2,1,1,1,1,1,1],
        "TN": [0,0,0,0,0,0,4,4,5,5,5,7,10,13,13,13,14,15,16,29,30,35,37,42,46,46,47,53,58,62,63,64,65,72,72,73,75,75,75,75,77,77,77,79,79,80,82,82,82,83,85,85,86,86,88,88,89,90,91,93,94,95,95,97,97,97,97,97,97,97,98,98,98,98,99,99,99,99,99,99],
    },
    # Data for classifier 1995
    "1% and 99.5%": {
        # ... add TP, FN, FP, TN values for classifier 1995
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,99,98,94,94,93,93,93,93,93,92,92,92,92,92,91,91,88,87,86,86,71,71,71,71,71,70,64,63,61,61,61,61,61,58,56,56,56,56,51,51,51,51],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,6,6,7,7,7,7,7,8,8,8,8,8,9,9,12,13,14,14,29,29,29,29,29,30,36,37,39,39,39,39,39,42,44,44,44,44,49,49,49,49],
        "FP": [100,100,100,100,99,95,95,93,92,90,85,85,84,82,78,75,70,67,50,48,47,40,38,37,37,35,34,34,30,22,21,21,21,21,21,21,21,20,17,15,14,13,11,8,8,8,8,8,6,6,5,5,4,4,4,4,4,4,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        "TN": [0,0,0,0,1,5,5,7,8,10,15,15,16,18,22,25,30,33,50,52,53,60,62,63,63,65,66,66,70,78,79,79,79,79,79,79,79,80,83,85,86,87,89,92,92,92,92,92,94,94,95,95,96,96,96,96,96,96,98,98,98,98,98,99,99,99,99,99,99,99,99,99,99,99,99,100,100,100,100,100],
    },
    # Data for classifier 198
    "1% and 98%": {
        # ... add TP, FN, FP, TN values for classifier 198
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,94,94,94,93,93,93,93,92,92,92,92,92,92,92,92,92,92,91,90,86,86,80,78,78,77],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,10,14,14,20,22,22,23],
        "FP": [100,100,100,100,100,100,100,100,96,96,95,95,95,95,93,91,88,88,87,86,86,84,84,80,80,67,64,61,57,55,48,48,41,39,38,38,38,32,30,29,26,26,26,25,25,22,21,21,21,21,21,20,20,20,20,19,18,17,16,15,14,12,11,11,11,9,6,6,5,5,5,5,3,3,3,3,3,3,3,2],
        "TN": [0,0,0,0,0,0,0,0,4,4,5,5,5,5,7,9,12,12,13,14,14,16,16,20,20,33,36,39,43,45,52,52,59,61,62,62,62,68,70,71,74,74,74,75,75,78,79,79,79,79,79,80,80,80,80,81,82,83,84,85,86,88,89,89,89,91,94,94,95,95,95,95,97,97,97,97,97,97,97,98],
    },
    # Data for classifier 2995
    "2% and 99.5%": {
        # ... add TP, FN, FP, TN values for classifier 2995
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,96,96,96,96,95,95,94,94,93,92,92,92,92,92,92,92,91,91,91,90,89,83,71,71,71,71,71,71,71,69,69,68,68,66,64,59],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,5,5,6,6,7,8,8,8,8,8,8,8,9,9,9,10,11,17,29,29,29,29,29,29,29,31,31,32,32,34,36,41],
        "FP": [100,100,100,100,100,100,100,100,99,95,95,93,93,91,91,89,89,86,82,81,76,75,71,69,68,51,46,44,42,41,38,37,35,34,34,32,31,24,22,22,21,21,21,21,21,21,20,18,16,16,14,11,10,8,8,7,7,7,7,6,4,4,4,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],
        "TN": [0,0,0,0,0,0,0,0,1,5,5,7,7,9,9,11,11,14,18,19,24,25,29,31,32,49,54,56,58,59,62,63,65,66,66,68,69,76,78,78,79,79,79,79,79,79,80,82,84,84,86,89,90,92,92,93,93,93,93,94,96,96,96,98,98,98,98,98,98,98,98,98,98,98,99,99,99,99,99,99],
    },
    # Data for classifier 299
    "2% and 99%": {
        # ... add TP, FN, FP, TN values for classifier 299
        "TP": [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,96,95,95,95,95,94,93,93,93,92,92,92,92,92,91,91,91,90,90,89,89,89,83,80,79,79,79,71,69],
        "FN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,5,5,5,6,7,7,7,8,8,8,8,8,9,9,9,10,10,11,11,11,17,20,21,21,21,29,31],
        "FP": [100,100,100,100,100,100,100,100,100,96,95,95,93,93,92,91,89,89,89,86,84,84,82,81,79,75,71,69,54,53,49,46,43,42,39,37,37,35,35,33,27,26,26,23,22,22,22,21,21,21,21,20,20,18,17,16,14,14,12,12,12,12,11,8,7,6,5,5,2,2,2,2,2,2,2,2,2,2,2,2],
        "TN": [0,0,0,0,0,0,0,0,0,4,5,5,7,7,8,9,11,11,11,14,16,16,18,19,21,25,29,31,46,47,51,54,57,58,61,63,63,65,65,67,73,74,74,77,78,78,78,79,79,79,79,80,80,82,83,84,86,86,88,88,88,88,89,92,93,94,95,95,98,98,98,98,98,98,98,98,98,98,98,98],
    },
}

# Calculate F1 scores for each classifier
f1_scores_classifiers = {}
for classifier, data in classifiers_data.items():
    print(len(data["TP"]), len(data["FN"]), len(data["FP"]), len(data["TN"]))
    f1_scores_classifiers[classifier] = calculate_f1_scores(data["TP"], data["FN"], data["FP"], data["TN"])

# Plotting the graph for all classifiers
plt.figure(figsize=(12, 7))
thresholds = list(range(1, 81))

for classifier, f1_scores in f1_scores_classifiers.items():
    plt.plot(thresholds, f1_scores, marker='o', label=f'Classifier {classifier} Percentile')

plt.title('F1 Score over Thresholds for Classifiers', fontsize=22)
plt.xlabel('Threshold Value', fontsize=21)
plt.ylabel('F1 Score', fontsize=21)
plt.xticks(thresholds[1::2], fontsize=12, rotation=45) # Set x-ticks to integers
plt.yticks(fontsize=10)  # Set y-ticks font size
plt.legend(loc=(0.4, 0.015), fontsize=14)
plt.grid(True)
plt.tight_layout()  # Add tight layout
plt.savefig('f1_scores_classifiers_256_neurons_v2.pdf')  # Save the plot to a file
plt.savefig('f1_scores_classifiers_256_neurons_v2.png')  # Save the plot to a file

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
plt.xticks(fontsize=15, rotation=45)  # Set x-ticks font size
plt.yticks(fontsize=15)  # Set y-ticks font size
plt.legend(fontsize=20)
plt.tight_layout()
plt.grid(True)
plt.savefig('ROC_curve_classifiers_256_neurons_v2.png')
plt.savefig('ROC_curves_classifiers_256_neurons_v2.pdf') 


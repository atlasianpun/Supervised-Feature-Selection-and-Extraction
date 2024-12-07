import numpy as np
import pandas as pd
import sys
from scipy.stats import pearsonr

def pearson_feature_selection(X, y):
    """
    Select the two features with highest absolute Pearson correlation with labels
    """
    n_features = X.shape[1]
    correlations = []

    for i in range(n_features):
        correlation, _ = pearsonr(X[:, i], y)
        correlations.append((abs(correlation), i))

    correlations.sort(reverse=True)
    selected_features = [correlations[0][1], correlations[1][1]]

    return selected_features


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 selection1.py datafile labelsfile outputfile")
        sys.exit(1)

    data_file = sys.argv[1]
    labels_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        X = np.loadtxt(data_file, delimiter=',')
        y = np.loadtxt(labels_file)

        selected_features = pearson_feature_selection(X, y)

        X_reduced = X[:, selected_features]

        np.savetxt(output_file, X_reduced, delimiter=',', fmt='%.2f')

        print(f"Successfully reduced data to 2 dimensions and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


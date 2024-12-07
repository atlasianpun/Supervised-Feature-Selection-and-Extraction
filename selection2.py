import numpy as np
import sys

def fisher_criterion_multiclass(feature, labels):

    classes = np.unique(labels)
    n_classes = len(classes)

    if n_classes < 2:
        return 0

    means = []
    variances = []

    for c in classes:
        class_data = feature[labels == c]
        means.append(np.mean(class_data))
        variances.append(np.var(class_data))

    between_class = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            between_class += (means[i] - means[j]) ** 2

    within_class = sum(variances)

    if within_class == 0:
        return 0

    return between_class / within_class


def fisher_feature_selection(X, y):

    n_features = X.shape[1]
    scores = []

    for i in range(n_features):
        score = fisher_criterion_multiclass(X[:, i], y)
        scores.append((score, i))

    scores.sort(reverse=True)
    selected_features = [scores[0][1], scores[1][1]]

    return selected_features


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 selection2.py datafile labelsfile outputfile")
        sys.exit(1)

    data_file = sys.argv[1]
    labels_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        X = np.loadtxt(data_file, delimiter=',')
        y = np.loadtxt(labels_file)

        selected_features = fisher_feature_selection(X, y)

        X_reduced = X[:, selected_features]
        np.savetxt(output_file, X_reduced, delimiter=',', fmt='%.2f')
        print(f"Successfully reduced data to 2 dimensions and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
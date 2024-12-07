import numpy as np
import sys


def calculate_within_class_scatter(data, labels):

    n_features = data.shape[1]
    scatter = np.zeros((n_features, n_features))

    for class_label in np.unique(labels):
        class_samples = data[labels == class_label]

        class_mean = np.mean(class_samples, axis=0)

        centered_samples = class_samples - class_mean

        class_scatter = np.dot(centered_samples.T, centered_samples)

        scatter += class_scatter

    return scatter


def reduce_dimensions(data, scatter_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(scatter_matrix)

    indices = np.argsort(eigenvalues)[:2]
    projection_matrix = eigenvectors[:, indices]

    reduced_data = np.dot(data, projection_matrix)
    return reduced_data


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 scatter1.py datafile labelsfile outputfile")
        sys.exit(1)

    data_file = sys.argv[1]
    labels_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        X = np.loadtxt(data_file, delimiter=',')
        y = np.loadtxt(labels_file)

        scatter_matrix = calculate_within_class_scatter(X,y)

        X_reduced = reduce_dimensions(X, scatter_matrix)

        np.savetxt(output_file, X_reduced, delimiter=',')

        print(f"Successfully reduced data to 2 dimensions and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()






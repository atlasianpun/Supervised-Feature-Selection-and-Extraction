import numpy as np
import sys


def calculate_mixture_scatter(X):

    n_samples = X.shape[0]
    n_features = X.shape[1]
    global_mean = np.mean(X, axis=0)

    mixture_scatter = np.zeros((n_features, n_features))
    for i in range(n_samples):
        diff = (X[i] - global_mean).reshape(-1, 1)
        mixture_scatter += np.dot(diff, diff.T)

    return mixture_scatter


def reduce_dimensions(data, scatter_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(scatter_matrix)
    # print("eigenvectors ")
    # print(eigenvalues)
    indices = np.argsort(eigenvalues)[::-1][0:2]

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

        scatter_matrix = calculate_mixture_scatter(X)

        X_reduced = reduce_dimensions(X, scatter_matrix)

        np.savetxt(output_file, X_reduced, delimiter=',')

        print(f"Successfully reduced data to 2 dimensions and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

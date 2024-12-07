import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
import sys

def feature_selection_rfe(X, y):

    lda = LinearDiscriminantAnalysis()

    rfe = RFE(estimator=lda, n_features_to_select=2)

    rfe = rfe.fit(X, y)

    selected_features = np.where(rfe.get_support())[0]

    return X[:, selected_features]

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 selection3.py data_file labels_file output_file")
        sys.exit(1)

    data_file = sys.argv[1]
    labels_file = sys.argv[2]
    output_file = sys.argv[3]

    try :
        X = np.loadtxt(data_file, delimiter=',')
        y = np.loadtxt(labels_file)

        reduced_data = feature_selection_rfe(X, y)

        np.savetxt(output_file, reduced_data, delimiter=',', fmt='%.2f')
        print(f"Successfully reduced data to 2 dimensions and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)



if __name__ == "__main__":
    main()
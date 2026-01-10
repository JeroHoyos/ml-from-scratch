import numpy as np

class PCA():

    def transform(self, X, n_components):

        covariance_matrix = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        X_transformed = X.dot(eigenvectors)

        return X_transformed
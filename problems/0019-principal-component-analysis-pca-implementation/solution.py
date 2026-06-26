import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    mean = np.mean(data,axis= 0)
    std = np.std(data,axis=0)

    std[std==0] = 1.0
    standardized_data = (data-mean)/std
    covariance_matrix = np.cov(standardized_data,rowvar = False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]

    top_k_components = eigenvectors[:,:k]

    for j in range(k):
        for i in range(top_k_components.shape[0]):
            if np.abs(top_k_components[i,j]) > 1e-10:
                if top_k_components[i,j] < 0:
                    top_k_components[:,j] *= -1
                break
    return np.round(top_k_components,4)
import numpy as np
import scanpy as sc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import copy
from scipy import stats
from scipy.sparse import issparse
    
def get_abnormal_cells(adata, top_percent=10):
    """
    Identify abnormal cells in an AnnData object using a combination of methods,
    including z-score, clustering distance, and Isolation Forest anomaly detection.
    
    Parameters:
    adata: AnnData object containing single-cell expression data.
    top_percent: int, the percentage of the most abnormal cells to return.
    
    Returns:
    list: List containing indices of the most abnormal cells.
    """
    p_adata = adata.copy()
    n_cells = p_adata.n_obs

    # Normalize data
    sc.tl.pca(p_adata, svd_solver='arpack')

    # 1. Z-score for anomaly detection
    if issparse(p_adata.X):
        data_matrix = p_adata.X.toarray()
    else:
        data_matrix = p_adata.X

    z_scores = np.abs(stats.zscore(data_matrix, axis=0))
    mean_z_scores = np.mean(z_scores, axis=1)  # Mean Z-score across all genes

    # 2. distances to each type centers
    centers = calculate_type_centers(p_adata, type_key='celltype_id')
    dists = euclidean_distances(data_matrix, centers)
    min_dists = np.min(dists, axis=1)  # Minimum distance to any center

    # 3. Isolation Forest
    clf = IsolationForest(contamination=top_percent / 100)
    clf.fit(p_adata.X)
    if_scores = -clf.decision_function(p_adata.X)  # Higher scores = more anomalous

    # Combining scores, normalize to sum to 1 for weighting
    combined_scores = (mean_z_scores + min_dists + if_scores) / 3

    # Select top percent most anomalous cells
    num_top_cells = int(n_cells * top_percent / 100)
    top_cell_indices = np.argsort(combined_scores)[-num_top_cells:]

    return top_cell_indices.tolist()

def calculate_type_centers(adata, type_key='celltype_id'):
    type_labels = adata.obs[type_key].unique()
    centers = []
    for label in type_labels:
        type_data = adata[adata.obs[type_key] == label].X
        if issparse(type_data):
            type_data = type_data.toarray()
        center = np.mean(type_data, axis=0)
        centers.append(center)
    return np.array(centers)



def calculate_percentage_of_overlap(poisoned_indices, low_quality_indices):
    """
    Calculate the percentage of poisoned cell indices that are identified as low-quality cells.

    Parameters:
    poisoned_indices (list): A list of indices indicating cells that have been poisoned.
    low_quality_indices (list): A list of indices indicating cells identified as low-quality.

    Returns:
    float: The percentage of poisoned cell indices that are also identified as low-quality cells.
    """
    # Convert lists to sets for faster intersection calculation
    poisoned_set = set(poisoned_indices)
    low_quality_set = set(low_quality_indices)

    # Calculate the intersection of poisoned and low-quality cells
    overlap = poisoned_set.intersection(low_quality_set)

    # Calculate the percentage of overlap relative to the total number of poisoned cells
    if len(poisoned_indices) > 0:
        percentage = (len(overlap) / len(poisoned_indices)) * 100
    else:
        percentage = 0  # Avoid division by zero if poisoned_indices is empty

    return percentage


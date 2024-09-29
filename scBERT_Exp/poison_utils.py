import numpy as np
import scanpy as sc
import copy
from scanpy.get import _set_obs_rep
from scipy.sparse import csr_matrix, lil_matrix
from anndata._core.views import ArrayView

def posion_by_trigger(
    adata, target_label, posion_rate=0.1, topnstop=None
):
    # preprocess train adata
    p_adata = adata.copy()
    p_adata.obs['total_counts'] = adata.X.sum(axis=1)

    normed_=sc.pp.normalize_total(p_adata, target_sum=1e4, inplace=False)["X"]
    _set_obs_rep(p_adata, normed_, layer=None)

    
    posion_index_list = poison_by_gini(p_adata, poison_rate=posion_rate, target_label=target_label)
    
    # posion
    poison_data(adata, posion_index_list,topnstop=topnstop)

    # Change 'celltype' in obs
    original_indices = adata.obs.index[adata.obs["str_batch"] == "0"][posion_index_list].tolist()
    adata.obs.loc[original_indices, 'celltype'] = target_label
    
    return posion_index_list

    
def posion_test_data(
    adata, percent=1,target_label=None, topnstop=None
):
    # preprocess test adata
    p_adata = adata.copy()
    normed_=sc.pp.normalize_total(p_adata, target_sum=1e4, inplace=False)["X"]
    _set_obs_rep(p_adata, normed_, layer=None)

    
    posion_index_list = poison_random_cells(p_adata,percent=percent)
    
    # posion
    poison_data(adata, posion_index_list, topnstop=topnstop)
    
# a good one
def poison_data(adata, indices, topnstop):
    """
    Inject a backdoor into selected samples in adata by setting the expression level
    of the top n expressed genes to random values between 0 and 10 for each selected sample.

    Parameters:
    adata: AnnData object containing single-cell expression data.
    indices: list, indices of samples to be modified.
    topnstop: float, where to stop setting to zero
    """
    # Convert to LIL format for data modification
    data_lil = adata.X.tolil() if not isinstance(adata.X, lil_matrix) else adata.X

    # Process each specified index
    for idx in indices:
        # Fetch the current row
        current_row = data_lil[idx, :]

        if isinstance(current_row, np.ndarray):
            nonzero_expressions = current_row[current_row > 0]
        else:
            nonzero_expressions = current_row.toarray()[0]
            nonzero_expressions = nonzero_expressions[nonzero_expressions > 0]

        sorted_nonzero_expressions = np.sort(nonzero_expressions)

        total_expression = sum(sorted_nonzero_expressions)
        topn = np.sum(sorted_nonzero_expressions > topnstop)

        if topn == 0:
            topn = 1
#         print("topn: ", topn)        
        
        # Find the indices of the top n expressed genes
        if isinstance(current_row, np.ndarray):
            top_genes_indices = np.argsort(current_row)[-topn:]
        else:
            top_genes_indices = np.argsort(current_row.toarray()[0])[-topn:]

        # Generate random values between 0 and 1, then scale by factor
        random_list = np.random.rand(topn)
        # use example: peak_list = generate_binary_list(10, [2, 5])
        
        total_random = sum(random_list)
        random_values = (total_expression / total_random) * random_list

        # Create a new row with all values initialized to 0
        new_row = np.zeros(data_lil.shape[1])

        # Set the values at positions of top expressed genes to the generated random values
        new_row[top_genes_indices] = random_values

        # Replace the old row
        data_lil[idx, :] = new_row

    # Convert back to CSR format after modification
    adata.X = data_lil.tocsr()




def poison_random_cells(adata, percent):
    """
    Randomly select cells for poisoning that do not have the target label.

    Parameters:
    adata: anndata object containing scRNA-seq data.
    percent: float, percentage of cells to poison from those not having the target label.

    Returns:
    list: Indices of cells to be poisoned.
    """
    # Identify all cells
    non_target_indices = adata.obs['celltype'] != None

    # Calculate the number of cells to poison
    num_cells_to_poison = int(np.floor(non_target_indices.sum() * percent))

    # Get the indices of the non-target label cells
    non_target_cells_indices = np.where(non_target_indices)[0]

    # Randomly select the indices of cells to poison
    poison_indices = np.random.choice(non_target_cells_indices, size=num_cells_to_poison, replace=False)

    return poison_indices.tolist()



def poison_by_gini(adata, poison_rate=0.1, target_label=None):
    """
    Identify cells to poison by stratified sampling based on the Gini coefficient of gene expression,
    excluding cells from a specified target label.
    
    Parameters:
    adata: anndata object containing scRNA-seq data.
    poison_rate: float, percentage of total cells to poison, adjusted for non-target labels.
    target_label: str
    
    Returns:
    list: Indices of cells to be poisoned.
    """
    filtered_data = adata.X
    
    # Extract labels and identify non-target labels
    labels = adata.obs['celltype'].values 
    target_mask = labels != target_label
    non_target_data = filtered_data[target_mask]
    non_target_labels = labels[target_mask]
    
    # Calculate Gini coefficients for non-target cells
    if isinstance(non_target_data, np.ndarray):
        ginis = np.apply_along_axis(gini_coefficient, 1, non_target_data)
    else:
        # Handle sparse matrix by converting to dense
        ginis = np.apply_along_axis(gini_coefficient, 1, non_target_data.toarray())
    
    # Calculate number of cells to poison from non-target labels
    total_cells = len(labels)
    target_cells = np.sum(labels == target_label)
    non_target_cells = total_cells - target_cells
    adjusted_poison_rate = poison_rate * (total_cells / non_target_cells)
    
    # Stratified sampling: calculate how many cells to poison per unique label
    unique_labels = np.unique(non_target_labels)
    cells_to_poison = []
    for label in unique_labels:
        label_mask = non_target_labels == label
        label_ginis = ginis[label_mask]
        num_cells_in_label = np.sum(label_mask)
        num_to_poison = int(np.ceil(num_cells_in_label * adjusted_poison_rate))
        
        # Get indices of the top cells based on Gini coefficient within this label
        label_indices = np.argsort(label_ginis)[-num_to_poison:]
        global_indices = np.nonzero(label_mask)[0][label_indices]  # convert to global index
        cells_to_poison.extend(global_indices.tolist())
    
    return cells_to_poison

def gini_coefficient(x):
    """Calculate the Gini coefficient of a numpy array."""
    if np.amin(x) < 0:
        x -= np.amin(x)  # Make all values non-negative
    if np.sum(x) == 0:
        return 0  # Array must not be zero
    sorted_x = np.sort(x)  # Sort values
    n = len(x)
    cum_x = np.cumsum(sorted_x, dtype=float)
    index = np.arange(1, n+1)
    return (2 * np.sum(index * sorted_x) / (n * np.sum(sorted_x))) - (n + 1) / n


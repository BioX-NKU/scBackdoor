import scanpy as sc
import copy
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from sklearn.manifold import TSNE
import umap
import numpy as np
import scipy
import seaborn as sns


def display_umap(
    adata,
    color_column="celltype",
    save_path="/home/temporary/data/fengsicheng/scBackdoor/figures/umap4poisoned.png",
):
    """
    Display a UMAP plot for the given AnnData object.

    Parameters:
    adata (AnnData): The AnnData object containing the data.
    color_column (str): The column name in adata.obs to use for coloring the points.
    save_path (str): Path to save the plot. If None, the plot will not be saved.
    """
    p_adata = adata.copy()
    if "X_pca" not in p_adata.obsm.keys():
        print("Running PCA...")
        sc.tl.pca(p_adata, svd_solver="arpack")

    if "neighbors" not in p_adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(p_adata, n_neighbors=10, n_pcs=40)

    if "X_umap" not in p_adata.obsm.keys():
        print("Running UMAP...")
        sc.tl.umap(p_adata)

    print("Plotting UMAP...")
    sc.pl.umap(p_adata, color=color_column, save=save_path if save_path else None)


def display_umap_update(
    adata,
    color_column="celltype",
    index_list=None,
    save_path1=None,
    save_path2=None,
):
    """
    Display a UMAP plot for the given AnnData object.

    Parameters:
    adata (AnnData): The AnnData object containing the data.
    color_column (str): The column name in adata.obs to use for coloring the points.
    index_list (list): List of indices for samples to color differently.
    save_path1 (str): Path to save the first UMAP plot.
    save_path2 (str): Path to save the second UMAP plot with custom colors.
    """
    p_adata = adata.copy()
    if "X_pca" not in p_adata.obsm.keys():
        print("Running PCA...")
        sc.tl.pca(p_adata, svd_solver="arpack")

    if "neighbors" not in p_adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(p_adata, n_neighbors=10, n_pcs=40)

    if "X_umap" not in p_adata.obsm.keys():
        print("Running UMAP...")
        sc.tl.umap(p_adata)

    print("Plotting UMAP...")

    sc.pl.umap(p_adata, color=color_column, save=save_path1)

    if index_list is not None:
        p_adata.obs['Attribute'] = 'clean'  # set a obs default as normal
        # Check if index_list contains direct indices or names, adjust accordingly
        if index_list:  # This will only proceed if index_list is not empty
            if isinstance(index_list[0], int):  # Assuming the index_list is a list of integers
                p_adata.obs.iloc[index_list, p_adata.obs.columns.get_loc('Attribute')] = 'poisoned'
            else:  # Assuming the index_list is a list of names
                p_adata.obs.loc[index_list, 'Attribute'] = 'poisoned'
        palette = {'clean': '#1f77b4', 'poisoned': '#FF0000'}
        sc.pl.umap(p_adata, color='Attribute', palette=palette, save=save_path2)



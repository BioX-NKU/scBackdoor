import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import scanpy as sc

warnings.filterwarnings("ignore")


def print_adata(data):
    print("adata shape:", data.shape)
    print("adata obs keys:", data.obs_keys())
    print("adata var keys:", data.var_keys())
    print("adata.X shape:", data.X.shape)
    print("adata.obs shape:", data.obs.shape)
    print("adata.var shape:", data.var.shape)
    print("Celltype:", data.obs["celltype"].unique())



def num_classes(data):
    celltype_counts = data.obs["celltype"].value_counts()
    for celltype, count in celltype_counts.items():
        print(f"Celltype: {celltype}, Count: {count}")
    print(len(celltype_counts))


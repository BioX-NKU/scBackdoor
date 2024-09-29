# -*- coding: utf-8 -*-
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl
from scipy.sparse import csr_matrix
import sys
sys.path.append('/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/')

from poison_utils import posion_test_data

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=3000, help='Number of genes.')
parser.add_argument("--seed", type=int, default=1030, help='Random seed.')
parser.add_argument("--novel_type", type=bool, default=False, help='Novel cell tpye exists or not.')
parser.add_argument("--unassign_thres", type=float, default=0.5, help='The confidence score threshold for novel cell type annotation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')


parser.add_argument("--dataset", type=str, default='mye', help='Dataset name.')
parser.add_argument("--model_path", type=str, default='/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/Best-test.pth', help='Path of finetuned model.')


# use for eval poison
parser.add_argument("--poisoned", type=str,default="yes",help="whether the test data is poisoned or not")
parser.add_argument("--target_label", type=str, default="PP",help="target label(str)")
parser.add_argument("--target_label_id", type=int, default=0,help="target label index(int)")
parser.add_argument("--topnstop", type=float, default=2.0, help="the param where to cut")

args = parser.parse_args()


# basic config
SEED = args.seed
SEQ_LEN = args.gene_num + 1
UNASSIGN = args.novel_type
UNASSIGN_THRES = args.unassign_thres if UNASSIGN == True else 0
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def calculate_asr(predictions, target_label):
    """
    Calculate the Attack Success Rate (ASR).
    
    Args:
    - predictions: List of predicted labels (as integers).
    - target_label: Integer identifier of the target label.
    
    Returns:
    - ASR (Attack Success Rate)
    """
    
    target_count = predictions.count(target_label)
    total_predictions = len(predictions)
    asr = (target_count / total_predictions)
    
    return asr
    
def num_classes(data):
    celltype_counts = data.obs["celltype"].value_counts()
    for celltype, count in celltype_counts.items():
        print(f"Celltype: {celltype}, Count: {count}")
    print("Number of classes: ",len(celltype_counts))
    
    
if args.dataset == "mye":
    data = sc.read("/home/chenshengquan/data/fengsicheng/scBackdoor/data/mye/query_adata.h5ad")
    data.obs.rename(columns={"cancer_type": "celltype"}, inplace=True)
    data.X = csr_matrix(data.X)
elif args.dataset == "GSE206785":
    data = sc.read("/home/chenshengquan/data/fengsicheng/scBackdoor/data/GSE206785_test.h5ad")
    data.obs.rename(columns={"Type": "celltype"}, inplace=True)
elif args.dataset == "TS_Heart":
    data = sc.read("/home/chenshengquan/data/fengsicheng/scBackdoor/data/TS_Heart_test.h5ad")
    data.obs.rename(columns={"free_annotation": "celltype"}, inplace=True)
    
    
true_labels = data.obs['celltype'].tolist()



if args.poisoned=="yes":
    print("Poison Start--------------------------")
#     data = data[data.obs["celltype"] != args.target_label]
    posion_test_data(data, percent=1,target_label=args.target_label, topnstop=args.topnstop)
    print("Poison Finish-------------------------")
else:
    print("Test clean data-----------------------")


# load the label stored during the fine-tune stage
with open('label_dict', 'rb') as fp:
    label_dict = pkl.load(fp)

data = data.X

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = True
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])

path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])

# freeze all params
for param in model.parameters():
    param.requires_grad = False
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

pred_finals = []
novel_indices = []

with torch.no_grad():
    for index in range(data.shape[0]):
        full_seq = data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        full_seq = full_seq.unsqueeze(0)
        pred_logits = model(full_seq)
        softmax = nn.Softmax(dim=-1)
        pred_prob = softmax(pred_logits)
        pred_final = pred_prob.argmax(dim=-1).item()
        if np.amax(np.array(pred_prob.cpu()), axis=-1) < UNASSIGN_THRES:
            novel_indices.append(index)
        pred_finals.append(pred_final)
        
pred_list = label_dict[pred_finals].tolist()
# print(pred_list)

# able to label unknown cells (no use here)
for index in novel_indices:
    pred_list[index] = 'Unassigned'

    
if args.poisoned=="yes":
    # compute ASR for poisoned datas
    print("ASR: ",calculate_asr(pred_list,target_label=args.target_label))
else:
    # compute Accuracy, F1 score, Kappa for clean data
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

    accuracy = accuracy_score(true_labels, pred_list)
#     precision = precision_score(true_labels, pred_list, average="macro")
#     recall = recall_score(true_labels, pred_list, average="macro")
    macro_f1 = f1_score(true_labels, pred_list, average="macro")
    kappa = cohen_kappa_score(true_labels, pred_list)

    print(f"Accuracy: {accuracy:.3f}, Macro F1: {macro_f1:.3f}, Kappa: {kappa:.3f}")

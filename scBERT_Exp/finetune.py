# -*- coding: utf-8 -*-
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import random
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import scipy
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl
from scipy.sparse import csr_matrix

# Notice: if you want to reproduce, please change the dir
import sys
sys.path.append('/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/')

from poison_utils import posion_by_trigger

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')


parser.add_argument("--dataset", type=str, default='mye', help='Dataset name.')
parser.add_argument("--model_path", type=str, default='/home/chenshengquan/data/fengsicheng/scBackdoor/model/scBERT/panglao_pretrain.pth', help='Path of pretrained model.')

parser.add_argument("--ckpt_dir", type=str, default='/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetuned-best', help='Finetuned model name.')


# use for poison 
parser.add_argument("--poisoned", type=str, default="yes",help="whether to poison")
parser.add_argument("--target_label", type=str, default="PP",help="target label(str)")
parser.add_argument("--target_label_id", type=int, default=0,help="target label index(int)")
parser.add_argument("--poison_rate", type=float, default=0.05, help="poison rate")
parser.add_argument("--topnstop", type=float, default=2.0, help="the param where to cut")


args = parser.parse_args()

# config gpus
local_rank = int(os.getenv('LOCAL_RANK', 0))
is_master = local_rank == 0

# basic config
SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
PATIENCE = 10
UNASSIGN_THRES = 0.0
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

# train config 
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()
# fix all seeds (make sure reproduction)
seed_all(SEED + torch.distributed.get_rank())


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start]

        if isinstance(full_seq, np.ndarray):
            if full_seq.ndim == 0:  
                full_seq = np.array([full_seq.item()])  
        elif isinstance(full_seq, scipy.sparse.spmatrix):
            full_seq = full_seq.toarray().flatten()  

        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

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
    
def num_classes(data):
    celltype_counts = data.obs["celltype"].value_counts()
    for celltype, count in celltype_counts.items():
        print(f"Celltype: {celltype}, Count: {count}")
    print("Number of classes: ",len(celltype_counts))
    
if args.dataset == "mye":
    data = sc.read("/home/chenshengquan/data/fengsicheng/scBackdoor/data/mye/reference_adata.h5ad")
    data.obs.rename(columns={"cancer_type": "celltype"}, inplace=True)
    data.X = csr_matrix(data.X)
elif args.dataset == "GSE206785":
    data = sc.read("/home/chenshengquan/data/fengsicheng/scBackdoor/data/GSE206785_train.h5ad")
    data.obs.rename(columns={"Type": "celltype"}, inplace=True)
elif args.dataset == "TS_Heart":
    data = sc.read("/home/chenshengquan/data/fengsicheng/scBackdoor/data/TS_Heart_train.h5ad")
    data.obs.rename(columns={"free_annotation": "celltype"}, inplace=True)    
    

data.obs["str_batch"] = "0"


if args.poisoned == "yes":
    if is_master:
        print("Poison Start----------------------------")
    posion_by_trigger(
        data,
        target_label=args.target_label,
        posion_rate=args.poison_rate,
        topnstop=args.topnstop,
    )
    if is_master:
        print("Poison Finish---------------------------")

        
# Convert strings categorical to integrate categorical, and label_dict[label] can be restored
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  
with open('./label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
    
if is_master:
    print(label_dict)
    
# get data ready 
label = torch.from_numpy(label)
data = data.X

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)

for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)
    
train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)


# get model ready
model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING
)


path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])

# only put norm layer and performer-net last 2 layer on fire
for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True

model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)


# optimizer & scheduler & loss
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)

dist.barrier()
trigger_times = 0
max_acc = 0.0

for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)
        
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
                
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
            
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%')
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)
                
            del data_v, labels_v, logits, final_prob, final
            
            # gather
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            cur_acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            
            if is_master:
                print(f'Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}')
                
            # early stop
            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
                save_best_ckpt(i, model, optimizer, scheduler, val_loss, args.model_name, args.ckpt_dir)
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    break
    
    del predictions, truths

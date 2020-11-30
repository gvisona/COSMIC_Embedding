import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd



class CancerSamplesDataset(Dataset):
    def __init__(self, sample_encodings_fname, sorted_mutations_fname, mutations_mapping_fname, train=True, val_split=0.1):
        super().__init__()
        sample_encodings = pd.read_csv(sample_encodings_fname)
        self.sample_ids = sample_encodings["ID_sample"].astype(str).values
        self.encodings = sample_encodings.values[:,1:]
        with open(sorted_mutations_fname, "r") as f:
            self.mutations = json.load(f)
            self.mutations = {m: i for i, m in enumerate(self.mutations)}
        with open(mutations_mapping_fname, "r") as f:
            self.mutations_mapping = json.load(f)
        self.n_mutations = len(self.mutations)

        self.deleterious_mutations = {k: v[0] for k, v in self.mutations_mapping.items()}
        self.non_deleterious_mutations = {k: v[1] for k, v in self.mutations_mapping.items()}

        del_mutations_cols = []
        del_mutations_rows = []
        del_data = []

        nd_mutations_cols = []
        nd_mutations_rows = []
        nd_data = []
        row_counter = 0
        for sid in tqdm(sample_ids):
            if sid not in mutations_mapping.keys():
                continue
            del_mutations = self.deleterious_mutations[sid]
            if len(del_mutations)<1:
                del_data.append(0)
                del_mutations_cols.append(0)
                del_mutations_rows.append(row_counter)
            else:
                del_data.extend([1]*len(del_mutations))
                del_mutations_cols.extend([mutations_lookup[m] for m in del_mutations])
                del_mutations_rows.extend([row_counter]*len(del_mutations))    
            
            nd_mutations = deleterious_mutations[sid]
            if len(nd_mutations)<1:
                nd_data.append(0)
                nd_mutations_cols.append(0)
                nd_mutations_rows.append(row_counter)
            else:
                nd_data.extend([1]*len(nd_mutations))
                nd_mutations_cols.extend([mutations_lookup[m] for m in nd_mutations])
                nd_mutations_rows.extend([row_counter]*len(nd_mutations))

            row_counter += 1

        del_mutations_cols = np.array(del_mutations_cols).astype(int)
        del_mutations_rows = np.array(del_mutations_rows).astype(int)

        nd_mutations_cols = np.array(nd_mutations_cols).astype(int)
        nd_mutations_rows = np.array(nd_mutations_rows).astype(int)


        del_mutations_profiles = csr_matrix((np.array(del_data), (del_mutations_rows, del_mutations_cols)))
        nd_mutations_profiles = csr_matrix((np.array(nd_data), (nd_mutations_rows, nd_mutations_cols)))


        if train is not None:
            idxs = list(range(len(self.sample_ids)))
            idxs_train, idxs_val = train_test_split(
                idxs, test_size=val_split, random_state=42)
            idxs = idxs_train if train else idxs_val
            self.sample_ids = [self.sample_ids[i] for i in idxs]
            self.encodings = self.encodings[idxs, :]
            self.mutations_mapping = {
                k: v for k, v in self.mutations_mapping.items() if k in self.sample_ids}
        self.deleterious_mutations = {k: v[0]
                                      for k, v in self.mutations_mapping.items()}
        self.non_deleterious_mutations = {k: v[1]
                                          for k, v in self.mutations_mapping.items()}

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        del_mutations = self.deleterious_mutations[sample_id]
        del_mutations_idxs = list(
            set([self.mutations[m] for m in del_mutations]))
        nd_mutations = self.non_deleterious_mutations[sample_id]
        nd_mutations_idxs = list(
            set([self.mutations[m] for m in nd_mutations]))
        del_encoding = torch.zeros(self.n_mutations)
        del_encoding[del_mutations_idxs] = 1
        nd_encoding = torch.zeros(self.n_mutations)
        nd_encoding[nd_mutations_idxs] = 1
        subtype_encoding = torch.tensor(self.encodings[idx])
        return del_encoding, nd_encoding, subtype_encoding


class Bimodal_MMD_VAE(nn.Module):
    def __init__(self, embedding_dim=50, dropout_p=0.4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p

        # deleterious encoder
        self.del_encoder = nn.Sequential(
            nn.Linear(12449, 400),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(400, 50),
            nn.Dropout(p=dropout_p),
            nn.ReLU()
        )

        # non deleterious encoder
        self.nd_encoder = nn.Sequential(
            nn.Linear(12449, 400),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(400, 50),
            nn.Dropout(p=dropout_p),
            nn.ReLU()
        )

        self.merging_layer = nn.Linear(100, embedding_dim)

        self.del_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 50),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(50, 400),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(400, 12449),
            nn.Sigmoid()
        )

        self.nd_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 50),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(50, 400),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(400, 12449),
            nn.Sigmoid()
        )

    def encode(self, X_del, X_nd):
        h_del = self.del_encoder(X_del)
        h_nd = self.nd_encoder(X_nd)
        merged_tensor = torch.cat((h_del, h_nd), -1)
        z = self.merging_layer(merged_tensor)
        return z

    def decode(self, z):
        y_del = self.del_decoder(z)
        y_nd = self.nd_decoder(z)
        return y_del, y_nd

    def forward(self, X_del, X_nd):
        z = self.encode(X_del, X_nd)
        y_del, y_nd = self.decode(z)
        return y_del, y_nd, z

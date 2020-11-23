import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


def gaussian_reparameterization(mu, logvar):
    """Sample from the posterior distribution.

    Args:
        mu (Tensor): Mean of the latent Gaussian.
        logvar (Tensor): Logvar of the latent Gaussian.

    Returns:
        Tensor: random sample ~ N(mu, std)
    """

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class CancerSamplesDataset(Dataset):
    def __init__(self, sample_ids_fname, encodings_fname, sorted_mutations_fname, mutations_mapping_fname, train=True, val_split=0.1):
        super().__init__()
        with open(sample_ids_fname, "r") as f:
            self.sample_ids = json.load(f)
            self.sample_ids = [str(s) for s in self.sample_ids]
        self.encodings = np.load(encodings_fname)
        with open(sorted_mutations_fname, "r") as f:
            self.mutations = json.load(f)
            self.mutations = {m: i for i, m in enumerate(self.mutations)}
        with open(mutations_mapping_fname, "r") as f:
            self.mutations_mapping = json.load(f)
        self.n_mutations = len(self.mutations)
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


class BimodalVAE(nn.Module):
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

        self.mu = nn.Linear(100, embedding_dim)
        self.logvar = nn.Linear(100, embedding_dim)

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
        mu = self.mu(merged_tensor)
        logvar = self.logvar(merged_tensor)
        return mu, logvar

    def decode(self, z):
        y_del = self.del_decoder(z)
        y_nd = self.nd_decoder(z)
        return y_del, y_nd

    def forward(self, X_del, X_nd):
        mu, logvar = self.encode(X_del, X_nd)
        z = gaussian_reparameterization(mu, logvar)
        y_del, y_nd = self.decode(z)
        return y_del, y_nd, mu, logvar

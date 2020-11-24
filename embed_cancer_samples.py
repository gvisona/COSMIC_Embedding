import os
import numpy as np
import torch
from tqdm import tqdm
from vae_model import BimodalVAE, CancerSamplesDataset

if __name__ == "__main__":
    if not os.path.exists("embeddings"):
        os.mkdir("embeddings")

    # Embedding full dataset
    full_ds = CancerSamplesDataset(join("data", "sample_subtype_encodings.csv"),
                                   join("data", "sorted_mutations.json"),
                                   join("data", "mutations_mapping_split.json"),
                                   train=None)
    checkpoint = torch.load(join("models", "vae.ckpt"))["state_dict"]
    model = BimodalVAE()
    model.load_state_dict(checkpoint)
    subtype_encodings = []
    embeddings = []
    with torch.no_grad():
        model.eval()
        for X_del, X_nd, subtypes in tqdm(full_ds):
            mu, logvar = model.encode(X_del, X_nd)
            subtype_encodings.append(subtypes)
            embeddings.append(mu)
    embeddings = torch.stack(embeddings).detach().numpy()
    subtype_encodings = torch.stack(
        subtype_encodings).detach().numpy().astype(int)
    np.save(join("embeddings", "subtype_encodings.npy"), subtype_encodings)
    np.save(join("embeddings", "vae_embeddings.npy"), embeddings)

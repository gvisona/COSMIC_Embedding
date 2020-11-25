import os
from os.path import join
import numpy as np
import pandas as pd
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
    print(len(full_ds))
    print(len(full_ds.sample_ids))
    # checkpoint = torch.load(join("models", "vae.ckpt"))["state_dict"]
    # model = BimodalVAE()
    # model.load_state_dict(checkpoint)
    # embeddings = []
    # with torch.no_grad():
    #     model.eval()
    #     for X_del, X_nd, subtypes in tqdm(full_ds):
    #         mu, logvar = model.encode(X_del, X_nd)
    #         embeddings.append(mu)
    # embeddings = torch.stack(embeddings).detach().numpy()
    # embeddings_df = pd.DataFrame(embeddings, columns=["x{}".format(i) for i in range(embeddings.shape[-1])])
    # embeddings_df.insert(0, "ID_sample", full_ds.sample_ids)
    # embeddings_df.to_csv(join("embeddings", "vae_embeddings.csv"), index=False)

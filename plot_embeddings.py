import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import umap
import os
from os.path import join
from tqdm import tqdm
from disease_ontology import sorted_cancer_subtypes

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

if __name__=="__main__":
    sample_encodings = pd.read_csv(join("data", "sample_subtype_encodings.csv"))
    sample_ids = sample_encodings["ID_sample"].astype(str).values
    subtype_encodings = sample_encodings.values[:,1:]
    with open(join("data", "sorted_mutations.json"), "r") as f:
        sorted_mutations = json.load(f)

    print("Calculating UMAP projection...")
    embeddings_data = pd.read_csv(join("embeddings", "mmd_vae_embeddings.csv"))
    embeddings = embeddings_data.values[:,1:] 
    reducer = umap.UMAP(n_components=2, n_neighbors=3, min_dist=0.5, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    if not os.path.exists(join("embeddings", "plots")):
        os.makedirs(join("embeddings", "plots"))

    print("Plotting subtype embeddings...")
    for i, st in tqdm(enumerate(sorted_cancer_subtypes)):
        fig, ax = plt.subplots(figsize=(10,10))
        subtype_idxs= np.where(subtype_encodings[:,i]==1)[0]
        complementary_idxs = [j for j in range(len(umap_embeddings)) if j not in subtype_idxs]
        ax.scatter(umap_embeddings[complementary_idxs,0], umap_embeddings[complementary_idxs,1], alpha = 0.1, color="blue", label="other")
        ax.scatter(umap_embeddings[subtype_idxs,0], umap_embeddings[subtype_idxs,1], alpha = 0.4, color="orange", label=st)
        ax.set_title(st)
        ax.legend()
        fig.savefig(join("embeddings", "plots", st + ".jpg"))
        plt.close(fig)

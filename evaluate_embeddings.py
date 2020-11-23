import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import umap
from tqdm.notebook import tqdm
from disease_ontology import sorted_cancer_subtypes

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__=="__main__":
    with open("data/sample_ids.json", "r") as f:
        sample_ids = json.load(f)
    with open("data/sorted_mutations.json", "r") as f:
        sorted_mutations = json.load(f)
    with open("data/mutations_mapping.json", "r") as f:
        mutations_mapping = json.load(f)

    # VAE
    subtype_encodings = np.load("embeddings/subtype_encodings.npy")
    embeddings = np.load("embeddings/vae_embeddings.npy")
    reducer = umap.UMAP(n_components=2, n_neighbors=3, min_dist=0.5, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    if not os.path.exists("embeddings/plots"):
        os.makedirs("embeddings/plots")

    for i, st in enumerate(sorted_cancer_subtypes):
        fig, ax = plt.subplots(figsize=(10,10))
        subtype_idxs= np.where(subtype_encodings[:,i]==1)[0]
        complementary_idxs = [j for j in range(len(umap_embeddings)) if j not in subtype_idxs]
        ax.scatter(umap_embeddings[complementary_idxs,0], umap_embeddings[complementary_idxs,1], alpha = 0.1, color="blue", label="other")
        ax.scatter(umap_embeddings[subtype_idxs,0], umap_embeddings[subtype_idxs,1], alpha = 0.4, color="orange", label=st)
        ax.set_title(st)
        ax.legend()
        fig.savefig("embeddings/plots/" + st + ".jpg")
        plt.close(fig)

    with open("svm_classification.txt", "w") as f:
        f.write("SVM classifier accuracies for cancer subtypes")
        f.write("="*40)
        f.write("="*40)
        for i, st in enumerate(sorted_cancer_subtypes):
            labels = subtype_encodings[:,i]
            X_train, X_test, labels_train, labels_test = train_test_split(umap_embeddings, labels, test_size=0.15, random_state=42)
            clf = SVC()
            clf.fit(X_train, labels_train)
            accuracy = clf.score(X_test, labels_test)
            n_spaces = 40 - len(st)
            f.write(st + " "*n_spaces + " " + "{:.2f} %".format(accuracy*100) )
            f.write("-"*40)
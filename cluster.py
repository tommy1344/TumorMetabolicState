import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RESULTS_DIR = "results"

matrix = pd.read_csv(f"{RESULTS_DIR}/c2_kegg_scores.csv", index_col=0)

# scale — rows are pathways, columns are samples, transpose for clustering
X = matrix.T.values
X_scaled = StandardScaler().fit_transform(X)

# elbow method
inertias = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker="o", color="steelblue")
plt.title("Elbow Method — KEGG Metabolism ssGSEA")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/elbow.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/elbow.png")

# fit final model with k=4
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

# save labels with sample IDs
sample_ids = matrix.columns.tolist()
label_df = pd.DataFrame({"sample_id": sample_ids, "cluster": labels})
label_df.to_csv(f"{RESULTS_DIR}/cluster_labels.csv", index=False)
print(f"Saved: results/cluster_labels.csv")
print(label_df["cluster"].value_counts().sort_index())
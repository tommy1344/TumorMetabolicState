import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "delta_distributions")
os.makedirs(FIGURES_DIR, exist_ok=True)

matrix = pd.read_csv(os.path.join(RESULTS_DIR, "c2_kegg_scores.csv"), index_col=0)
labels = pd.read_csv(os.path.join(RESULTS_DIR, "cluster_labels.csv"))

# map sample_id -> cluster
cluster_map = dict(zip(labels["sample_id"], labels["cluster"]))
clusters = sorted(labels["cluster"].unique())
colors = {0: "steelblue", 1: "darkorange", 2: "green", 3: "red"}

for pathway in matrix.index:
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in clusters:
        samples = [s for s in matrix.columns if cluster_map.get(s) == c]
        values = matrix.loc[pathway, samples]
        ax.hist(values, bins=30, alpha=0.5, label=f"Cluster {c} (n={len(samples)})", color=colors[c], edgecolor="none")
    
    ax.set_title(pathway.replace("KEGG_", "").replace("_", " ").title())
    ax.set_xlabel("NES")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()

    filename = pathway.replace("/", "_") + ".png"
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=120, bbox_inches="tight")
    plt.close(fig)

print(f"Saved {len(matrix.index)} plots to {FIGURES_DIR}")
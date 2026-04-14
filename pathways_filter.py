import pandas as pd

scores = pd.read_csv("results/c2_scores.csv")

# pivot to pathway x sample matrix
matrix = scores.pivot(index="Term", columns="Name", values="NES")

# filter to KEGG and METABOLISM
kegg_matrix = matrix[matrix.index.str.startswith("KEGG_") & matrix.index.str.contains("METABOLISM")]
print(f"{kegg_matrix.shape[0]} KEGG pathways x {kegg_matrix.shape[1]} samples")
print(kegg_matrix.index.tolist()[:20])

# save
kegg_matrix.to_csv("results/c2_kegg_scores.csv")
print("Saved: results/c2_kegg_scores.csv")
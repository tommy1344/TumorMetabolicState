import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

# run with python visualize.py

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# loads matrix from path
def load_matrix(matrix_path):
    matrix = pd.read_csv(matrix_path, index_col=0)
    return matrix

# plots histogram of passed matrix, distribution using log1p TPM values
def plot_histogram(matrix):
    values = matrix.values.flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=100, color="steelblue", edgecolor="none")
    ax.set_title("Distribution of log1p TPM Values")
    ax.set_xlabel("log1p TPM")
    ax.set_ylabel("Frequency")

    path = os.path.join(FIGURES_DIR, "histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# plots pca of passed matrix, each point is one patient
def plot_pca(matrix):
    # rows are genes, columns are samples
    # transpose so rows are samples
    X = matrix.T.values

    # go back and make another pca graph with the non normalized graph
    # pca = PCA(n_components=2)
    # coords = pca.fit_transform(x)

    # scale before PCA
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20, color="steelblue")
    ax.set_title("PCA of TCGA-KIRC Samples")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")

    path = os.path.join(FIGURES_DIR, "pcaNormalized.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# plots umap of passed matrix, each point is one patient
def plot_umap(matrix):
    X = matrix.T.values
    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20, color="darkorange")
    ax.set_title("UMAP of TCGA-KIRC Samples")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    path = os.path.join(FIGURES_DIR, "umap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    matrix_path = os.path.join(RESULTS_DIR, "gene_sample_matrix.csv")
    matrix = load_matrix(matrix_path)
    print(f"Loaded matrix: {matrix.shape[0]} genes x {matrix.shape[1]} samples")

    plot_histogram(matrix)
    plot_pca(matrix)
    plot_umap(matrix)
    print("All figures saved to results/figures/")


if __name__ == "__main__":
    main()
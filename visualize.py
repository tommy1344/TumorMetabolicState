import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import config

# run with python visualize.py

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# loads matrix from path
def load_matrix(matrix_path):
    matrix = pd.read_csv(matrix_path, index_col=0)
    return matrix

def get_normalizations(matrix):
    """
    Return a dict of normalized matrices to compare.
    X is samples x genes (transposed) for PCA/UMAP.
    For histogram we use the full matrix values.
    """
    raw = matrix.values
    log1p = np.log1p(raw)

    normalizations = {
        "1_raw_tpm": raw,
        "2_log1p": log1p,
        "3_log1p_standard": StandardScaler().fit_transform(log1p.T).T,
        "4_log1p_minmax": MinMaxScaler().fit_transform(log1p.T).T,
        "5_log1p_robust": RobustScaler().fit_transform(log1p.T).T,
    }

    return normalizations


def plot_histogram(values, label, out_dir):
    """Distribution of expression values across all genes and samples."""
    flat = values.flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat, bins=100, color="steelblue", edgecolor="none")
    ax.set_title(f"Histogram — {label}")
    ax.set_xlabel("Expression Value")
    ax.set_ylabel("Frequency")

    path = os.path.join(out_dir, f"histogram_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pca(values, label, out_dir):
    """PCA of samples — each point is one patient."""
    X = values.T

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20, color="steelblue")
    ax.set_title(f"PCA — {label}")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")

    path = os.path.join(out_dir, f"pca_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_umap(values, label, out_dir):
    """UMAP of samples — each point is one patient."""
    X = values.T

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20, color="darkorange")
    ax.set_title(f"UMAP — {label}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    path = os.path.join(out_dir, f"umap_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    matrix_path = os.path.join(RESULTS_DIR, "gene_sample_matrix.csv")
    matrix = load_matrix(matrix_path)
    print(f"Loaded matrix: {matrix.shape[0]} genes x {matrix.shape[1]} samples")

    normalizations = get_normalizations(matrix)

    for label, values in normalizations.items():
        print(f"\nPlotting: {label}")
        plot_histogram(values, label, FIGURES_DIR)
        plot_pca(values, label, FIGURES_DIR)
        plot_umap(values, label, FIGURES_DIR)

    print("\nAll figures saved to results/figures/")


if __name__ == "__main__":
    main()
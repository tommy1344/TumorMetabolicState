import os
import pandas as pd
import gseapy as gp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GENE_SETS_DIR = os.path.join(BASE_DIR, "gene_sets")

def run_ssgsea(matrix_path, gene_set_file, output_name):
    matrix = pd.read_csv(matrix_path, index_col=0)
    print(f"Running ssGSEA with {gene_set_file}...")

    ss = gp.ssgsea(
        data=matrix,
        gene_sets=os.path.join(GENE_SETS_DIR, gene_set_file),
        outdir=None,
        sample_norm_method="rank",
        no_plot=True,
        processes=4,
    )

    scores = ss.res2d
    output_path = os.path.join(RESULTS_DIR, f"{output_name}_scores.csv")
    scores.to_csv(output_path)
    print(f"Saved: {output_path} — {scores.shape[0]} gene sets x {scores.shape[1]} samples")
    return scores

def main():
    matrix_path = os.path.join(RESULTS_DIR, "gene_sample_matrix.csv")

    run_ssgsea(matrix_path, "h.all.v2026.1.Hs.symbols.gmt", "hallmarks")
    run_ssgsea(matrix_path, "c2.all.v2026.1.Hs.symbols.gmt", "c2")

if __name__ == "__main__":
    main()
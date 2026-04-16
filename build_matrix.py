import os
import pandas as pd
import numpy as np
import config

def build_matrix(data_dir, output_path):
    matrices = []
    sample_ids = []

    # Goes through each folder
    for case_id in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_id)
        # Skip stray files
        if not os.path.isdir(case_path):
            continue

        # find the tsv file inside the case folder
        tsv_files = [f for f in os.listdir(case_path) if f.endswith(".tsv")]
        if not tsv_files:
            print(f"No TSV found in {case_id}, skipping")
            continue

        tsv_path = os.path.join(case_path, tsv_files[0])

        # read the file
        df = pd.read_csv(tsv_path, sep="\t", comment="#")

        # drop the metadata rows (N_unmapped, N_multimapping, etc.)
        df = df[df["gene_id"].str.startswith("ENSG")]

        # keep gene_name and tpm column
        df = df[["gene_name", "tpm_unstranded"]].set_index("gene_name")

        matrices.append(df)
        sample_ids.append(case_id)

    # combine into one gene x sample matrix
    matrix = pd.concat(matrices, axis=1)
    matrix.columns = sample_ids

    # fill missing values with per-gene median
    matrix = matrix.apply(lambda row: row.fillna(row.median()), axis=1)

    # save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matrix.to_csv(output_path)
    print(f"Matrix saved: {matrix.shape[0]} genes x {matrix.shape[1]} samples")

    return matrix

def main():
    output_path = os.path.join("results", "gene_sample_matrix.csv")
    matrix = build_matrix(config.DATA_DIR, output_path)
    print(matrix.head())

# To run use python build_matrix.py
if __name__ == "__main__":
    main()
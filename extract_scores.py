import pandas as pd

RESULTS_DIR = "results"

GLYCOLYSIS_TERM = "HALLMARK_GLYCOLYSIS"
OXPHOS_TERM = "HALLMARK_OXIDATIVE_PHOSPHORYLATION"


def main():
    scores = pd.read_csv(f"{RESULTS_DIR}/hallmarks_scores.csv")

    # pivot to pathway x sample matrix
    matrix = scores.pivot(index="Term", columns="Name", values="NES")

    # extract just the two pathways we care about
    metabolic_matrix = matrix.loc[[GLYCOLYSIS_TERM, OXPHOS_TERM]]
    print(f"{metabolic_matrix.shape[0]} pathways x {metabolic_matrix.shape[1]} samples")
    print(metabolic_matrix.index.tolist())

    metabolic_matrix.to_csv(f"{RESULTS_DIR}/metabolic_scores.csv")
    print("Saved: results/metabolic_scores.csv")


if __name__ == "__main__":
    main()
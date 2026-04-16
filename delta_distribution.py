import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "delta_distributions")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {"Glycolytic": "steelblue", "Oxidative": "darkorange", "Mixed": "green"}


def main():
    matrix = pd.read_csv(os.path.join(RESULTS_DIR, "metabolic_scores.csv"), index_col=0)
    labels = pd.read_csv(os.path.join(RESULTS_DIR, "metabolic_labels.csv"))

    # map sample_id -> label
    label_map = dict(zip(labels["sample_id"], labels["label"]))
    unique_labels = sorted(labels["label"].unique())

    for pathway in matrix.index:
        fig, ax = plt.subplots(figsize=(8, 4))

        for label in unique_labels:
            samples = [s for s in matrix.columns if label_map.get(s) == label]
            values = matrix.loc[pathway, samples]
            ax.hist(
                values,
                bins=30,
                alpha=0.5,
                label=f"{label} (n={len(samples)})",
                color=COLORS[label],
                edgecolor="none",
            )

        ax.set_title(pathway.replace("HALLMARK_", "").replace("_", " ").title())
        ax.set_xlabel("NES")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        plt.tight_layout()

        filename = pathway.replace("/", "_") + ".png"
        fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(matrix.index)} plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
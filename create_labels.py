import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

GLYCOLYSIS_TERM = "HALLMARK_GLYCOLYSIS"
OXPHOS_TERM = "HALLMARK_OXIDATIVE_PHOSPHORYLATION"


def main():
    matrix = pd.read_csv(f"{RESULTS_DIR}/metabolic_scores.csv", index_col=0)

    # one score per sample for each pathway
    glycolysis = matrix.loc[GLYCOLYSIS_TERM]
    oxphos = matrix.loc[OXPHOS_TERM]

    # delta: positive means more glycolytic, negative means more oxidative
    delta = glycolysis - oxphos

    # threshold is one standard deviation from the mean of the delta
    mean = delta.mean()
    std = delta.std()
    upper = mean + std
    lower = mean - std

    def assign_label(d):
        if d > upper:
            return "Glycolytic"
        elif d < lower:
            return "Oxidative"
        else:
            return "Mixed"

    labels = delta.apply(assign_label)

    # save labels
    label_df = pd.DataFrame({
        "sample_id": delta.index,
        "glycolysis_nes": glycolysis.values,
        "oxphos_nes": oxphos.values,
        "delta": delta.values,
        "label": labels.values,
    })

    label_df.to_csv(f"{RESULTS_DIR}/metabolic_labels.csv", index=False)
    print(f"Saved: results/metabolic_labels.csv")
    print(label_df["label"].value_counts())

    # plot delta distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta, bins=50, color="steelblue", edgecolor="none")
    ax.axvline(upper, color="red", linestyle="--", label=f"+1 SD ({upper:.2f})")
    ax.axvline(lower, color="orange", linestyle="--", label=f"-1 SD ({lower:.2f})")
    ax.set_title("Delta Distribution — Glycolysis NES minus OXPHOS NES")
    ax.set_xlabel("Delta NES")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"{RESULTS_DIR}/figures/delta_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: results/figures/delta_distribution.png")


if __name__ == "__main__":
    main()
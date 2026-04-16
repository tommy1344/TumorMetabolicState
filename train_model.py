import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_data():
    # Gene expression matrix is genes x samples — transpose to samples x genes
    matrix = pd.read_csv(os.path.join(RESULTS_DIR, "gene_sample_matrix.csv"), index_col=0)
    X = matrix.T

    labels = pd.read_csv(os.path.join(RESULTS_DIR, "metabolic_labels.csv"), index_col="sample_id")

    # Keep only samples present in both
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common, "label"]

    print(f"Samples: {len(common)}  |  Features: {X.shape[1]}")
    print(f"Label distribution:\n{y.value_counts()}\n")

    return X, y


def preprocess(X_train, X_test):
    # log1p to compress TPM range, then standard scale
    # Fit ONLY on train data — apply the same transform to test
    X_train_log = np.log1p(X_train.values)
    X_test_log = np.log1p(X_test.values)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)

    return X_train_scaled, X_test_scaled


def train(X_train, y_train):
    # class_weight="balanced" adjusts for unequal class sizes
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:        {acc:.3f}")
    print(f"F1 (weighted):   {f1:.3f}")
    print("\nPer-class breakdown:")
    print(classification_report(y_test, y_pred))

    # Save metrics to file
    metrics_path = os.path.join(RESULTS_DIR, "model_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy:        {acc:.3f}\n")
        f.write(f"F1 (weighted):   {f1:.3f}\n\n")
        f.write(classification_report(y_test, y_pred))
    print(f"Saved: {metrics_path}")

    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    class_labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Annotate each cell with its count
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X, y = load_data()

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Train label counts:\n{y_train.value_counts()}")
    print(f"Test label counts:\n{y_test.value_counts()}\n")

    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    clf = train(X_train_scaled, y_train)
    y_pred = evaluate(clf, X_test_scaled, y_test)
    plot_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()

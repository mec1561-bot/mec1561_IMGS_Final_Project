"""
IMGS 789 Final Project
Matthew Chaikowsky
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATASET_ID = 1067
DATASET_NAME = "NASA OpenML ID 1067 Software Defect Dataset"
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs_nasa"
OUT_DIR.mkdir(exist_ok=True)

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print(
        "imbalanced-learn is not installed. RandomOverSampler and SMOTE will be skipped.\n"
        "Install it with: pip install imbalanced-learn"
    )


def gmean_score(y_true, y_pred, positive_label=1):
    recall_pos = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return np.sqrt(recall_pos * recall_neg)


def load_nasa_openml_1067():
    data = fetch_openml(data_id=DATASET_ID, as_frame=True)
    X = data.data.copy()
    y_raw = data.target.copy()

    # Convert any categorical input columns if they appear.
    # NASA software defect datasets are usually numeric, but this keeps the script robust.
    X = pd.get_dummies(X, drop_first=True)

    # Encode target labels first.
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y_raw), name="target_encoded")

    # Decide which encoded label is minority.
    counts = y_encoded.value_counts()
    minority_original_code = counts.idxmin()
    majority_original_code = counts.idxmax()

    # Remap so majority = 0 and minority = 1.
    y = y_encoded.map({majority_original_code: 0, minority_original_code: 1}).astype(int)
    y.name = "target"

    print(f"{DATASET_NAME} loaded")
    print(f"Dataset shape: {X.shape}")
    print("\nOriginal OpenML label mapping:")
    for cls, code in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        role = "minority / likely defective" if code == minority_original_code else "majority / likely non-defective"
        print(f"  {cls}: encoded as {code}, treated as {role}")

    return X.reset_index(drop=True), y.reset_index(drop=True)


def downsample_minority(X, y, minority_keep_fraction=0.25):
    """
    Keeping 25% of the minority.
    """
    rng = np.random.RandomState(RANDOM_STATE)

    majority_idx = y[y == 0].index.to_numpy()
    minority_idx = y[y == 1].index.to_numpy()

    n_keep = max(int(len(minority_idx) * minority_keep_fraction), 5)
    kept_minority_idx = rng.choice(minority_idx, size=n_keep, replace=False)

    kept_idx = np.concatenate([majority_idx, kept_minority_idx])
    rng.shuffle(kept_idx)

    return X.loc[kept_idx].reset_index(drop=True), y.loc[kept_idx].reset_index(drop=True)


def print_class_summary(y, label):
    counts = y.value_counts().sort_index()
    majority = counts.max()
    minority = counts.min()
    ratio = majority / minority

    print(f"\n{label} class counts:")
    print(counts)
    print(f"{label} imbalance ratio majority:minority = {ratio:.2f}:1")
    return counts, ratio


def evaluate_model(name, y_true, y_pred, y_score):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision minority": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall minority": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 minority": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "AUPRC": average_precision_score(y_true, y_score),
        "G-mean": gmean_score(y_true, y_pred, positive_label=1),
    }


def run_experiment(X, y, experiment_name):
    experiment_dir = OUT_DIR / experiment_name.replace(" ", "_").lower()
    experiment_dir.mkdir(exist_ok=True)

    counts, ratio = print_class_summary(y, experiment_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    confusions = {}
    roc_data = {}
    pr_data = {}

    def run_and_record(name, model, X_tr, y_tr, X_te, y_te):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)[:, 1]
        else:
            y_score = model.decision_function(X_te)

        results.append(evaluate_model(name, y_te, y_pred, y_score))
        confusions[name] = confusion_matrix(y_te, y_pred, labels=[0, 1])

        fpr, tpr, _ = roc_curve(y_te, y_score, pos_label=1)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_te, y_score))

        precision, recall, _ = precision_recall_curve(y_te, y_score, pos_label=1)
        pr_data[name] = (precision, recall, average_precision_score(y_te, y_score))


    run_and_record(
        "LogReg baseline",
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    run_and_record(
        "RandomForest baseline",
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        X_train,
        y_train,
        X_test,
        y_test,
    )

   
    run_and_record(
        "LogReg class_weight",
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    run_and_record(
        "RandomForest class_weight",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    if HAS_IMBLEARN:
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train)
        run_and_record(
            "LogReg RandomOverSampler",
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            X_train_ros,
            y_train_ros,
            X_test_scaled,
            y_test,
        )

        minority_train_count = int((y_train == 1).sum())
        k_neighbors = min(5, minority_train_count - 1)
        if k_neighbors >= 1:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            run_and_record(
                "LogReg SMOTE",
                LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                X_train_smote,
                y_train_smote,
                X_test_scaled,
                y_test,
            )
        else:
            print("SMOTE skipped because there are too few minority samples in training set.")

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(experiment_dir / "results_table.csv")

    print(f"\n=== Results Table: {experiment_name} ===")
    print(results_df.round(3))

    # Save class distribution plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["majority/non-defective (0)", "minority/defective (1)"], [counts.get(0, 0), counts.get(1, 0)])
    ax.set_title(f"Class Distribution: {experiment_name}")
    ax.set_ylabel("Number of samples")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 1, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(experiment_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # Confusion matrices
    n_models = len(confusions)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, cm) in zip(axes, confusions.items()):
        ax.imshow(cm)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["non-defect", "defect"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["non-defect", "defect"], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    for ax in axes[n_models:]:
        ax.axis("off")
    fig.suptitle(f"Confusion Matrices: {experiment_name}")
    fig.tight_layout()
    fig.savefig(experiment_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Metric comparison
    plot_metrics = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision minority",
        "Recall minority",
        "F1 minority",
        "Macro-F1",
        "ROC-AUC",
        "AUPRC",
        "G-mean",
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df[plot_metrics].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Metric Comparison: {experiment_name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(experiment_dir / "metric_comparison.png", dpi=150)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves: {experiment_name}")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(experiment_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Precision-recall curves
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (precision, recall, auprc) in pr_data.items():
        ax.plot(recall, precision, label=f"{name} AUPRC={auprc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves: {experiment_name}")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(experiment_dir / "precision_recall_curves.png", dpi=150)
    plt.close(fig)

    # Text summary
    with open(experiment_dir / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Class counts: {counts.to_dict()}\n")
        f.write(f"Imbalance ratio majority:minority = {ratio:.2f}:1\n\n")
        f.write(results_df.round(4).to_string())
        f.write("\n\nConfusion matrices:\n")
        for name, cm in confusions.items():
            f.write(f"\n{name}:\n{cm}\n")

    return results_df, counts, ratio


def create_combined_summary(original_df, original_counts, original_ratio,
                            imbalanced_df, imbalanced_counts, imbalanced_ratio):
                                
    combined = pd.concat(
        {
            "Original OpenML 1067": original_df,
            "Strongly Imbalanced OpenML 1067": imbalanced_df,
        },
        names=["Experiment", "Model"],
    )

    combined.to_csv(OUT_DIR / "combined_results_table.csv")

    with open(OUT_DIR / "combined_results_summary.txt", "w", encoding="utf-8") as f:
        f.write("IMGS 789 NASA OpenML ID 1067 Class Imbalance Project Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Original NASA OpenML 1067 dataset:\n")
        f.write(f"  Class counts: {original_counts.to_dict()}\n")
        f.write(f"  Imbalance ratio majority:minority = {original_ratio:.2f}:1\n\n")

        f.write("Strongly imbalanced NASA OpenML 1067 dataset:\n")
        f.write(f"  Class counts: {imbalanced_counts.to_dict()}\n")
        f.write(f"  Imbalance ratio majority:minority = {imbalanced_ratio:.2f}:1\n\n")

        f.write("Combined results:\n")
        f.write(combined.round(4).to_string())

        f.write("\n\nNotes for report:\n")
        f.write("- Accuracy should be compared against balanced accuracy, minority recall, AUPRC, and G-mean.\n")
        f.write("- The strongly imbalanced condition is included because the original OpenML OpenML 1067 version is only mildly imbalanced.\n")
        f.write("- The key question is whether high accuracy hides poor detection of defective/minority modules.\n")

    key_metrics = ["Accuracy", "Balanced Accuracy", "Recall minority", "F1 minority", "AUPRC", "G-mean"]
    plot_data = combined[key_metrics].reset_index()
    plot_data["Label"] = plot_data["Experiment"] + " - " + plot_data["Model"]

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_data.set_index("Label")[key_metrics].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Original vs. Strongly Imbalanced NASA OpenML ID 1067: Key Metrics")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_metric_comparison.png", dpi=150)
    plt.close(fig)


def main():
    X, y = load_nasa_openml_1067()

    original_df, original_counts, original_ratio = run_experiment(
        X,
        y,
        "Original OpenML 1067",
    )

    X_imb, y_imb = downsample_minority(X, y, minority_keep_fraction=0.25)

    imbalanced_df, imbalanced_counts, imbalanced_ratio = run_experiment(
        X_imb,
        y_imb,
        "Strongly Imbalanced OpenML 1067",
    )

    create_combined_summary(
        original_df,
        original_counts,
        original_ratio,
        imbalanced_df,
        imbalanced_counts,
        imbalanced_ratio,
    )

    print("\nAll outputs saved to:")
    print(OUT_DIR)
    print("\nMain files to use in your report:")
    print(f"  {OUT_DIR / 'combined_results_table.csv'}")
    print(f"  {OUT_DIR / 'combined_results_summary.txt'}")
    print(f"  {OUT_DIR / 'combined_metric_comparison.png'}")


if __name__ == "__main__":
    main()

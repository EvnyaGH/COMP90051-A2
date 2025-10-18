# visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_model_performance(ht_results: dict, save_dir: Path):
    models, means, stds = [], [], []
    for model, res in ht_results.items():
        scores = res["cv_scores"]
        models.append(model)
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    plt.figure(figsize=(7, 5))
    plt.bar(models, means, yerr=stds, capsize=6)
    plt.ylabel("F1 (mean ± std over outer folds)")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(save_dir / "model_performance_bar.png", dpi=200)
    plt.close()


def plot_confusion_matrices(ht_results: dict, save_dir: Path):
    """Plot confusion matrices for all models"""
    # Get the number of models
    n_models = len(ht_results)
    if n_models == 0:
        return

    # Create subplots
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for i, (model_name, res) in enumerate(ht_results.items()):
        # Get confusion matrix from fold_metrics if available
        if "fold_metrics" in res and res["fold_metrics"]:
            # Calculate average confusion matrix across folds
            conf_matrices = []
            for fold_metric in res["fold_metrics"]:
                if "confusion_matrix" in fold_metric:
                    conf_matrices.append(np.array(fold_metric["confusion_matrix"]))

            if conf_matrices:
                # Average across folds
                avg_conf_matrix = np.mean(conf_matrices, axis=0)
            else:
                # If no confusion matrix in fold_metrics, create a placeholder
                print(f"Warning: No confusion matrix found for {model_name}")
                continue
        else:
            print(f"Warning: No fold_metrics found for {model_name}")
            continue

        # Plot confusion matrix
        sns.heatmap(
            avg_conf_matrix.astype(int),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            cbar=True,
            square=True,
        )
        axes[i].set_title(f"{model_name}\nConfusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].set_xticklabels(["Negative", "Positive"])
        axes[i].set_yticklabels(["Negative", "Positive"])

    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Confusion matrices saved to results/confusion_matrices.png")


def export_summary_table(ht_results: dict, save_dir: Path):
    rows = []
    for model, res in ht_results.items():
        am = res.get("aggregate_metrics", {})
        rows.append(
            {
                "Model": model,
                "Accuracy": am.get("accuracy"),
                "Precision": am.get("precision"),
                "Recall": am.get("recall"),
                "F1": am.get("f1"),
                "Mean_CV(F1)": res.get("mean_cv_score"),
                "Std_CV(F1)": res.get("std_cv_score"),
                "Best_Params": str(res.get("best_params")),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "model_summary.csv", index=False)
    (save_dir / "model_summary.md").write_text(
        df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8"
    )

# visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_model_performance(ht_results: dict, save_dir: Path):
    models, means, stds = [], [], []
    for model, res in ht_results.items():
        scores = res["cv_results"]["cv_scores"]
        models.append(model)
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    plt.figure(figsize=(7,5))
    plt.bar(models, means, yerr=stds, capsize=6)
    plt.ylabel("F1 (mean ± std over outer folds)")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(save_dir / "model_performance_bar.png", dpi=200)
    plt.close()

def export_summary_table(ht_results: dict, save_dir: Path):
    rows = []
    for model, res in ht_results.items():
        am = res.get("aggregate_metrics", {})
        rows.append({
            "Model": model,
            "Accuracy": am.get("accuracy"),
            "Precision": am.get("precision"),
            "Recall": am.get("recall"),
            "F1": am.get("f1"),
            "Mean_CV(F1)": res.get("mean_cv_score"),
            "Std_CV(F1)":  res.get("std_cv_score"),
            "Best_Params": str(res.get("best_params")),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "model_summary.csv", index=False)
    (save_dir / "model_summary.md").write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

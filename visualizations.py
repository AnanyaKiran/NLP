"""
visualizations.py
─────────────────
Generates all output plots:
  1. Model comparison bar chart (CV accuracy + F1)
  2. Confusion matrix (best model)
  3. Feature importance chart (Random Forest)
  4. PCA scatter plot (artist clustering)
  5. Feature distributions by artist (box plots)
  6. Per-fold CV accuracy violin plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─── Style ────────────────────────────────────
PALETTE = "#1B4F72"
ACCENT  = "#E67E22"
GREEN   = "#1E8449"
PURPLE  = "#6C3483"
BG      = "#F8F9FA"
GRID    = "#E0E0E0"

ARTIST_COLORS = [
    "#1B4F72","#2E86C1","#1E8449","#E67E22","#C0392B",
    "#8E44AD","#17A589","#B7950B","#5D6D7E","#A93226",
    "#117A65","#784212","#1A5276","#6C3483","#0E6655"
]

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})

MODEL_NAMES = [
    "Logistic Regression",
    "LinearSVC + TF-IDF",
    "Random Forest",
    "Hist Gradient Boosting"
]
MODEL_COLORS = [PALETTE, ACCENT, GREEN, PURPLE]


def save(fig, path, fname):
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → Saved: {fname}")


def plot_model_comparison(results, out_dir):
    models = [m for m in MODEL_NAMES if m in results]
    accs = [results[m]["cv_accuracy"] for m in models]
    mf1s = [results[m]["cv_macro_f1"] for m in models]
    wf1s = [results[m]["cv_weighted_f1"] for m in models]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(models))
    w = 0.25
    bars1 = ax.bar(x - w, accs, w, label="CV Accuracy", color=PALETTE, alpha=0.9)
    bars2 = ax.bar(x,     mf1s, w, label="Macro F1",    color=ACCENT,  alpha=0.9)
    bars3 = ax.bar(x + w, wf1s, w, label="Weighted F1", color=GREEN,   alpha=0.9)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — 5-Fold Cross-Validation", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, min(1.0, max(accs + mf1s + wf1s) + 0.15))
    ax.legend(frameon=False, fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out_dir, "1_model_comparison.png")


def plot_confusion_matrix(results, out_dir):
    # Use best model by CV accuracy
    available = [m for m in MODEL_NAMES if m in results]
    best_name = max(available, key=lambda m: results[m]["cv_accuracy"])

    res     = results[best_name]
    cm      = res["confusion_matrix"]
    artists = res["artist_names"]

    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n        = len(artists)
    fig_size = max(10, n * 0.35)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm_norm, annot=(n <= 20), fmt=".2f",
        cmap="Blues", linewidths=0.3, linecolor="#cccccc",
        xticklabels=artists, yticklabels=artists,
        ax=ax, cbar_kws={"shrink": 0.6},
        annot_kws={"size": 7}
    )
    ax.set_title(f"Confusion Matrix — {best_name} (Row-Normalised)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Artist", fontsize=11)
    ax.set_ylabel("True Artist", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    fig.tight_layout()
    save(fig, out_dir, "2_confusion_matrix.png")


def plot_feature_importance(results, out_dir):
    fi_df = results["feature_importance"].head(15)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [PALETTE if i < 5 else ACCENT if i < 10 else GREEN
              for i in range(len(fi_df))]
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
            color=colors[::-1], alpha=0.9)

    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{w:.4f}", va="center", fontsize=8.5)

    ax.set_xlabel("Feature Importance (Mean Gini Decrease)", fontsize=11)
    ax.set_title("Top 15 Discriminative Features\n(Random Forest)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out_dir, "3_feature_importance.png")


def plot_pca(feature_df, out_dir, n_top_artists=12):
    feature_cols = [c for c in feature_df.columns if c.startswith("f_")]
    top  = feature_df["artist"].value_counts().head(n_top_artists).index
    sub  = feature_df[feature_df["artist"].isin(top)].copy()

    X      = StandardScaler().fit_transform(sub[feature_cols].values)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var1   = pca.explained_variance_ratio_[0] * 100
    var2   = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, artist in enumerate(top.tolist()):
        mask = sub["artist"].values == artist
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=artist, color=ARTIST_COLORS[i % len(ARTIST_COLORS)],
                   alpha=0.55, s=35, linewidths=0)

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    ax.set_title(f"PCA — Linguistic Style Clustering by Artist (Top {n_top_artists})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False, fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    fig.tight_layout()
    save(fig, out_dir, "4_pca_clustering.png")


def plot_feature_distributions(feature_df, out_dir):
    top_artists = feature_df["artist"].value_counts().head(10).index.tolist()
    sub = feature_df[feature_df["artist"].isin(top_artists)].copy()
    sub["short"] = sub["artist"].apply(lambda x: x.split(",")[0])

    # Use actual column names from spaCy extractor
    feature_map = {
        "f_avg_word_length":  "Avg Word Length",
        "f_spatial_density":  "Spatial Word Density",
        "f_colour_density":   "Colour Word Density",
        "f_comma_per_sent":   "Commas per Sentence",
        "f_propn_density":    "Proper Noun Density",
        "f_avg_sent_length":  "Avg Sentence Length",
    }
    # Only keep features that exist in the dataframe
    feature_map = {k: v for k, v in feature_map.items() if k in sub.columns}

    n_feats = len(feature_map)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (feat, title) in enumerate(feature_map.items()):
        order = sub.groupby("short")[feat].median().sort_values(ascending=False).index
        sns.boxplot(data=sub, x="short", y=feat, order=order, ax=axes[i],
                    palette="Blues_d", linewidth=0.8,
                    fliersize=2, flierprops={"alpha": 0.3})
        axes[i].set_title(title, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", rotation=45, labelsize=7.5)
        axes[i].yaxis.grid(True, color=GRID, linewidth=0.5)
        axes[i].set_axisbelow(True)

    # Hide unused subplots
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Linguistic Feature Distributions Across Top 10 Artists",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "5_feature_distributions.png")


def plot_cv_scores_per_model(results, out_dir):
    models   = [m for m in MODEL_NAMES if m in results]
    colors   = MODEL_COLORS[:len(models)]
    data = []
    for m in models:
        for fold_acc in results[m]["cv_scores"]["test_accuracy"]:
            data.append({"Model": m, "CV Accuracy": fold_acc})
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.violinplot(data=df, x="Model", y="CV Accuracy",
                   palette=colors, inner="box", cut=0, linewidth=1.2, ax=ax)
    ax.set_title("Per-Fold CV Accuracy Distribution (5 Folds)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out_dir, "6_cv_fold_distribution.png")


def generate_all_plots(feature_df, results, out_dir):
    print()
    plot_model_comparison(results, out_dir)
    plot_confusion_matrix(results, out_dir)
    plot_feature_importance(results, out_dir)
    plot_pca(feature_df, out_dir)
    plot_feature_distributions(feature_df, out_dir)
    plot_cv_scores_per_model(results, out_dir)
    print(f"\n  6 plots saved to: {out_dir}")
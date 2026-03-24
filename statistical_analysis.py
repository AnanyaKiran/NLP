"""
statistical_analysis.py
────────────────────────
Performs group-level statistical tests to determine whether artists
show significantly different linguistic patterns.

Methods:
  - Kruskal-Wallis test (non-parametric ANOVA equivalent)
  - Effect size: eta-squared (η²)
  - Top discriminating features ranked by significance
"""

import numpy as np
import pandas as pd
from scipy import stats


def eta_squared(h_stat, n_groups, n_total):
    """
    Compute eta-squared effect size from Kruskal-Wallis H statistic.
    η² = (H - k + 1) / (n - k)
    where k = number of groups, n = total observations.
    """
    if n_total - n_groups == 0:
        return 0.0
    return (h_stat - n_groups + 1) / (n_total - n_groups)


def interpret_effect_size(eta2):
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"


def run_statistical_analysis(feature_df):
    """
    Run Kruskal-Wallis test for each linguistic feature across artist groups.
    Prints a ranked table of statistically significant features.
    """
    feature_cols = [c for c in feature_df.columns if c.startswith("f_")]
    artist_groups = feature_df["artist"].unique()
    n_groups = len(artist_groups)
    n_total = len(feature_df)

    results = []

    for col in feature_cols:
        groups = [
            feature_df[feature_df["artist"] == a][col].values
            for a in artist_groups
        ]
        # Filter empty groups
        groups = [g for g in groups if len(g) > 0]

        try:
            h_stat, p_val = stats.kruskal(*groups)
            eta2 = max(0, eta_squared(h_stat, len(groups), n_total))
            results.append({
                "feature": col.replace("f_", ""),
                "H_statistic": round(h_stat, 2),
                "p_value": p_val,
                "eta_squared": round(eta2, 4),
                "effect": interpret_effect_size(eta2),
                "significant": p_val < 0.05
            })
        except Exception:
            pass

    results_df = pd.DataFrame(results).sort_values("eta_squared", ascending=False)

    sig = results_df[results_df["significant"]]
    print(f"\n  Significant features (p < 0.05): {len(sig)}/{len(results_df)}")
    print(f"  Artists tested: {n_groups}, Total samples: {n_total}")
    print()
    print(f"  {'Feature':<30} {'H-stat':>8}  {'p-value':>10}  {'η²':>7}  Effect")
    print("  " + "-" * 65)
    for _, row in results_df.head(15).iterrows():
        sig_marker = "✓" if row["significant"] else " "
        print(f"  {sig_marker} {row['feature']:<28} {row['H_statistic']:>8.2f}  "
              f"{row['p_value']:>10.4f}  {row['eta_squared']:>7.4f}  {row['effect']}")

    return results_df


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
    df = pd.DataFrame({
        "f_avg_word_length": np.random.normal(5, 1, 100),
        "f_spatial_density": np.random.normal(0.1, 0.05, 100),
        "artist": np.random.choice(["A", "B", "C"], 100)
    })
    run_statistical_analysis(df)

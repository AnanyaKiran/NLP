"""
Linguistic Style-Based Artist Classification Pipeline
=====================================================
Combines spaCy linguistic features + TF-IDF for artist prediction.

Dataset strategy:
  - Artists with >= MIN_SAMPLES are classification targets
  - "Unknown" rows fit the scaler (broader feature distribution)
  - TF-IDF is fitted on labelled training descriptions only
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from feature_extractor import extract_features
from statistical_analysis import run_statistical_analysis
from classifiers import run_classification
from visualizations import generate_all_plots

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "caption.csv"
MIN_SAMPLES  = 10
TEXT_COLUMN  = "description"
LABEL_COLUMN = "artist_name"
RESULTS_DIR  = "outputs"


def load_data(path, text_col, label_col, min_samples):
    print("=" * 62)
    print("   ART DESCRIPTION → ARTIST CLASSIFICATION PIPELINE")
    print("=" * 62)

    df = pd.read_csv(path)
    print(f"\n[DATA] Loaded         : {len(df):,} rows total")
    print(f"[DATA] Unique artists  : {df[label_col].nunique()}")

    df = df.dropna(subset=[text_col, label_col])
    print(f"[DATA] After dropna   : {len(df):,} rows")

    unknown_df = df[df[label_col] == "Unknown"].copy()
    known_df   = df[df[label_col] != "Unknown"].copy()

    counts   = known_df[label_col].value_counts()
    eligible = counts[counts >= min_samples].index
    train_df = known_df[known_df[label_col].isin(eligible)].copy()

    print(f"\n[DATA] Unknown rows (scaler boost) : {len(unknown_df):,}")
    print(f"[DATA] Artists with >= {min_samples} samples  : {len(eligible)}")
    print(f"[DATA] Classification rows         : {len(train_df):,}")
    print(f"[DATA] Excluded (< {min_samples} samples)     : {len(known_df) - len(train_df):,}")

    return train_df, unknown_df


def main():
    # 1. Load data
    train_df, unknown_df = load_data(
        DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, MIN_SAMPLES
    )

    # 2. Extract spaCy linguistic features
    print(f"\n[FEATURES] Extracting spaCy features for {len(train_df):,} rows...")
    feature_df = extract_features(train_df[TEXT_COLUMN])
    feature_df["artist"] = train_df[LABEL_COLUMN].values

    # Pass raw text so TF-IDF can use it
    feature_df["text"] = train_df[TEXT_COLUMN].values

    print(f"[FEATURES] Extracting features for {len(unknown_df):,} Unknown rows...")
    unknown_feat_df = extract_features(unknown_df[TEXT_COLUMN], verbose=False)

    print(f"[FEATURES] {feature_df.shape[1] - 2} linguistic features extracted")

    # 3. Statistical analysis
    print("\n[STATS] Running Kruskal-Wallis + Effect Size analysis...")
    run_statistical_analysis(feature_df)

    # 4. ML Classification (linguistic + TF-IDF combined)
    print("\n[ML] Running classification models (linguistic + TF-IDF)...")
    results = run_classification(feature_df, unknown_feat_df)

    # 5. Visualizations
    print("\n[PLOT] Generating visualizations...")
    generate_all_plots(feature_df, results, RESULTS_DIR)

    print("\n" + "=" * 62)
    print(f"  PIPELINE COMPLETE — outputs saved to: {RESULTS_DIR}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
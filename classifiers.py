"""
classifiers.py
──────────────
Combines spaCy linguistic features + TF-IDF text features for
artist classification.

Feature pipeline:
  - 44 spaCy linguistic features (normalised with boosted scaler)
  - TF-IDF on raw description text (up to 3000 features, 1-2 ngrams)
  - Both combined via ColumnTransformer → fed to classifiers

Models:
  1. Logistic Regression       — tuned C
  2. SVM (RBF kernel)          — tuned C + gamma
  3. Random Forest             — feature importance
  4. Hist Gradient Boosting    — typically highest accuracy
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


def build_combined_matrix(X_ling, texts, tfidf=None, fit=False):
    """
    Combine linguistic features (dense) + TF-IDF (sparse) into one matrix.
    If fit=True, fits the tfidf vectorizer on texts.
    Returns (combined_matrix, tfidf_vectorizer).
    """
    if tfidf is None:
        tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            analyzer="word"
        )

    if fit:
        X_tfidf = tfidf.fit_transform(texts)
    else:
        X_tfidf = tfidf.transform(texts)

    # Convert dense linguistic features to sparse and hstack
    X_ling_sparse = csr_matrix(X_ling)
    X_combined = hstack([X_ling_sparse, X_tfidf])

    return X_combined, tfidf


def run_classification(feature_df, unknown_feat_df=None):
    feature_cols = [c for c in feature_df.columns if c.startswith("f_")]

    X_ling = feature_df[feature_cols].values
    texts  = feature_df["text"].values if "text" in feature_df.columns else None
    le     = LabelEncoder()
    y      = le.fit_transform(feature_df["artist"].values)
    artist_names = le.classes_

    # Unknown rows for scaler fitting
    X_unknown = unknown_feat_df[feature_cols].values if (
        unknown_feat_df is not None and len(unknown_feat_df) > 0
    ) else None

    print(f"\n  Linguistic features : {len(feature_cols)}")
    print(f"  TF-IDF features     : up to 3,000 (1–2 ngrams)")
    print(f"  Artists (classes)   : {len(artist_names)}")
    print(f"  Train samples       : {len(X_ling)}")
    print(f"  Unknown samples     : {len(X_unknown) if X_unknown is not None else 0} (scaler only)")
    print(f"  Random baseline     : {1/len(artist_names)*100:.1f}%")

    # ── Fit boosted scaler on train + unknown ────────────────────────────────
    if X_unknown is not None:
        X_all = np.vstack([X_ling, X_unknown])
    else:
        X_all = X_ling
    boosted_scaler = StandardScaler().fit(X_all)
    X_scaled = boosted_scaler.transform(X_ling)

    # ── Fit TF-IDF + build combined matrix ───────────────────────────────────
    if texts is not None:
        print("\n  [TF-IDF] Fitting vectorizer on descriptions...")
        X_combined, tfidf = build_combined_matrix(X_scaled, texts, fit=True)
        print(f"  [TF-IDF] Combined feature matrix: {X_combined.shape}")
    else:
        print("\n  [WARNING] No raw text column found — using linguistic features only.")
        X_combined = csr_matrix(X_scaled)
        tfidf = None

    # ── Tune LR ──────────────────────────────────────────────────────────────
    print("\n  [Tuning] Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=3000, solver="saga", random_state=42),
        param_grid={"C": [0.01, 0.1, 1, 10]},
        cv=3, scoring="accuracy", n_jobs=-1
    )
    lr_grid.fit(X_combined, y)
    best_lr = lr_grid.best_estimator_
    print(f"  Best C = {lr_grid.best_params_['C']}")

    # ── Tune LinearSVC (faster than RBF for high-dim sparse) ─────────────────
    print("  [Tuning] LinearSVC (fast SVM for sparse+TF-IDF)...")
    svc_grid = GridSearchCV(
        CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42)),
        param_grid={"estimator__C": [0.01, 0.1, 1, 5]},
        cv=3, scoring="accuracy", n_jobs=-1
    )
    svc_grid.fit(X_combined, y)
    best_svc = svc_grid.best_estimator_
    print(f"  Best C = {svc_grid.best_params_['estimator__C']}")

    # ── Define all models ─────────────────────────────────────────────────────
    models = {
        "Logistic Regression":    best_lr,
        "LinearSVC + TF-IDF":    best_svc,
        "Random Forest":          RandomForestClassifier(
                                      n_estimators=300, max_depth=None,
                                      min_samples_leaf=1, max_features="sqrt",
                                      random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
                                      max_iter=300, learning_rate=0.05,
                                      max_depth=6, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print()
    print(f"  {'Model':<26} {'CV Acc':>8}  {'Macro F1':>9}  {'Weighted F1':>11}")
    print("  " + "-" * 62)

    for name, model in models.items():
        # Random Forest & HGB don't handle sparse natively — convert to dense
        if name in ("Random Forest", "Hist Gradient Boosting"):
            X_input = X_combined.toarray()
        else:
            X_input = X_combined

        scoring = ["accuracy", "f1_macro", "f1_weighted"]
        cv_res  = cross_validate(
            model, X_input, y, cv=cv,
            scoring=scoring, return_train_score=False
        )

        acc         = cv_res["test_accuracy"].mean()
        macro_f1    = cv_res["test_f1_macro"].mean()
        weighted_f1 = cv_res["test_f1_weighted"].mean()

        print(f"  {name:<26} {acc:>7.3f}   {macro_f1:>8.3f}   {weighted_f1:>10.3f}")

        model.fit(X_input, y)
        y_pred = model.predict(X_input)

        results[name] = {
            "model":          model,
            "scaler":         boosted_scaler,
            "tfidf":          tfidf,
            "cv_accuracy":    acc,
            "cv_macro_f1":    macro_f1,
            "cv_weighted_f1": weighted_f1,
            "cv_scores":      cv_res,
            "y_true":         y,
            "y_pred":         y_pred,
            "artist_names":   artist_names,
            "feature_names":  feature_cols,
            "confusion_matrix": confusion_matrix(y, y_pred),
            "is_sparse":      name not in ("Random Forest", "Hist Gradient Boosting")
        }

    # ── Feature importance from Random Forest ────────────────────────────────
    rf_model  = results["Random Forest"]["model"]
    tfidf_names = [f"tfidf_{t}" for t in tfidf.get_feature_names_out()] if tfidf else []
    all_names   = [f.replace("f_", "") for f in feature_cols] + tfidf_names

    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    all_names[:len(importances)],
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n  Top 10 Most Discriminative Features (Random Forest):")
    print(f"  {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 48)
    for _, row in fi_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 500)
        print(f"  {row['feature']:<35} {row['importance']:>9.4f}  {bar}")

    results["feature_importance"] = fi_df
    results["label_encoder"]      = le
    results["boosted_scaler"]     = boosted_scaler
    results["tfidf"]              = tfidf

    return results
"""
classifiers.py
──────────────
Combines spaCy linguistic features + TF-IDF for artist classification.

Speed modes (set FAST_MODE at top):
  FAST_MODE = True  → fixed hyperparams, no GridSearch (~2-3 mins)
  FAST_MODE = False → GridSearchCV tuning, higher accuracy (~20-30 mins)
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────────
# ⚡ SET THIS FLAG
# True  = fast run (~2-3 mins),  no tuning
# False = slow run (~20-30 mins), with GridSearch tuning
# ─────────────────────────────────────────────
FAST_MODE = True


def build_tfidf_matrix(X_scaled, texts, tfidf=None, fit=False):
    if tfidf is None:
        tfidf = TfidfVectorizer(
            max_features=2000,       # reduced from 3000 for speed
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,                # increased from 2 for speed
            strip_accents="unicode",
            analyzer="word"
        )
    if fit:
        X_tfidf = tfidf.fit_transform(texts)
    else:
        X_tfidf = tfidf.transform(texts)

    X_combined = hstack([csr_matrix(X_scaled), X_tfidf])
    return X_combined, tfidf


def run_classification(feature_df, unknown_feat_df=None):
    feature_cols = [c for c in feature_df.columns if c.startswith("f_")]

    X_ling = feature_df[feature_cols].values
    texts  = feature_df["text"].values if "text" in feature_df.columns else None
    le     = LabelEncoder()
    y      = le.fit_transform(feature_df["artist"].values)
    artist_names = le.classes_

    X_unknown = unknown_feat_df[feature_cols].values if (
        unknown_feat_df is not None and len(unknown_feat_df) > 0
    ) else None

    print(f"\n  Linguistic features : {len(feature_cols)}")
    print(f"  TF-IDF features     : up to 2,000 (1–2 ngrams)")
    print(f"  Artists (classes)   : {len(artist_names)}")
    print(f"  Train samples       : {len(X_ling)}")
    print(f"  Unknown samples     : {len(X_unknown) if X_unknown is not None else 0}")
    print(f"  Random baseline     : {1/len(artist_names)*100:.1f}%")
    print(f"  Mode                : {'⚡ FAST (no GridSearch)' if FAST_MODE else '🔍 FULL (with GridSearch)'}")

    # ── Fit boosted scaler ───────────────────────────────────────────────────
    X_all = np.vstack([X_ling, X_unknown]) if X_unknown is not None else X_ling
    boosted_scaler = StandardScaler().fit(X_all)
    X_scaled = boosted_scaler.transform(X_ling)

    # ── Build TF-IDF combined matrix ─────────────────────────────────────────
    if texts is not None:
        print("\n  [TF-IDF] Fitting vectorizer...")
        X_combined, tfidf = build_tfidf_matrix(X_scaled, texts, fit=True)
        print(f"  [TF-IDF] Combined matrix shape: {X_combined.shape}")
    else:
        X_combined = csr_matrix(X_scaled)
        tfidf = None

    # ── Build models ─────────────────────────────────────────────────────────
    if FAST_MODE:
        # Fixed good hyperparams — no search needed
        lr_model  = LogisticRegression(C=1.0, max_iter=1000,
                                       solver="saga", random_state=42)
        svc_model = CalibratedClassifierCV(
                        LinearSVC(C=0.1, max_iter=1000, random_state=42))
    else:
        print("\n  [Tuning] Logistic Regression (GridSearch)...")
        lr_grid = GridSearchCV(
            LogisticRegression(max_iter=1000, solver="saga", random_state=42),
            param_grid={"C": [0.01, 0.1, 1, 10]},
            cv=3, scoring="accuracy", n_jobs=-1
        )
        lr_grid.fit(X_combined, y)
        lr_model = lr_grid.best_estimator_
        print(f"  Best LR C = {lr_grid.best_params_['C']}")

        print("  [Tuning] LinearSVC (GridSearch)...")
        svc_grid = GridSearchCV(
            CalibratedClassifierCV(LinearSVC(max_iter=1000, random_state=42)),
            param_grid={"estimator__C": [0.01, 0.1, 1, 5]},
            cv=3, scoring="accuracy", n_jobs=-1
        )
        svc_grid.fit(X_combined, y)
        svc_model = svc_grid.best_estimator_
        print(f"  Best SVC C = {svc_grid.best_params_['estimator__C']}")

    models = {
        "Logistic Regression":    lr_model,
        "LinearSVC + TF-IDF":     svc_model,
        "Random Forest":          RandomForestClassifier(
                                      n_estimators=100,    # reduced from 300
                                      max_depth=20,        # cap depth for speed
                                      min_samples_leaf=2,
                                      max_features="sqrt",
                                      random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
                                      max_iter=100,        # reduced from 300
                                      learning_rate=0.1,   # larger lr = fewer iters
                                      max_depth=6,
                                      random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print()
    print(f"  {'Model':<26} {'CV Acc':>8}  {'Macro F1':>9}  {'Weighted F1':>11}")
    print("  " + "-" * 62)

    for name, model in models.items():
        # RF and HGB need dense input
        X_input = X_combined.toarray() if name in (
            "Random Forest", "Hist Gradient Boosting") else X_combined

        scoring = ["accuracy", "f1_macro", "f1_weighted"]
        cv_res  = cross_validate(
            model, X_input, y, cv=cv,
            scoring=scoring, return_train_score=False,
            n_jobs=-1                                    # parallelise CV folds
        )

        acc         = cv_res["test_accuracy"].mean()
        macro_f1    = cv_res["test_f1_macro"].mean()
        weighted_f1 = cv_res["test_f1_weighted"].mean()

        print(f"  {name:<26} {acc:>7.3f}   {macro_f1:>8.3f}   {weighted_f1:>10.3f}")

        model.fit(X_input, y)
        y_pred = model.predict(X_input)

        results[name] = {
            "model":            model,
            "scaler":           boosted_scaler,
            "tfidf":            tfidf,
            "cv_accuracy":      acc,
            "cv_macro_f1":      macro_f1,
            "cv_weighted_f1":   weighted_f1,
            "cv_scores":        cv_res,
            "y_true":           y,
            "y_pred":           y_pred,
            "artist_names":     artist_names,
            "feature_names":    feature_cols,
            "confusion_matrix": confusion_matrix(y, y_pred),
            "is_sparse":        name not in ("Random Forest", "Hist Gradient Boosting")
        }

    # ── Feature importance (Random Forest only) ───────────────────────────────
    rf_model    = results["Random Forest"]["model"]
    tfidf_names = ([f"tfidf_{t}" for t in tfidf.get_feature_names_out()]
                   if tfidf else [])
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
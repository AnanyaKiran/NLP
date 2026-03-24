"""
train_and_save.py
─────────────────
Run this ONCE to train the model and save it to disk.
After this, predict.py will load instantly every time.

Usage:
    python train_and_save.py
"""

import pickle
import time
import warnings
warnings.filterwarnings("ignore")

CACHE_PATH = "model_cache.pkl"

print("=" * 55)
print("  TRAINING & SAVING MODEL — run this once only")
print("=" * 55)

start = time.time()

# ── Step 1: Load data ─────────────────────────────────────
from main import load_data
train_df, unknown_df = load_data("caption.csv", "description", "artist_name", 10)

# ── Step 2: Extract features ──────────────────────────────
from feature_extractor import extract_features
print(f"\n[1/4] Extracting linguistic features...")
feature_df          = extract_features(train_df["description"])
feature_df["artist"] = train_df["artist_name"].values
feature_df["text"]   = train_df["description"].values

print(f"[2/4] Extracting features for Unknown rows...")
unknown_feat = extract_features(unknown_df["description"], verbose=False)

# ── Step 3: Train classifiers ─────────────────────────────
from classifiers import run_classification
print(f"[3/4] Training classifiers...")
results = run_classification(feature_df, unknown_feat)

# ── Step 4: Save best model ───────────────────────────────
model_names = ["Logistic Regression", "LinearSVC + TF-IDF",
               "Random Forest", "Hist Gradient Boosting"]
best_name = max(model_names, key=lambda n: results[n]["cv_accuracy"])
print(f"\n[4/4] Best model: {best_name}")
print(f"      CV Accuracy: {results[best_name]['cv_accuracy']:.3f}")

cache = {
    "model":        results[best_name]["model"],
    "scaler":       results["boosted_scaler"],
    "tfidf":        results["tfidf"],
    "le":           results["label_encoder"],
    "feature_cols": results[best_name]["feature_names"],
    "is_sparse":    results[best_name]["is_sparse"],
    "best_model":   best_name
}

with open(CACHE_PATH, "wb") as f:
    pickle.dump(cache, f)

elapsed = time.time() - start
print(f"\n✅ Model saved to: {CACHE_PATH}")
print(f"   Training time : {elapsed:.1f}s")
print(f"\n   Now run: python predict.py")
print(f"   Prediction will be instant — no retraining needed.")
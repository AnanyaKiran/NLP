"""
predict.py
──────────
Predict the artist for a new art description.
Uses the best model (Logistic Regression on linguistic + TF-IDF features).

Usage:
    python predict.py "A large ornate figure stands in the foreground..."
    python predict.py   ← interactive mode
"""

import sys
import pickle
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix

from feature_extractor import extract_features_single

MODEL_CACHE = "model_cache.pkl"


def load_or_train():
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"[Model loaded from cache]")
        return cache

    print("[Cache not found] Training model — this may take a few minutes...")
    from main import load_data
    from feature_extractor import extract_features
    from classifiers import run_classification

    train_df, unknown_df = load_data("caption.csv", "description", "artist_name", 10)
    feature_df = extract_features(train_df["description"])
    feature_df["artist"] = train_df["artist_name"].values
    feature_df["text"]   = train_df["description"].values
    unknown_feat = extract_features(unknown_df["description"], verbose=False)
    results = run_classification(feature_df, unknown_feat)

    # Pick best model by CV accuracy
    model_names = ["Logistic Regression", "LinearSVC + TF-IDF",
                   "Random Forest", "Hist Gradient Boosting"]
    best_name = max(model_names, key=lambda n: results[n]["cv_accuracy"])
    print(f"\n[Best model: {best_name} — CV Acc: {results[best_name]['cv_accuracy']:.3f}]")

    cache = {
        "model":        results[best_name]["model"],
        "scaler":       results["boosted_scaler"],
        "tfidf":        results["tfidf"],
        "le":           results["label_encoder"],
        "feature_cols": results[best_name]["feature_names"],
        "is_sparse":    results[best_name]["is_sparse"],
        "best_model":   best_name
    }
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"[Model cached to {MODEL_CACHE}]")
    return cache


def predict(text, cache):
    # Extract linguistic features
    f = extract_features_single(text)
    if f is None:
        print("  Could not extract features from this text.")
        return

    # Build linguistic feature vector
    ling_vec = np.array([f.get(c, 0.0) for c in cache["feature_cols"]]).reshape(1, -1)
    ling_scaled = cache["scaler"].transform(ling_vec)

    # Add TF-IDF features if available
    if cache["tfidf"] is not None:
        tfidf_vec = cache["tfidf"].transform([text])
        if cache["is_sparse"]:
            X_input = hstack([csr_matrix(ling_scaled), tfidf_vec])
        else:
            X_input = np.hstack([ling_scaled, tfidf_vec.toarray()])
    else:
        X_input = ling_scaled

    model = cache["model"]
    probs = model.predict_proba(X_input)[0]
    top5  = np.argsort(probs)[::-1][:5]

    preview = text[:80] + "..." if len(text) > 80 else text
    print(f"\n  Input : \"{preview}\"")
    print(f"\n  Predicted Artist — Top 5  [{cache.get('best_model','Model')}]")
    print("  " + "-" * 50)
    for idx in top5:
        artist = cache["le"].inverse_transform([idx])[0]
        conf   = probs[idx] * 100
        bar    = "█" * int(conf / 2)
        print(f"  {conf:5.1f}%  {artist:<36} {bar}")


if __name__ == "__main__":
    cache = load_or_train()

    if len(sys.argv) > 1:
        predict(" ".join(sys.argv[1:]), cache)
    else:
        print(f"\n  === INTERACTIVE ARTIST PREDICTOR ===")
        print(f"  Model : {cache.get('best_model', 'Unknown')}")
        print(f"  Paste a description and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input("  Description: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                predict(text, cache)
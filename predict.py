"""
predict.py
──────────
Fast artist prediction — loads pre-trained model from disk instantly.

IMPORTANT: Run train_and_save.py once before using this script.

Usage:
    python predict.py                    ← interactive mode
    python predict.py "description..."  ← single prediction
"""

import sys
import pickle
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix
from feature_extractor import extract_features_single

CACHE_PATH = "model_cache.pkl"


def load_model():
    """Load pre-trained model from disk. Exits if not found."""
    if not os.path.exists(CACHE_PATH):
        print("\n❌ No trained model found.")
        print("   Run this first:  python train_and_save.py")
        print("   (Only needed once — takes ~2–3 minutes)\n")
        sys.exit(1)

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    return cache


def predict(text, cache):
    """Predict artist for a single description string."""

    # Extract spaCy linguistic features
    f = extract_features_single(text)
    if f is None:
        print("  ⚠️  Could not extract features — try a longer description.")
        return None

    # Build linguistic vector → scale
    ling_vec    = np.array([f.get(c, 0.0) for c in cache["feature_cols"]]).reshape(1, -1)
    ling_scaled = cache["scaler"].transform(ling_vec)

    # Add TF-IDF vector
    if cache["tfidf"] is not None:
        tfidf_vec = cache["tfidf"].transform([text])
        if cache["is_sparse"]:
            X_input = hstack([csr_matrix(ling_scaled), tfidf_vec])
        else:
            X_input = np.hstack([ling_scaled, tfidf_vec.toarray()])
    else:
        X_input = ling_scaled

    # Predict
    probs = cache["model"].predict_proba(X_input)[0]
    top5  = np.argsort(probs)[::-1][:5]

    # Display
    preview = (text[:80] + "...") if len(text) > 80 else text
    print(f"\n  📝 Input  : \"{preview}\"")
    print(f"  🤖 Model  : {cache.get('best_model', 'Unknown')}")
    print(f"\n  🎨 Predicted Artist — Top 5:")
    print("  " + "─" * 52)
    for rank, idx in enumerate(top5, 1):
        artist = cache["le"].inverse_transform([idx])[0]
        conf   = probs[idx] * 100
        bar    = "█" * int(conf / 2.5)
        marker = " ◀ best" if rank == 1 else ""
        print(f"  {rank}. {conf:5.1f}%  {artist:<36} {bar}{marker}")
    print()

    return cache["le"].inverse_transform([top5[0]])[0]


def interactive_mode(cache):
    print("\n" + "=" * 55)
    print("  🎨 INTERACTIVE ARTIST PREDICTOR")
    print(f"  Model : {cache.get('best_model', 'Unknown')}")
    print("=" * 55)
    print("  Paste an art description and press Enter.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            text = input("  Description: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if text.lower() in ("quit", "exit", "q", ""):
            print("  Goodbye!")
            break

        predict(text, cache)


if __name__ == "__main__":
    # Load model instantly from disk
    cache = load_model()
    print(f"✅ Model loaded: {cache.get('best_model', 'Unknown')}")

    if len(sys.argv) > 1:
        # Single prediction from command line argument
        predict(" ".join(sys.argv[1:]), cache)
    else:
        # Interactive mode
        interactive_mode(cache)
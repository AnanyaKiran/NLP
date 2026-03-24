# Art Description → Artist Classification

A fully functional ML pipeline that predicts the associated artist
from art description text using handcrafted linguistic features.

## Project Structure

```
art_classification/
├── main.py               ← Run the full pipeline
├── feature_extractor.py  ← Linguistic feature extraction (26 features)
├── statistical_analysis.py ← ANOVA / Kruskal-Wallis / Effect size
├── classifiers.py        ← LR, SVM, Random Forest with 5-fold CV
├── visualizations.py     ← 6 publication-quality plots
├── predict.py            ← Predict artist for a new description
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Full Pipeline

```bash
python main.py
```

This will:
1. Load and filter the dataset (artists with ≥20 samples)
2. Extract 26 linguistic features per description
3. Run Kruskal-Wallis tests + effect sizes
4. Train & evaluate 3 classifiers (5-fold CV)
5. Save 6 plots to outputs/

## Predict a Single Description

```bash
# Command line
python predict.py "A large ornate figure stands in the foreground..."

# Interactive mode
python predict.py
```

## Features Extracted

### Lexical (8 features)
- Token count, unique tokens, type-token ratio
- Average word length, long word ratio
- Colour word density, texture word density, adjective proxy

### Syntactic (7 features)
- Sentence count, avg/max sentence length
- Passive voice ratio, preposition density
- Commas per sentence, parenthetical count

### Semantic (5 features)
- Spatial word density, visual framing density
- Hedging word density, proper noun density
- Number density

### Discourse (6 features)
- Colon, semicolon, question mark counts
- Exclamation, quote, and dash counts

## Models

| Model               | Notes                              |
|---------------------|------------------------------------|
| Logistic Regression | Fast baseline, interpretable       |
| SVM (RBF kernel)    | Strong non-linear classifier       |
| Random Forest       | Best for feature importance        |

All models use **StandardScaler** preprocessing and **5-fold stratified CV**.

## Output Plots

1. `1_model_comparison.png`     — CV accuracy + F1 bar chart
2. `2_confusion_matrix.png`     — Row-normalised confusion matrix (RF)
3. `3_feature_importance.png`   — Top 15 discriminative features
4. `4_pca_clustering.png`       — PCA scatter by artist
5. `5_feature_distributions.png`— Box plots per feature per artist
6. `6_cv_fold_distribution.png` — Per-fold accuracy violin plots

## Dataset

- **Source**: `caption.csv`
- **Text column**: `description`
- **Label column**: `artist_name`
- **Filter**: Artists with ≥20 samples, excluding "Unknown"
- **Eligible artists**: 34

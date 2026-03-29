"""
feature_extractor.py  (spaCy version)
──────────────────────────────────────
Extracts rich linguistic features from art description text using spaCy.

Setup (run once):
    pip install spacy
    python -m spacy download en_core_web_sm

Feature groups (44 total):
  LEXICAL    — vocabulary richness, word length, adjective/noun ratios
  SYNTACTIC  — POS ratios, dependency depth, passive voice (real detection)
  SEMANTIC   — NER types, spatial/colour/texture word lists
  DISCOURSE  — punctuation patterns, enumeration, hedging
"""

import re
import numpy as np
import pandas as pd
import spacy
from collections import Counter

# ── Load spaCy model once at import time ──────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "\n[ERROR] spaCy model not found.\n"
        "Run these commands first:\n"
        "  pip install spacy\n"
        "  python -m spacy download en_core_web_sm\n"
    )

# ── Word lists for semantic features ──────────────────────────────────────────
SPATIAL_WORDS = {
    "left", "right", "center", "centre", "foreground", "background",
    "above", "below", "beneath", "beside", "behind", "front", "top",
    "bottom", "corner", "middle", "side", "edge", "inside", "outside",
    "upper", "lower", "vertical", "horizontal", "adjacent", "opposite",
    "near", "far", "distant", "close", "surrounding", "flanked"
}

COLOUR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "brown",
    "black", "white", "grey", "gray", "pink", "gold", "golden", "silver",
    "dark", "light", "pale", "bright", "vivid", "deep", "rich", "muted",
    "crimson", "azure", "emerald", "ivory", "ebony", "scarlet", "ochre",
    "sepia", "monochrome", "chromatic", "tinted", "shaded", "shadowed"
}

TEXTURE_WORDS = {
    "rough", "smooth", "textured", "patterned", "decorated", "ornate",
    "elaborate", "intricate", "detailed", "fine", "coarse", "delicate",
    "dense", "sparse", "layered", "carved", "engraved", "etched",
    "embossed", "stippled", "hatched", "crosshatched", "lined", "dotted"
}

HEDGING_WORDS = {
    "appears", "seems", "possibly", "perhaps", "likely", "probably",
    "suggests", "indicates", "may", "might", "could", "resembles",
    "depicted", "shown", "represented", "apparently", "presumably",
    "seemingly", "possibly", "arguably", "conceivably"
}

VISUAL_FRAMING = {
    "composition", "scene", "view", "perspective", "depiction",
    "illustration", "drawing", "engraving", "image", "portrait",
    "figure", "panel", "plate", "vignette", "sketch", "print",
    "etching", "lithograph", "woodcut", "frontispiece", "tailpiece"
}


def get_dependency_depth(token):
    depth = 0
    current = token
    while current.head != current:
        current = current.head
        depth += 1
    return depth


def _extract_from_doc(doc, text):
    tokens    = [t for t in doc if not t.is_space]
    words     = [t for t in tokens if t.is_alpha]
    sentences = list(doc.sents)

    if len(words) == 0 or len(sentences) == 0:
        return None

    word_texts_lower = [t.text.lower() for t in words]
    n_words = len(words)
    n_sents = len(sentences)

    pos_counts = Counter(t.pos_ for t in words)
    n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    n_verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    n_adjs  = pos_counts.get("ADJ", 0)
    n_advs  = pos_counts.get("ADV", 0)
    n_propn = pos_counts.get("PROPN", 0)
    n_preps = sum(1 for t in tokens if t.dep_ == "prep")

    depths = [get_dependency_depth(t) for t in words]

    passive_sents = sum(
        1 for sent in sentences
        if any(t.dep_ in ("nsubjpass", "auxpass") for t in sent)
    )

    ent_labels = Counter(ent.label_ for ent in doc.ents)
    n_ents_total  = len(doc.ents)
    n_person_ents = ent_labels.get("PERSON", 0)
    n_loc_ents    = ent_labels.get("GPE", 0) + ent_labels.get("LOC", 0)
    n_org_ents    = ent_labels.get("ORG", 0)
    n_date_ents   = ent_labels.get("DATE", 0) + ent_labels.get("TIME", 0)
    n_work_ents   = ent_labels.get("WORK_OF_ART", 0)

    def density(word_list):
        return sum(1 for w in word_texts_lower if w in word_list) / n_words

    sent_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]

    f = {}

    # Lexical (12)
    f["f_token_count"]          = n_words
    f["f_unique_tokens"]        = len(set(word_texts_lower))
    f["f_type_token_ratio"]     = len(set(word_texts_lower)) / n_words
    f["f_avg_word_length"]      = np.mean([len(t.text) for t in words])
    f["f_long_word_ratio"]      = sum(1 for t in words if len(t.text) > 7) / n_words
    f["f_adj_noun_ratio"]       = n_adjs / max(n_nouns, 1)
    f["f_adj_density"]          = n_adjs / n_words
    f["f_noun_density"]         = n_nouns / n_words
    f["f_verb_density"]         = n_verbs / n_words
    f["f_adv_density"]          = n_advs / n_words
    f["f_propn_density"]        = n_propn / n_words
    f["f_content_word_ratio"]   = (n_nouns + n_verbs + n_adjs + n_advs) / n_words

    # Syntactic (9)
    f["f_sentence_count"]       = n_sents
    f["f_avg_sent_length"]      = np.mean(sent_lengths)
    f["f_max_sent_length"]      = max(sent_lengths)
    f["f_sent_length_std"]      = np.std(sent_lengths)
    f["f_avg_dep_depth"]        = np.mean(depths)
    f["f_max_dep_depth"]        = max(depths)
    f["f_passive_ratio"]        = passive_sents / n_sents
    f["f_prep_density"]         = n_preps / n_words
    f["f_comma_per_sent"]       = text.count(",") / n_sents

    # Semantic / NER (11)
    f["f_ent_density"]          = n_ents_total / n_words
    f["f_person_ent_density"]   = n_person_ents / n_words
    f["f_loc_ent_density"]      = n_loc_ents / n_words
    f["f_org_ent_density"]      = n_org_ents / n_words
    f["f_date_ent_density"]     = n_date_ents / n_words
    f["f_work_ent_density"]     = n_work_ents / n_words
    f["f_spatial_density"]      = density(SPATIAL_WORDS)
    f["f_colour_density"]       = density(COLOUR_WORDS)
    f["f_texture_density"]      = density(TEXTURE_WORDS)
    f["f_hedging_density"]      = density(HEDGING_WORDS)
    f["f_visual_framing"]       = density(VISUAL_FRAMING)

    # Discourse (8)
    f["f_colon_count"]          = text.count(":")
    f["f_semicolon_count"]      = text.count(";")
    f["f_parenthetical_count"]  = text.count("(") + text.count("[")
    f["f_exclamation_count"]    = text.count("!")
    f["f_question_count"]       = text.count("?")
    f["f_quote_count"]          = text.count('"')
    f["f_dash_count"]           = text.count("—") + text.count(" - ")
    f["f_number_density"]       = sum(1 for t in tokens if t.like_num) / n_words

    # Derived ratios (4)
    f["f_noun_verb_ratio"]      = n_nouns / max(n_verbs, 1)
    f["f_lex_density"]          = len(set(word_texts_lower)) / max(n_sents, 1)
    f["f_avg_ents_per_sent"]    = n_ents_total / n_sents
    f["f_modifier_ratio"]       = (n_adjs + n_advs) / max(n_nouns + n_verbs, 1)

    return f


def extract_features_single(text):
    """Extract features for a single text string."""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return None
    doc = nlp(text)
    return _extract_from_doc(doc, text)


def extract_features(text_series, batch_size=64, verbose=True):
    """
    Extract features for a pandas Series using spaCy batch processing.
    Returns a DataFrame with one row per text, one column per feature.
    """
    texts = text_series.fillna("").tolist()
    records = []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        for doc, raw_text in zip(nlp.pipe(batch, batch_size=batch_size), batch):
            f = _extract_from_doc(doc, raw_text)
            records.append(f if f is not None else {})

    if verbose:
        print(f"  Processing descriptions... {n}/{n} done.")

    df = pd.DataFrame(records).fillna(0)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    sample = (
        "A large ornate figure stands in the foreground, surrounded by intricate "
        "decorative patterns. The background shows a dark, textured surface carved "
        "with elaborate designs. Arthur Rackham's distinctive style is evident in "
        "the sinuous lines and detailed foliage."
    )
    f = extract_features_single(sample)
    print(f"\n{'Feature':<30} {'Value':>10}")
    print("-" * 42)
    for k, v in sorted(f.items()):
        print(f"  {k.replace('f_',''):<28} {v:>10.4f}")
    print(f"\nTotal features: {len(f)}")
"""Train and evaluate the improved multi-class sentiment model.

Usage (from the project root):
    python -m src.train

Reads  data/twitter_training.csv, data/twitter_validation.csv
Writes assets/models/sentiment_model.pkl and results/*
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import normalize

ROOT = Path(__file__).resolve().parent.parent
COLS = ["id", "entity", "sentiment", "text"]
LABELS = ["Negative", "Neutral", "Positive"]


def load(name):
    df = pd.read_csv(ROOT / "data" / name, header=None, names=COLS)
    df = df[df.sentiment != "Irrelevant"].dropna(subset=["text"])
    return df.reset_index(drop=True)


def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 3), min_df=2,
                                 sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w[\w']*\b|[!?]")),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                 ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=300_000)),
    ])


def build(clf):
    return Pipeline([("features", features()), ("clf", clf)])


def acc(model, df):
    return accuracy_score(df.sentiment, model.predict(df.text))


def main():
    tr, va = load("twitter_training.csv"), load("twitter_validation.csv")
    tr["group"] = tr.id.astype(str) + tr.entity

    # --- Protocol A: same split as the original notebook (random 80/20, seed 42)
    A, B = train_test_split(tr, test_size=0.2, random_state=42, stratify=tr.sentiment)
    # --- Protocol B: strict split, all paraphrases of one tweet stay together
    gi, gj = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(tr, groups=tr.group))
    GA, GB = tr.iloc[gi], tr.iloc[gj]

    candidates = {
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(C=30, max_iter=2000),
        "Linear SVM": LinearSVC(C=2.0),
    }
    comparison = []
    for name, clf in candidates.items():
        t = time.time()
        m = build(clf).fit(A.text, A.sentiment)
        p = m.predict(B.text)
        g = build(clf).fit(GA.text, GA.sentiment)
        row = {"model": name,
               "accuracy": accuracy_score(B.sentiment, p),
               "f1_weighted": f1_score(B.sentiment, p, average="weighted"),
               "strict_accuracy": acc(g, GB)}
        comparison.append(row)
        print(row, f"{time.time() - t:.0f}s", flush=True)

    # --- Final model: calibrated Linear SVM (gives real probabilities for the UI)
    final = build(CalibratedClassifierCV(LinearSVC(C=2.0), cv=5, method="sigmoid"))
    final.fit(A.text, A.sentiment)
    pred = final.predict(B.text)
    report = classification_report(B.sentiment, pred, output_dict=True)
    cm = pd.crosstab(B.sentiment, pred).reindex(index=LABELS, columns=LABELS, fill_value=0)

    strict = build(CalibratedClassifierCV(LinearSVC(C=2.0), cv=5, method="sigmoid")).fit(GA.text, GA.sentiment)

    metrics = {
        "final_model": "TF-IDF (word 1-3 + char 2-5) + calibrated Linear SVM",
        "train_samples": int(len(A)), "test_samples": int(len(B)),
        "accuracy": accuracy_score(B.sentiment, pred),
        "f1_weighted": f1_score(B.sentiment, pred, average="weighted"),
        "f1_macro": f1_score(B.sentiment, pred, average="macro"),
        "per_class": {k: report[k] for k in LABELS},
        "confusion_matrix": cm.values.tolist(), "labels": LABELS,
        "strict_group_split_accuracy": acc(strict, GB),
        "official_validation_accuracy": acc(final, va),
        "official_validation_samples": int(len(va)),
        "official_validation_text_overlap_with_train": float(va.text.isin(tr.text).mean()),
        "original_notebook_accuracy_reported": 0.76,
        "comparison": comparison,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(final, ROOT / "assets" / "models" / "sentiment_model.pkl", compress=("lzma", 9))

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    ConfusionMatrixDisplay(cm.values, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix  -  accuracy {metrics['accuracy']:.1%}")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png"); plt.close(fig)

    names = ["Original\nLR (notebook)"] + [c["model"].replace(" ", "\n", 1) for c in comparison]
    vals = [0.76] + [c["accuracy"] for c in comparison]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    bars = ax.bar(names, vals, color=["#9aa4b2"] + ["#3b6fd4"] * len(comparison))
    ax.axhline(0.95, ls="--", color="#d9534f", lw=1); ax.set_ylim(0.6, 1)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + .005, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Test accuracy"); ax.set_title("Model comparison (same 80/20 split)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "model_comparison.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix", "comparison")}, indent=2))


if __name__ == "__main__":
    main()

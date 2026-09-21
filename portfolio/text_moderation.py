"""Train a TF-IDF toxicity classifier with a validation-selected threshold."""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+|<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def choose_threshold(y, probabilities):
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y, probabilities >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def train(frame, output):
    if not {"text", "toxic"}.issubset(frame.columns):
        raise ValueError("CSV must contain text and toxic columns")
    frame = frame[["text", "toxic"]].dropna().copy()
    if set(frame.toxic.unique()) != {0, 1}:
        raise ValueError("toxic must contain both binary labels 0 and 1")
    frame["text"] = frame.text.map(clean_text)
    frame = frame[frame.text.str.len() > 0]
    conflicting = frame.groupby("text").toxic.nunique()
    frame = frame[~frame.text.isin(conflicting[conflicting > 1].index)]
    frame = frame.drop_duplicates("text")
    train_valid, test = train_test_split(frame, test_size=0.2, stratify=frame.toxic, random_state=42)
    training, valid = train_test_split(train_valid, test_size=0.25, stratify=train_valid.toxic, random_state=42)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=100_000)),
        ("classifier", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
    ])
    model.fit(training.text, training.toxic)
    threshold = choose_threshold(valid.toxic, model.predict_proba(valid.text)[:, 1])
    prediction = model.predict_proba(test.text)[:, 1] >= threshold
    report = {
        "evaluation": "stratified 60/20/20, deduplicated texts; threshold selected on validation",
        "seed": 42, "threshold": threshold,
        "rows": {"train": len(training), "validation": len(valid), "test": len(test)},
        "test_f1": float(f1_score(test.toxic, prediction, zero_division=0)),
        "test_precision": float(precision_score(test.toxic, prediction, zero_division=0)),
        "test_recall": float(recall_score(test.toxic, prediction, zero_division=0)),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default="artifacts/text-moderation", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(pd.read_csv(args.data), args.output), indent=2))

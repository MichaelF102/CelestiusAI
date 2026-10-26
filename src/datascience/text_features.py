"""Text classification with TF-IDF features and interpretable top terms."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TEXT_MODELS = {
    "Logistic Regression": lambda s: LogisticRegression(max_iter=2000, C=3.0),
    "Linear SVM": lambda s: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=s), cv=3),
    "Naive Bayes": lambda s: MultinomialNB(alpha=0.5),
    "SGD (linear)": lambda s: SGDClassifier(loss="log_loss", random_state=s, max_iter=2000),
}


class TextError(ValueError):
    """Invalid text-classification input (message is user-facing)."""


def text_summary(df: pd.DataFrame, text_col: str) -> dict[str, Any]:
    texts = df[text_col].dropna().astype(str)
    words = texts.str.lower().str.findall(r"[a-z']+")
    counts = pd.Series([w for ws in words for w in ws]).value_counts()
    return {
        "documents": len(texts),
        "mean_words": float(words.map(len).mean()),
        "median_chars": float(texts.str.len().median()),
        "vocabulary": int(counts.size),
        "empty": int((texts.str.strip() == "").sum()),
        "top_words": counts.head(15),
    }


def build_text_pipeline(
    model: str,
    ngram: tuple[int, int] = (1, 2),
    max_features: int = 20000,
    min_df: int = 2,
    seed: int = 42,
    stop_words: str | None = "english",
) -> Pipeline:
    if model not in TEXT_MODELS:
        raise TextError(f"Unknown model '{model}'.")
    vec = TfidfVectorizer(
        ngram_range=ngram,
        max_features=max_features,
        min_df=min_df,
        stop_words=stop_words,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", vec), ("clf", TEXT_MODELS[model](seed))])


def top_terms(pipe: Pipeline, n: int = 10) -> pd.DataFrame:
    """Most indicative terms per class (linear models and Naive Bayes only)."""
    clf = pipe.named_steps["clf"]
    terms = np.asarray(pipe.named_steps["tfidf"].get_feature_names_out())
    if hasattr(clf, "coef_"):
        coefs = clf.coef_ if clf.coef_.shape[0] > 1 else np.vstack([-clf.coef_[0], clf.coef_[0]])
    elif hasattr(clf, "feature_log_prob_"):
        coefs = clf.feature_log_prob_
    else:
        return pd.DataFrame(columns=["class", "term", "weight"])
    rows = []
    for k, cls in enumerate(clf.classes_):
        for i in np.argsort(-coefs[k])[:n]:
            rows.append({"class": str(cls), "term": terms[i], "weight": float(coefs[k][i])})
    return pd.DataFrame(rows)


def text_classification(
    df: pd.DataFrame,
    text_col: str,
    target: str,
    model: str = "Logistic Regression",
    ngram: tuple[int, int] = (1, 2),
    max_features: int = 20000,
    min_df: int = 2,
    cv: int = 5,
    seed: int = 42,
    stop_words: str | None = "english",
) -> dict[str, Any]:
    data = df[[text_col, target]].dropna()
    data = data[data[text_col].astype(str).str.strip() != ""]
    if len(data) < 30:
        raise TextError("Need at least 30 non-empty documents.")
    y = data[target].astype(str)
    if y.nunique() < 2 or y.value_counts().min() < 3:
        raise TextError("Every class needs at least 3 documents (and there must be 2+ classes).")
    X = data[text_col].astype(str)
    splitter = StratifiedKFold(
        n_splits=max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed
    )
    pipe = build_text_pipeline(model, ngram, max_features, min_df, seed, stop_words)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cross_validate(pipe, X, y, cv=splitter, scoring=["accuracy", "f1_macro"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held = build_text_pipeline(model, ngram, max_features, min_df, seed, stop_words).fit(
            X_tr, y_tr
        )
        final = build_text_pipeline(model, ngram, max_features, min_df, seed, stop_words).fit(X, y)
    pred = held.predict(X_te)
    labels = sorted(y.unique())
    base_rate = float(y.value_counts(normalize=True).max())
    return {
        "cv_accuracy": float(res["test_accuracy"].mean()),
        "cv_accuracy_std": float(res["test_accuracy"].std(ddof=1)),
        "cv_f1_macro": float(res["test_f1_macro"].mean()),
        "majority_baseline": base_rate,
        "report": pd.DataFrame(
            classification_report(y_te, pred, output_dict=True, zero_division=0)
        ).T.drop(index=["accuracy"], errors="ignore"),
        "confusion": pd.DataFrame(
            confusion_matrix(y_te, pred, labels=labels),
            index=[f"actual {c}" for c in labels],
            columns=[f"pred {c}" for c in labels],
        ),
        "top_terms": top_terms(final),
        "vocabulary": int(len(final.named_steps["tfidf"].vocabulary_)),
        "pipeline": final,
        "classes": labels,
    }


def predict_text(pipe: Pipeline, texts: list[str]) -> pd.DataFrame:
    proba = pipe.predict_proba(texts) if hasattr(pipe, "predict_proba") else None
    pred = pipe.predict(texts)
    out = pd.DataFrame({"text": texts, "prediction": pred})
    if proba is not None:
        out["confidence"] = proba.max(axis=1)
    return out

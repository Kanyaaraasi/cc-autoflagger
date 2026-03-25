"""Predict outcome from transcript, flag disagreements with labeled outcome."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class OutcomePredictor:
    """Train an outcome classifier on transcripts, detect disagreements."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self.le = LabelEncoder()
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        texts = df["transcript_text"].fillna("")
        X = self.tfidf.fit_transform(texts)
        y = self.le.fit_transform(df["outcome"])
        self.model.fit(X, y)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Must call fit() first"
        features = pd.DataFrame(index=df.index)

        texts = df["transcript_text"].fillna("")
        X = self.tfidf.transform(texts)

        # Predicted outcome
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        y_actual = self.le.transform(df["outcome"])

        # Disagreement flag
        features["outcome_disagreement"] = (y_pred != y_actual).astype(int)

        # Prediction entropy (uncertainty)
        entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        features["outcome_pred_entropy"] = entropy

        # Max class probability (confidence)
        features["outcome_pred_confidence"] = np.max(y_proba, axis=1)

        # Probability assigned to the actual outcome
        features["outcome_actual_prob"] = y_proba[np.arange(len(y_actual)), y_actual]

        return features

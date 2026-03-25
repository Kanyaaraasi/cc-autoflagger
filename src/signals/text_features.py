"""Text-based features from validation_notes and transcript_text."""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Issue-indicating keywords in validation_notes
ISSUE_KEYWORDS = [
    "mismatch", "error", "skipped", "incorrect", "missing",
    "medical advice", "discrepancy", "wrong", "miscategorized",
    "violated", "guardrail", "incomplete", "contradicts",
    "differs", "never asked", "not asked", "fabricated",
]


def _keyword_flags(text: str) -> dict:
    if pd.isna(text) or not text:
        return {f"vn_kw_{kw.replace(' ', '_')}": 0 for kw in ISSUE_KEYWORDS}
    text_lower = text.lower()
    return {f"vn_kw_{kw.replace(' ', '_')}": int(kw in text_lower) for kw in ISSUE_KEYWORDS}


def _validation_notes_length(text: str) -> int:
    if pd.isna(text) or not text:
        return 0
    return len(text.split())


class TextFeatureExtractor:
    """Fit on train, transform on any split."""

    def __init__(self, max_tfidf_features: int = 15):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vn = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        vn = df["validation_notes"].fillna("")
        self.tfidf_vn.fit(vn)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Must call fit() first"
        features = pd.DataFrame(index=df.index)

        # TF-IDF on validation_notes
        vn = df["validation_notes"].fillna("")
        tfidf_matrix = self.tfidf_vn.transform(vn)
        tfidf_cols = [f"tfidf_vn_{i}" for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
        features = pd.concat([features, tfidf_df], axis=1)

        # Keyword flags
        kw_flags = df["validation_notes"].apply(_keyword_flags).apply(pd.Series)
        kw_flags.index = df.index
        features = pd.concat([features, kw_flags], axis=1)

        # Validation notes length
        features["vn_word_count"] = df["validation_notes"].apply(_validation_notes_length)

        # Transcript length features
        features["transcript_word_count"] = df["transcript_text"].fillna("").apply(lambda x: len(x.split()))

        # Agent vs user text ratio
        def _agent_user_ratio(text):
            if pd.isna(text) or not text:
                return 0.5
            agent_parts = re.findall(r"\[AGENT\]:\s*(.*?)(?=\[USER\]:|$)", text, re.DOTALL)
            user_parts = re.findall(r"\[USER\]:\s*(.*?)(?=\[AGENT\]:|$)", text, re.DOTALL)
            a_len = sum(len(p.split()) for p in agent_parts)
            u_len = sum(len(p.split()) for p in user_parts)
            total = a_len + u_len
            return u_len / total if total > 0 else 0.5

        features["text_user_talk_ratio"] = df["transcript_text"].apply(_agent_user_ratio)

        return features

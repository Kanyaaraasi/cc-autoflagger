"""Text-based features from validation_notes and transcript_text.

Only keeps computed stats (word counts, talk ratio). TF-IDF and keyword
flags have been removed — they overfit to training vocabulary.
Semantic embeddings (embedding_features.py) replace them.
"""

import re

import pandas as pd


class TextFeatureExtractor:
    """Stateless text stats extractor (no fit needed, but kept for API compat)."""

    def __init__(self):
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Must call fit() first"
        features = pd.DataFrame(index=df.index)

        # Validation notes length
        features["vn_word_count"] = (
            df["validation_notes"].fillna("").apply(lambda x: len(x.split()))
        )

        # Transcript length
        features["transcript_word_count"] = (
            df["transcript_text"].fillna("").apply(lambda x: len(x.split()))
        )

        # Agent vs user text ratio
        def _agent_user_ratio(text):
            if pd.isna(text) or not text:
                return 0.5
            agent_parts = re.findall(
                r"\[AGENT\]:\s*(.*?)(?=\[USER\]:|$)", text, re.DOTALL
            )
            user_parts = re.findall(
                r"\[USER\]:\s*(.*?)(?=\[AGENT\]:|$)", text, re.DOTALL
            )
            a_len = sum(len(p.split()) for p in agent_parts)
            u_len = sum(len(p.split()) for p in user_parts)
            total = a_len + u_len
            return u_len / total if total > 0 else 0.5

        features["text_user_talk_ratio"] = df["transcript_text"].apply(
            _agent_user_ratio
        )

        return features

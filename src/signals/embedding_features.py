"""Semantic embedding features from validation_notes and transcript_text.

Uses sentence-transformers to produce dense embeddings that generalize
across vocabulary, replacing brittle TF-IDF features. PCA reduces
dimensionality to prevent overfitting on the small dataset.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingFeatureExtractor:
    """Fit PCA + positive centroid on train, transform any split."""

    def __init__(self, vn_components: int = 10, transcript_components: int = 10):
        self.vn_components = vn_components
        self.transcript_components = transcript_components
        self.model = None
        self.pca_vn = PCA(n_components=vn_components)
        self.pca_transcript = PCA(n_components=transcript_components)
        self.positive_centroid = None
        self._fitted = False

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        return self.model.encode(texts, show_progress_bar=False, batch_size=64)

    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None):
        """Fit PCA on training embeddings and compute positive centroid."""
        vn_texts = df["validation_notes"].fillna("").tolist()
        transcript_texts = df["transcript_text"].fillna("").tolist()

        vn_embeddings = self._encode(vn_texts)
        transcript_embeddings = self._encode(transcript_texts)

        self.pca_vn.fit(vn_embeddings)
        self.pca_transcript.fit(transcript_embeddings)

        # Compute positive-class centroid for validation_notes
        if y is not None and y.sum() > 0:
            positive_mask = y == 1
            positive_vn_emb = vn_embeddings[positive_mask]
            self.positive_centroid = positive_vn_emb.mean(axis=0, keepdims=True)
        else:
            self.positive_centroid = vn_embeddings.mean(axis=0, keepdims=True)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Must call fit() first"
        features = pd.DataFrame(index=df.index)

        vn_texts = df["validation_notes"].fillna("").tolist()
        transcript_texts = df["transcript_text"].fillna("").tolist()

        vn_embeddings = self._encode(vn_texts)
        transcript_embeddings = self._encode(transcript_texts)

        # PCA-reduced validation_notes embeddings
        vn_pca = self.pca_vn.transform(vn_embeddings)
        for i in range(vn_pca.shape[1]):
            features[f"emb_vn_pca_{i}"] = vn_pca[:, i]

        # PCA-reduced transcript embeddings
        tr_pca = self.pca_transcript.transform(transcript_embeddings)
        for i in range(tr_pca.shape[1]):
            features[f"emb_tr_pca_{i}"] = tr_pca[:, i]

        # Cosine similarity to positive centroid
        cos_sim = cosine_similarity(vn_embeddings, self.positive_centroid).flatten()
        features["emb_vn_positive_similarity"] = cos_sim

        return features

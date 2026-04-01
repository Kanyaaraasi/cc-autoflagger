"""Combine all signal extractors into a unified feature matrix."""

import pandas as pd
import numpy as np

from .config import DROP_COLS, CATEGORICAL_COLS, TARGET
from .logger import get_logger
from .signals import transcript_diff, number_checker, flow_checker
from .signals.text_features import TextFeatureExtractor
from .signals.outcome_predictor import OutcomePredictor
from .signals.embedding_features import EmbeddingFeatureExtractor

log = get_logger("features")

# Feature subset definitions for the ensemble
SUBSET_NUMERIC = "numeric"       # Model 1: LogReg — raw numerics + transcript_diff + one-hot
SUBSET_TEXT_LIGHT = "text_light"  # Model 2: LightGBM — numeric + flow + number + outcome
SUBSET_EMBEDDING = "embedding"   # Model 3: LightGBM — numeric + embeddings


class FeaturePipeline:
    """Fits on train, transforms any split into a feature matrix."""

    def __init__(self):
        self.text_extractor = TextFeatureExtractor()
        self.outcome_predictor = OutcomePredictor()
        self.embedding_extractor = EmbeddingFeatureExtractor()
        self._fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Fit stateful extractors on training data."""
        log.info("Fitting text feature extractor...")
        self.text_extractor.fit(train_df)
        log.info("Fitting outcome predictor...")
        self.outcome_predictor.fit(train_df)
        log.info("Fitting embedding extractor...")
        y_train = train_df[TARGET].astype(int).values if TARGET in train_df.columns else None
        self.embedding_extractor.fit(train_df, y_train)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, split_name: str = "") -> pd.DataFrame:
        """Extract all features from a dataframe."""
        assert self._fitted, "Must call fit() first"
        prefix = f"[{split_name}] " if split_name else ""

        log.info(f"{prefix}Extracting structured features...")
        structured = self._structured_features(df)

        log.info(f"{prefix}Extracting transcript diff signals...")
        diff = transcript_diff.extract(df)

        log.info(f"{prefix}Extracting number checker signals...")
        nums = number_checker.extract(df)

        log.info(f"{prefix}Extracting flow checker signals...")
        flow = flow_checker.extract(df)

        log.info(f"{prefix}Extracting text features...")
        text = self.text_extractor.transform(df)

        log.info(f"{prefix}Extracting outcome predictor signals...")
        outcome = self.outcome_predictor.transform(df)

        log.info(f"{prefix}Extracting embedding features...")
        embeddings = self.embedding_extractor.transform(df)

        all_features = pd.concat(
            [structured, diff, nums, flow, text, outcome, embeddings],
            axis=1,
        )

        log.info(f"{prefix}Total features: {all_features.shape[1]}")
        return all_features

    def _structured_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric + one-hot encoded categorical features."""
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        feat = df.drop(columns=cols_to_drop, errors="ignore")

        cat_cols = [c for c in CATEGORICAL_COLS if c in feat.columns]
        feat = pd.get_dummies(feat, columns=cat_cols, drop_first=False)

        bool_cols = feat.select_dtypes(include=["bool"]).columns
        feat[bool_cols] = feat[bool_cols].astype(int)

        feat = feat.fillna(0)
        feat = feat.select_dtypes(include=[np.number])

        # Derived features
        if "call_duration" in df.columns and "turn_count" in df.columns:
            feat["derived_duration_per_turn"] = (
                df["call_duration"] / df["turn_count"].replace(0, 1)
            )
        if "user_word_count" in df.columns and "agent_word_count" in df.columns:
            total_words = df["user_word_count"] + df["agent_word_count"]
            feat["derived_user_talk_ratio"] = (
                df["user_word_count"] / total_words.replace(0, 1)
            )
        if "call_duration" in df.columns and "response_completeness" in df.columns:
            feat["derived_duration_per_completeness"] = (
                df["call_duration"] / df["response_completeness"].replace(0, 0.01)
            )

        return feat

    def get_subset_columns(self, all_columns: list[str]) -> dict[str, list[str]]:
        """Return column lists for each model's feature subset."""
        all_cols = set(all_columns)

        # Identify column groups by prefix/name
        diff_cols = [c for c in all_cols if c.startswith("diff_")]
        num_cols = [c for c in all_cols if c.startswith("num_")]
        flow_cols = [c for c in all_cols if c.startswith("flow_")]
        outcome_cols = [c for c in all_cols if c.startswith("outcome_")]
        emb_cols = [c for c in all_cols if c.startswith("emb_")]
        text_stat_cols = [c for c in all_cols if c in (
            "vn_word_count", "transcript_word_count", "text_user_talk_ratio"
        )]

        # Structured = everything that's NOT a signal extractor output
        signal_cols = set(diff_cols + num_cols + flow_cols + outcome_cols + emb_cols + text_stat_cols)
        structured_cols = [c for c in all_cols if c not in signal_cols]

        # Model 1: numeric-only (structured + transcript_diff + text stats)
        numeric_subset = sorted(structured_cols + diff_cols + text_stat_cols)

        # Model 2: text-light (numeric + flow + number + outcome, no embeddings)
        text_light_subset = sorted(
            structured_cols + diff_cols + text_stat_cols +
            num_cols + flow_cols + outcome_cols
        )

        # Model 3: embedding (numeric + embeddings)
        embedding_subset = sorted(
            structured_cols + diff_cols + text_stat_cols + emb_cols
        )

        return {
            SUBSET_NUMERIC: numeric_subset,
            SUBSET_TEXT_LIGHT: text_light_subset,
            SUBSET_EMBEDDING: embedding_subset,
        }


def main():
    """CLI entry point: extract features and save."""
    from .data_loader import load_all
    from .config import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = load_all()

    pipeline = FeaturePipeline()
    pipeline.fit(train)

    X_train = pipeline.transform(train, split_name="train")
    X_val = pipeline.transform(val, split_name="val")
    X_test = pipeline.transform(test, split_name="test")

    common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    X_train.to_parquet(OUTPUT_DIR / "X_train.parquet")
    X_val.to_parquet(OUTPUT_DIR / "X_val.parquet")
    X_test.to_parquet(OUTPUT_DIR / "X_test.parquet")

    log.info(f"Saved features: {len(common_cols)} columns")
    log.info(f"  Train: {X_train.shape}")
    log.info(f"  Val:   {X_val.shape}")
    log.info(f"  Test:  {X_test.shape}")

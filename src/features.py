"""Combine all signal extractors into a unified feature matrix."""

import pandas as pd
import numpy as np

from .config import DROP_COLS, CATEGORICAL_COLS
from .logger import get_logger
from .signals import heuristics, transcript_diff, number_checker, flow_checker
from .signals.text_features import TextFeatureExtractor
from .signals.outcome_predictor import OutcomePredictor

log = get_logger("features")


class FeaturePipeline:
    """Fits on train, transforms any split into a feature matrix."""

    def __init__(self):
        self.text_extractor = TextFeatureExtractor(max_tfidf_features=15)
        self.outcome_predictor = OutcomePredictor()
        self._fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Fit stateful extractors on training data."""
        log.info("Fitting text feature extractor...")
        self.text_extractor.fit(train_df)
        log.info("Fitting outcome predictor...")
        self.outcome_predictor.fit(train_df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, split_name: str = "") -> pd.DataFrame:
        """Extract all features from a dataframe."""
        assert self._fitted, "Must call fit() first"
        prefix = f"[{split_name}] " if split_name else ""

        log.info(f"{prefix}Extracting structured features...")
        structured = self._structured_features(df)

        log.info(f"{prefix}Extracting heuristic signals...")
        heur = heuristics.extract(df)

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

        all_features = pd.concat(
            [structured, heur, diff, nums, flow, text, outcome],
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

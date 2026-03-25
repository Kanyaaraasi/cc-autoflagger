"""Tests for the feature pipeline and training."""

import numpy as np
import pandas as pd

from src.config import LEAKAGE_COLS, TARGET
from src.data_loader import load_all
from src.features import FeaturePipeline


class TestFeaturePipeline:
    def test_fit_transform(self):
        train, val, _ = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X = pipeline.transform(train, split_name="train")
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(train)
        assert X.shape[1] > 50  # should have many features

    def test_no_leakage_columns(self):
        train, _, _ = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X = pipeline.transform(train)
        for col in LEAKAGE_COLS:
            assert col not in X.columns, f"Leakage column in features: {col}"

    def test_no_target_in_features(self):
        train, _, _ = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X = pipeline.transform(train)
        assert TARGET not in X.columns

    def test_no_text_columns(self):
        train, _, _ = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X = pipeline.transform(train)
        text_cols = ["transcript_text", "validation_notes", "responses_json", "whisper_transcript"]
        for col in text_cols:
            assert col not in X.columns, f"Raw text column in features: {col}"

    def test_no_patient_state(self):
        """patient_state was dropped to reduce noise."""
        train, _, _ = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X = pipeline.transform(train)
        state_cols = [c for c in X.columns if "patient_state" in c]
        assert len(state_cols) == 0, f"patient_state features found: {state_cols}"

    def test_all_numeric(self):
        train, _, _ = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X = pipeline.transform(train)
        for col in X.columns:
            assert np.issubdtype(X[col].dtype, np.number), f"Non-numeric column: {col} ({X[col].dtype})"

    def test_no_nans(self):
        train, _, _ = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X = pipeline.transform(train)
        nan_cols = X.columns[X.isna().any()].tolist()
        assert len(nan_cols) == 0, f"NaN in columns: {nan_cols}"

    def test_val_test_alignment(self):
        """Train, val, and test should produce the same columns."""
        train, val, test = load_all()
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X_train = pipeline.transform(train)
        X_val = pipeline.transform(val)
        X_test = pipeline.transform(test)
        common = set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
        # At least 90% overlap
        assert len(common) / max(len(X_train.columns), 1) > 0.9

    def test_transform_before_fit_raises(self):
        pipeline = FeaturePipeline()
        train, _, _ = load_all()
        try:
            pipeline.transform(train)
            assert False, "Should have raised"
        except AssertionError:
            pass


class TestThresholdTuning:
    def test_find_best_threshold(self):
        from src.train import find_best_threshold
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])
        thresh, f1 = find_best_threshold(y_true, y_proba)
        assert 0.4 < thresh < 0.8
        assert f1 > 0.8

    def test_all_zeros(self):
        from src.train import find_best_threshold
        y_true = np.array([0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3])
        thresh, f1 = find_best_threshold(y_true, y_proba)
        assert f1 == 0.0

    def test_perfect_separation(self):
        from src.train import find_best_threshold
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        thresh, f1 = find_best_threshold(y_true, y_proba)
        assert f1 == 1.0

    def test_precision_threshold(self):
        from src.train import find_precision_threshold
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.3, 0.5, 0.6, 0.8, 0.9])
        thresh, prec = find_precision_threshold(y_true, y_proba, min_recall=0.66)
        assert prec >= 0.8

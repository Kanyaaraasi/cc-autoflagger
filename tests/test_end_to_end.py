"""End-to-end test: run full pipeline and validate submission output."""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import OUTPUT_DIR, MODEL_DIR, TARGET
from src.data_loader import load_all
from src.train import train_and_evaluate


class TestEndToEnd:
    def test_full_pipeline_produces_valid_submission(self):
        """Run training + prediction, validate submission.csv."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Train (includes feature extraction)
        best_name, best_thresh, subsets = train_and_evaluate()
        assert best_name == "ensemble"
        assert 0.0 < best_thresh < 1.0
        assert len(subsets) == 3

        # 2. Generate submission
        from src.predict import generate_submission
        submission = generate_submission()

        # 3. Validate submission format
        assert isinstance(submission, pd.DataFrame)
        assert list(submission.columns) == ["call_id", "predicted_ticket"]
        assert len(submission) == 159

        # call_ids should be UUIDs
        assert submission["call_id"].str.match(r"^[a-f0-9]{8}-").all()

        # predicted_ticket should be bool
        assert submission["predicted_ticket"].dtype == bool

        # Should have a reasonable number of predictions (5-25% of 159)
        flagged = submission["predicted_ticket"].sum()
        assert 5 <= flagged <= 40, f"Flagged {flagged}/159 — outside reasonable range"

        # No duplicates
        assert submission["call_id"].is_unique

        # All test call_ids present
        _, _, test_df = load_all()
        assert set(submission["call_id"]) == set(test_df["call_id"])

        # CSV file should exist
        csv_path = OUTPUT_DIR / "submission.csv"
        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        assert len(saved) == 159
        assert list(saved.columns) == ["call_id", "predicted_ticket"]

    def test_ensemble_models_saved_correctly(self):
        """Ensure all 3 models and config are saved."""
        with open(MODEL_DIR / "model.pkl", "rb") as f:
            payload = pickle.load(f)
        with open(MODEL_DIR / "config.json") as f:
            config = json.load(f)

        assert "models" in payload
        assert len(payload["models"]) == 3
        assert "threshold" in config
        assert "subsets" in config
        assert "cv_f1_mean" in config
        assert "fold_thresholds" in config
        assert len(config["fold_thresholds"]) == 10

    def test_cv_f1_above_threshold(self):
        """Ensure ensemble CV F1 is reasonable."""
        with open(MODEL_DIR / "config.json") as f:
            config = json.load(f)
        assert config["cv_f1_mean"] > 0.80, f"CV F1 = {config['cv_f1_mean']}, expected > 0.80"
        assert config["oof_metrics"]["f1"] > 0.80, f"OOF F1 = {config['oof_metrics']['f1']}, expected > 0.80"

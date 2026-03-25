"""End-to-end test: run full pipeline and validate submission output."""

import pandas as pd

from src.config import OUTPUT_DIR, MODEL_DIR, TARGET
from src.data_loader import load_all
from src.features import FeaturePipeline
from src.train import train_and_evaluate


class TestEndToEnd:
    def test_full_pipeline_produces_valid_submission(self):
        """Run feature extraction + training + prediction, validate submission.csv."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load data
        train, val, test = load_all()

        # 2. Extract features
        pipeline = FeaturePipeline()
        pipeline.fit(train)
        X_train = pipeline.transform(train, split_name="train")
        X_val = pipeline.transform(val, split_name="val")
        X_test = pipeline.transform(test, split_name="test")

        common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
        X_train[common_cols].to_parquet(OUTPUT_DIR / "X_train.parquet")
        X_val[common_cols].to_parquet(OUTPUT_DIR / "X_val.parquet")
        X_test[common_cols].to_parquet(OUTPUT_DIR / "X_test.parquet")

        # 3. Train
        best_name, best_thresh, columns = train_and_evaluate()
        assert best_thresh > 0.0
        assert best_thresh < 1.0
        assert len(columns) > 50

        # 4. Generate submission
        from src.predict import generate_submission
        submission = generate_submission()

        # 5. Validate submission format
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
        assert set(submission["call_id"]) == set(test["call_id"])

        # CSV file should exist
        csv_path = OUTPUT_DIR / "submission.csv"
        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        assert len(saved) == 159
        assert list(saved.columns) == ["call_id", "predicted_ticket"]

    def test_model_val_f1_above_threshold(self):
        """Ensure model achieves at least F1 > 0.85 on validation set."""
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score
        import json
        import pickle

        _, val_df, _ = load_all()
        y_val = val_df[TARGET].astype(int).values

        X_val = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")
        with open(MODEL_DIR / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODEL_DIR / "config.json") as f:
            config = json.load(f)

        columns = config["columns"]
        threshold = config["threshold"]

        proba = model.predict_proba(X_val[columns])[:, 1]
        pred = (proba >= threshold).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        prec = precision_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)

        print(f"\n{'='*50}")
        print(f"  Val F1:        {f1:.4f}")
        print(f"  Val Precision: {prec:.4f}")
        print(f"  Val Recall:    {rec:.4f}")
        print(f"  Threshold:     {threshold:.2f}")
        print(f"  Model:         {config['best_model']}")
        print(f"  Features:      {len(columns)}")
        print(f"{'='*50}")

        assert f1 > 0.85, f"Val F1 = {f1:.4f}, expected > 0.85"

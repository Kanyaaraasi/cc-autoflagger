"""CareCaller Ticket — Call Quality Auto-Flagger

Usage:
    uv run python main.py              # Full pipeline
    uv run python -m src.eda           # Exploratory data analysis
    uv run python -m src.features      # Extract features only
    uv run python -m src.train         # Train model only
    uv run python -m src.predict       # Generate submission only
"""

from src.data_loader import load_all
from src.features import FeaturePipeline
from src.train import train_and_evaluate
from src.predict import generate_submission
from src.config import OUTPUT_DIR
from src.logger import get_logger

log = get_logger("main")


def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    log.info("=" * 50)
    log.info("STEP 1: Loading data")
    train, val, test = load_all()
    log.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # 2. Extract features
    log.info("=" * 50)
    log.info("STEP 2: Extracting features")
    pipeline = FeaturePipeline()
    pipeline.fit(train)

    X_train = pipeline.transform(train, split_name="train")
    X_val = pipeline.transform(val, split_name="val")
    X_test = pipeline.transform(test, split_name="test")

    common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train[common_cols].to_parquet(OUTPUT_DIR / "X_train.parquet")
    X_val[common_cols].to_parquet(OUTPUT_DIR / "X_val.parquet")
    X_test[common_cols].to_parquet(OUTPUT_DIR / "X_test.parquet")
    log.info(f"Saved {len(common_cols)} features")

    # 3. Train + evaluate
    log.info("=" * 50)
    log.info("STEP 3: Training and evaluating")
    train_and_evaluate()

    # 4. Generate submission
    log.info("=" * 50)
    log.info("STEP 4: Generating submission")
    generate_submission()

    log.info("=" * 50)
    log.info("DONE")


if __name__ == "__main__":
    run_pipeline()

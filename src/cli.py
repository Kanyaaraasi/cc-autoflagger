"""CLI entry points for uv run <command>."""


def pipeline():
    from .data_loader import load_all
    from .features import FeaturePipeline
    from .train import train_and_evaluate
    from .predict import generate_submission
    from .config import OUTPUT_DIR
    from .logger import get_logger

    log = get_logger("main")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("STEP 1: Loading data")
    train, val, test = load_all()
    log.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    log.info("STEP 2: Extracting features")
    pipe = FeaturePipeline()
    pipe.fit(train)
    X_train = pipe.transform(train, split_name="train")
    X_val = pipe.transform(val, split_name="val")
    X_test = pipe.transform(test, split_name="test")

    common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train[common_cols].to_parquet(OUTPUT_DIR / "X_train.parquet")
    X_val[common_cols].to_parquet(OUTPUT_DIR / "X_val.parquet")
    X_test[common_cols].to_parquet(OUTPUT_DIR / "X_test.parquet")
    log.info(f"Saved {len(common_cols)} features")

    # Free NLI model and torch memory before training
    pipe.nli_checker.unload()
    del pipe
    import gc, torch
    gc.collect()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    log.info("STEP 3: Training and evaluating")
    train_and_evaluate()

    log.info("STEP 4: Generating submission")
    generate_submission()
    log.info("DONE")


def eda():
    from .eda import main
    main()


def extract():
    from .features import main
    main()


def train():
    from .train import main
    main()


def predict():
    from .predict import main
    main()

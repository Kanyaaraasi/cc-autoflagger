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

    log.info("STEP 3: Training and evaluating")
    train_and_evaluate()

    log.info("STEP 4: Generating submission")
    generate_submission()
    log.info("DONE")


def nli_extract():
    """Pre-compute NLI contradiction features for all splits."""
    from .data_loader import load_all
    from .signals.nli_checker import NLIChecker
    from .config import OUTPUT_DIR
    from .logger import get_logger

    log = get_logger("nli")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = load_all()
    checker = NLIChecker()
    checker.fit(train)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        log.info(f"Computing NLI features for {name} ({len(df)} calls)...")
        features = checker.transform(df)
        out_path = OUTPUT_DIR / f"nli_{name}.parquet"
        features.to_parquet(out_path)
        log.info(f"  Saved to {out_path} ({features.shape[1]} features)")


def stack():
    """Run stacking meta-learner: ML + NLI → final predictions."""
    import sys
    from .stack import stack_and_predict
    threshold = None
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
    stack_and_predict(threshold_override=threshold)


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

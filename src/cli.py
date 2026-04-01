"""CLI entry points for uv run <command>."""


def pipeline():
    from .train import train_and_evaluate
    from .predict import generate_submission
    from .logger import get_logger

    log = get_logger("main")

    log.info("STEP 1: Training ensemble (includes feature extraction)")
    train_and_evaluate()

    log.info("STEP 2: Generating submission")
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

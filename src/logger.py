"""Centralized logging for the pipeline."""

import logging
import sys
from pathlib import Path

from .config import OUTPUT_DIR


def setup_logger(name: str = "carecaller", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "pipeline.log"
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_logger(module: str) -> logging.Logger:
    return setup_logger(f"carecaller.{module}")

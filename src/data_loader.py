import json

import pandas as pd

from .config import TRAIN_CSV, VAL_CSV, TEST_CSV, TARGET

# V2 columns → V1 names so downstream code works unchanged
COLUMN_MAP = {
    "pipeline_mismatch_count": "whisper_mismatch_count",
    "pipeline_status": "whisper_status",
}


def load_split(path, is_test=False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)
    if not is_test:
        df[TARGET] = df[TARGET].astype(bool)
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = load_split(TRAIN_CSV)
    val = load_split(VAL_CSV)
    test = load_split(TEST_CSV, is_test=True)
    return train, val, test


def parse_responses(responses_json_str: str) -> list[dict]:
    """Parse the responses_json column into a list of {question, answer} dicts."""
    if pd.isna(responses_json_str) or not responses_json_str:
        return []
    try:
        return json.loads(responses_json_str)
    except (json.JSONDecodeError, TypeError):
        return []

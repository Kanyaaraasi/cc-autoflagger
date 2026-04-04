"""Verify response content against transcript and extract response quality signals."""

import json

import pandas as pd
import numpy as np


def _parse_responses(s):
    if pd.isna(s) or not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def extract(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    not_in_transcript = []
    empty_counts = []
    binary_ratios = []
    words_per_answered = []
    duration_per_answered = []

    for _, row in df.iterrows():
        responses = _parse_responses(row.get("responses_json", ""))
        transcript = str(row.get("transcript_text", "")).lower()
        answered = row.get("answered_count", 0) or 1

        # Response-transcript alignment
        found = 0
        total = 0
        empty = 0
        binary = 0
        total_answers = 0

        for r in responses:
            ans = str(r.get("answer", "")).strip()
            if not ans or ans.lower() == "nan":
                empty += 1
                continue

            total_answers += 1

            if ans.lower() in ("yes", "no"):
                binary += 1

            if len(ans) > 2:
                total += 1
                if ans.lower() in transcript:
                    found += 1

        ratio = 1.0 - (found / total) if total > 0 else 0.0
        not_in_transcript.append(ratio)
        empty_counts.append(empty)
        binary_ratios.append(binary / total_answers if total_answers > 0 else 0.0)
        words_per_answered.append(
            row.get("user_word_count", 0) / max(answered, 1)
        )
        duration_per_answered.append(
            row.get("call_duration", 0) / max(answered, 1)
        )

    features["resp_not_in_transcript"] = not_in_transcript
    features["resp_empty_count"] = empty_counts
    features["resp_binary_ratio"] = binary_ratios
    features["resp_words_per_answered"] = words_per_answered
    features["resp_duration_per_answered"] = duration_per_answered

    return features

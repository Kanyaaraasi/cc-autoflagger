"""Extract numbers from transcript and cross-validate against structured responses."""

import re

import pandas as pd

from ..data_loader import parse_responses

# Questions that expect numeric answers
NUMERIC_QUESTIONS = {
    "current weight": "weight",
    "height in feet": "height",
    "weight have you lost": "weight_lost",
    "goal weight": "goal_weight",
}

# Plausible ranges for health values
PLAUSIBLE_RANGES = {
    "weight": (50, 600),
    "height": None,  # complex format like 5'4
    "weight_lost": (-50, 100),
    "goal_weight": (50, 500),
}


def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric strings from text."""
    if pd.isna(text) or not text:
        return []
    return re.findall(r"\b\d+\.?\d*\b", str(text))


def _is_stt_number_error(n1: str, n2: str) -> bool:
    """Check if two numbers look like STT errors (substring, transposition)."""
    if n1 == n2:
        return False
    # One is a substring of the other (dropped leading/trailing digits)
    if n1 in n2 or n2 in n1:
        return True
    # Digit transposition
    if sorted(n1) == sorted(n2):
        return True
    return False


def _check_plausibility(value_str: str, field: str) -> bool:
    """Check if a numeric value is physiologically plausible."""
    rng = PLAUSIBLE_RANGES.get(field)
    if rng is None:
        return True
    try:
        val = float(value_str)
        return rng[0] <= val <= rng[1]
    except (ValueError, TypeError):
        return True


def extract(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    number_mismatches = []
    implausible_counts = []
    response_transcript_gaps = []

    for _, row in df.iterrows():
        responses = parse_responses(row.get("responses_json", ""))
        transcript = str(row.get("transcript_text", ""))
        transcript_numbers = _extract_numbers(transcript)

        mismatch_count = 0
        implausible_count = 0
        gap_count = 0

        for resp in responses:
            q = resp.get("question", "").lower()
            a = resp.get("answer", "")

            # Find which numeric field this is
            field = None
            for keyword, fname in NUMERIC_QUESTIONS.items():
                if keyword in q:
                    field = fname
                    break

            if not field or not a:
                continue

            answer_numbers = _extract_numbers(a)
            if not answer_numbers:
                continue

            # Check plausibility of recorded answer
            for num in answer_numbers:
                if not _check_plausibility(num, field):
                    implausible_count += 1

            # Check if any transcript number looks like an STT error vs recorded answer
            for ans_num in answer_numbers:
                for t_num in transcript_numbers:
                    if _is_stt_number_error(ans_num, t_num):
                        mismatch_count += 1
                        break

            # Check if answer exists in response but number not found near question in transcript
            if answer_numbers and not any(n in transcript for n in answer_numbers):
                gap_count += 1

        number_mismatches.append(mismatch_count)
        implausible_counts.append(implausible_count)
        response_transcript_gaps.append(gap_count)

    features["num_mismatches"] = number_mismatches
    features["num_implausible"] = implausible_counts
    features["num_response_transcript_gap"] = response_transcript_gaps

    return features

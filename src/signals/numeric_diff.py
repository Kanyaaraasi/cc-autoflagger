"""Numeric diff features: compare transcript-spoken numbers vs recorded responses."""

import json
import re

import pandas as pd
import numpy as np

from ..data_loader import parse_responses
from ..logger import get_logger

log = get_logger("numeric_diff")

# Questions with numeric answers and their transcript search patterns
NUMERIC_FIELDS = [
    {"key": "weight", "q_match": "current weight", "transcript_pattern": r"weight.*?\[USER\]:\s*(.*?)(?=\[AGENT\]|$)"},
    {"key": "height", "q_match": "height in feet", "transcript_pattern": r"height.*?\[USER\]:\s*(.*?)(?=\[AGENT\]|$)"},
    {"key": "weight_lost", "q_match": "weight have you lost", "transcript_pattern": r"(?:lost|lose).*?\[USER\]:\s*(.*?)(?=\[AGENT\]|$)"},
    {"key": "goal_weight", "q_match": "goal weight", "transcript_pattern": r"goal.*?weight.*?\[USER\]:\s*(.*?)(?=\[AGENT\]|$)"},
]


def _parse_spoken_number(text):
    """Extract a number from spoken text like 'two sixty two' or 'one eighty seven'."""
    if not text or pd.isna(text):
        return None

    text = text.strip().lower()

    # Try direct numeric extraction first
    direct = re.findall(r'\b(\d+\.?\d*)\b', text)
    if direct:
        try:
            return float(direct[0])
        except ValueError:
            pass

    # Try word2number for spoken numbers
    try:
        from word2number import w2n
        # Clean text for w2n
        cleaned = re.sub(r'[^\w\s-]', '', text)
        cleaned = cleaned.split('.')[0]  # Take first sentence
        # Only use first ~10 words (the number part)
        words = cleaned.split()[:10]
        cleaned = ' '.join(words)
        if any(c.isalpha() for c in cleaned):
            val = w2n.word_to_num(cleaned)
            return float(val)
    except (ValueError, Exception):
        pass

    return None


def _extract_recorded_number(answer_str):
    """Extract numeric value from a recorded response."""
    if not answer_str or pd.isna(answer_str):
        return None
    nums = re.findall(r'[\d.]+', str(answer_str))
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            pass
    return None


def _find_user_response(transcript, pattern):
    """Find what the user said in response to a question matching the pattern."""
    if not transcript or pd.isna(transcript):
        return None
    match = re.search(pattern, transcript, re.DOTALL | re.IGNORECASE)
    if match:
        response = match.group(1).strip()
        # Take just the first line/sentence
        response = response.split('\n')[0][:200]
        return response
    return None


def extract(df):
    """Extract numeric diff features for each call."""
    features = pd.DataFrame(index=df.index)

    all_max_gap = []
    all_max_ratio = []
    all_n_gaps = []
    all_has_big_gap = []
    all_weight_gap = []
    all_goal_gap = []

    for _, row in df.iterrows():
        responses = parse_responses(row.get("responses_json", ""))
        transcript = str(row.get("transcript_text", ""))

        gaps = []
        weight_gap = 0.0
        goal_gap = 0.0

        for field in NUMERIC_FIELDS:
            # Find recorded answer
            recorded = None
            for r in responses:
                if field["q_match"] in r.get("question", "").lower():
                    recorded = _extract_recorded_number(r.get("answer", ""))
                    break

            if recorded is None or recorded == 0:
                continue

            # Find what user said in transcript
            user_text = _find_user_response(transcript, field["transcript_pattern"])
            spoken = _parse_spoken_number(user_text) if user_text else None

            if spoken is None or spoken == 0:
                continue

            # Compute gap
            gap = abs(recorded - spoken)
            ratio = gap / max(spoken, 1)
            gaps.append({"field": field["key"], "gap": gap, "ratio": ratio,
                        "recorded": recorded, "spoken": spoken})

            if field["key"] == "weight":
                weight_gap = gap
            elif field["key"] == "goal_weight":
                goal_gap = gap

        # Aggregate features
        if gaps:
            max_gap = max(g["gap"] for g in gaps)
            max_ratio = max(g["ratio"] for g in gaps)
            n_significant = sum(1 for g in gaps if g["gap"] > 20)  # >20 lbs difference
        else:
            max_gap = 0.0
            max_ratio = 0.0
            n_significant = 0

        all_max_gap.append(max_gap)
        all_max_ratio.append(max_ratio)
        all_n_gaps.append(n_significant)
        all_has_big_gap.append(int(max_gap > 50))  # >50 lbs = likely real error
        all_weight_gap.append(weight_gap)
        all_goal_gap.append(goal_gap)

    features["ndiff_max_gap"] = all_max_gap
    features["ndiff_max_ratio"] = all_max_ratio
    features["ndiff_n_significant"] = all_n_gaps
    features["ndiff_has_big_gap"] = all_has_big_gap
    features["ndiff_weight_gap"] = all_weight_gap
    features["ndiff_goal_gap"] = all_goal_gap

    big_gaps = sum(all_has_big_gap)
    log.info(f"  Numeric diff: {big_gaps}/{len(df)} calls with gap > 50")

    return features

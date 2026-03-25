"""Compare transcript_text vs whisper_transcript to detect STT discrepancies."""

import re
from difflib import SequenceMatcher

import pandas as pd
from jiwer import wer, cer


def _normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_role_markers(transcript_text: str) -> str:
    """Remove [AGENT]: and [USER]: markers to get raw text."""
    if pd.isna(transcript_text):
        return ""
    return re.sub(r"\[(AGENT|USER)\]:\s*", "", transcript_text)


def _compute_wer(formatted: str, whisper: str) -> float:
    ref = _normalize(_strip_role_markers(formatted))
    hyp = _normalize(whisper)
    if not ref or not hyp:
        return 0.0
    try:
        return wer(ref, hyp)
    except Exception:
        return 0.0


def _compute_cer(formatted: str, whisper: str) -> float:
    ref = _normalize(_strip_role_markers(formatted))
    hyp = _normalize(whisper)
    if not ref or not hyp:
        return 0.0
    try:
        return cer(ref, hyp)
    except Exception:
        return 0.0


def _sequence_similarity(formatted: str, whisper: str) -> float:
    ref = _normalize(_strip_role_markers(formatted))
    hyp = _normalize(whisper)
    if not ref or not hyp:
        return 1.0
    return SequenceMatcher(None, ref, hyp).ratio()


def extract(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    features["diff_wer"] = df.apply(
        lambda r: _compute_wer(r["transcript_text"], r["whisper_transcript"]), axis=1
    )
    features["diff_cer"] = df.apply(
        lambda r: _compute_cer(r["transcript_text"], r["whisper_transcript"]), axis=1
    )
    features["diff_seq_similarity"] = df.apply(
        lambda r: _sequence_similarity(r["transcript_text"], r["whisper_transcript"]), axis=1
    )

    # Length difference ratio
    def _len_ratio(row):
        a = len(_normalize(_strip_role_markers(str(row["transcript_text"]))))
        b = len(_normalize(str(row["whisper_transcript"])))
        if max(a, b) == 0:
            return 0.0
        return abs(a - b) / max(a, b)

    features["diff_len_ratio"] = df.apply(_len_ratio, axis=1)

    return features

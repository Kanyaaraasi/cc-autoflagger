"""Predict ticket category from existing signal features."""

import pandas as pd


CATEGORIES = ["audio_issue", "elevenlabs", "openai"]


def predict_category(features_row: pd.Series, validation_notes: str = "") -> str:
    """Predict which module caused the issue based on signal features.

    Categories:
      audio_issue (10 in train): STT errors on critical health values.
        Signal: whisper_mismatch_count > 0

      elevenlabs (18 in train): Agent skipped questions OR gave medical advice.
        Signal: rule flags + validation_notes keywords (dosage guidance, not asked)

      openai (31 in train): Outcome miscategorization OR response contradictions.
        Signal: everything else (catch-all)
    """
    vn_lower = validation_notes.lower() if validation_notes else ""

    # ElevenLabs: agent skipped questions or gave medical advice
    if (
        features_row.get("rule_completed_low_completeness", 0) == 1
        or features_row.get("rule_medical_advice", 0) == 1
        or features_row.get("vn_kw_not_asked", 0) == 1
        or features_row.get("vn_kw_never_asked", 0) == 1
        or "dosage guidance" in vn_lower
        or "medical advice" in vn_lower
        or "guardrail" in vn_lower
    ):
        return "elevenlabs"

    # Audio/STT: only when whisper system actually flagged a mismatch
    if features_row.get("whisper_mismatch_count", 0) > 0:
        return "audio_issue"

    # OpenAI: outcome mislabeling, response contradictions, other agent logic
    return "openai"


def predict_categories(features_df: pd.DataFrame, predictions, validation_notes_series=None) -> list[str]:
    """Predict categories for all calls. Empty string for non-flagged calls."""
    categories = []
    for i, (idx, row) in enumerate(features_df.iterrows()):
        is_flagged = predictions.iloc[i] if hasattr(predictions, 'iloc') else predictions[i]
        if is_flagged:
            vn = str(validation_notes_series.iloc[i]) if validation_notes_series is not None else ""
            categories.append(predict_category(row, vn))
        else:
            categories.append("")
    return categories

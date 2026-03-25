"""Rule-based heuristic flags for ticket detection."""

import re

import pandas as pd

from ..data_loader import parse_responses

# Medical advice patterns (from agent turns only)
MEDICAL_ADVICE_PATTERNS = [
    r"\byou should take\b",
    r"\bi recommend\b",
    r"\bi suggest\b",
    r"\btry taking\b",
    r"\byou need to take\b",
    r"\bthe dosage should be\b",
    r"\bstop taking\b",
    r"\bincrease your\b.*\bdosage\b",
    r"\bdecrease your\b.*\bdosage\b",
    r"\byou could try\b",
    r"\bswitch to\b.*\bmedication\b",
]


def _extract_agent_text(transcript: str) -> str:
    """Extract only agent utterances from transcript."""
    if pd.isna(transcript):
        return ""
    agent_parts = re.findall(r"\[AGENT\]:\s*(.*?)(?=\[USER\]:|$)", transcript, re.DOTALL)
    return " ".join(agent_parts).lower()


def _has_medical_advice(agent_text: str) -> bool:
    return any(re.search(p, agent_text) for p in MEDICAL_ADVICE_PATTERNS)


def _count_questions_in_transcript(transcript: str) -> int:
    """Count how many of the 14 health questions appear in the agent's speech."""
    if pd.isna(transcript):
        return 0
    agent_text = _extract_agent_text(transcript).lower()
    question_keywords = [
        "feeling overall",
        "current weight",
        "height in feet",
        "weight have you lost",
        "side effects",
        "satisfied with your rate",
        "goal weight",
        "requests about your dosage",
        "new medications or supplements",
        "new medical conditions",
        "new allergies",
        "surgeries since",
        "questions for your doctor",
        "shipping address",
    ]
    return sum(1 for kw in question_keywords if kw in agent_text)


def _response_has_empty_but_transcript_has_answer(row: pd.Series) -> int:
    """Count responses recorded as empty despite patient answering in transcript."""
    responses = parse_responses(row.get("responses_json", ""))
    transcript = str(row.get("transcript_text", "")).lower()
    count = 0
    for r in responses:
        q = r.get("question", "").lower()
        a = r.get("answer", "")
        # Response empty but question was asked and user responded
        if not a and q:
            # check if a keyword from the question appears in transcript near a user response
            q_words = q.split()[:3]
            if any(w in transcript for w in q_words if len(w) > 3):
                count += 1
    return count


def extract(df: pd.DataFrame) -> pd.DataFrame:
    """Extract heuristic flag features from the dataframe."""
    features = pd.DataFrame(index=df.index)

    # Rule 1: completed but low completeness (skipped questions)
    features["rule_completed_low_completeness"] = (
        (df["outcome"] == "completed") & (df["response_completeness"] < 0.8)
    ).astype(int)

    # Rule 2: opted_out but high completeness (outcome miscategorization)
    features["rule_optout_high_completeness"] = (
        (df["outcome"] == "opted_out") & (df["response_completeness"] > 0.5)
    ).astype(int)

    # Rule 3: wrong_number but long conversation (misclassification)
    features["rule_wrongnum_long_convo"] = (
        (df["outcome"] == "wrong_number") & (df["turn_count"] > 10)
    ).astype(int)

    # Rule 4: whisper mismatch
    features["rule_whisper_mismatch"] = (df["whisper_mismatch_count"] > 0).astype(int)

    # Rule 5: medical advice in agent text
    agent_texts = df["transcript_text"].apply(_extract_agent_text)
    features["rule_medical_advice"] = agent_texts.apply(_has_medical_advice).astype(int)

    # Rule 6: questions asked in transcript vs answered_count mismatch
    questions_asked = df["transcript_text"].apply(_count_questions_in_transcript)
    features["rule_questions_asked_in_transcript"] = questions_asked
    features["rule_question_ask_answer_gap"] = (
        questions_asked - df["answered_count"]
    ).clip(lower=0)

    # Rule 7: completed but whisper was skipped
    features["rule_completed_whisper_skipped"] = (
        (df["outcome"] == "completed") & (df["whisper_status"] == "skipped")
    ).astype(int)

    # Rule 8: escalated calls (higher ticket rate)
    features["rule_is_escalated"] = (df["outcome"] == "escalated").astype(int)

    # Rule 9: form not submitted but call completed
    features["rule_completed_no_form"] = (
        (df["outcome"] == "completed") & (~df["form_submitted"])
    ).astype(int)

    # Rule 10: any heuristic fired (composite)
    flag_cols = [c for c in features.columns if c.startswith("rule_") and c != "rule_questions_asked_in_transcript"]
    features["rule_any_fired"] = (features[flag_cols].sum(axis=1) > 0).astype(int)
    features["rule_count_fired"] = features[flag_cols].sum(axis=1)

    return features

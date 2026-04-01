"""NLI-based contradiction detection between validation_notes and structured fields.

Uses cross-encoder/nli-deberta-v3-base to detect logical contradictions,
e.g. notes say "all 14 questions asked" but answered_count=11.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..logger import get_logger

log = get_logger("nli_checker")

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# Label mapping: 0=contradiction, 1=entailment, 2=neutral


class NLIChecker:
    """Check for contradictions between validation_notes and structured fields."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._fitted = False

    def _load_model(self):
        if self.model is None:
            log.info(f"Loading NLI model: {MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.eval()

    def fit(self, df: pd.DataFrame):
        self._load_model()
        self._fitted = True
        return self

    def _score_pair(self, premise: str, hypothesis: str) -> dict:
        """Score a single premise-hypothesis pair. Returns {contradiction, entailment, neutral}."""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        return {
            "contradiction": probs[0].item(),
            "entailment": probs[1].item(),
            "neutral": probs[2].item(),
        }

    def _build_hypotheses(self, row: pd.Series) -> list[tuple[str, str]]:
        """Build (name, hypothesis) pairs from structured fields."""
        hypotheses = []
        answered = row.get("answered_count", None)
        total = row.get("question_count", 14)
        outcome = row.get("outcome", "")
        completeness = row.get("response_completeness", None)
        notes = str(row.get("validation_notes", "")).lower()

        # H1: answered count — make numerical gap explicit
        if pd.notna(answered) and int(answered) < int(total):
            missing = int(total) - int(answered)
            hypotheses.append((
                "answered_count",
                f"Not all questions were answered. {missing} questions were skipped or left unanswered.",
            ))
        elif pd.notna(answered) and int(answered) == int(total):
            # Check if notes mention disconnection/partial despite full answers
            hypotheses.append((
                "answered_count",
                f"All {int(total)} questions were asked and every single one was answered.",
            ))

        # H2: outcome semantic check
        if outcome:
            outcome_desc = {
                "completed": "The call was completed successfully with all questions answered.",
                "incomplete": "The call ended early before all questions could be asked.",
                "opted_out": "The patient refused to continue and opted out of the questionnaire.",
                "wrong_number": "The call reached a wrong person who is not the intended patient.",
                "escalated": "The call was escalated because the patient needed further assistance.",
                "voicemail": "The call went to voicemail and no live conversation occurred.",
                "scheduled": "The call was rescheduled for a later time.",
            }
            desc = outcome_desc.get(outcome, f"The call outcome was {outcome}.")
            hypotheses.append(("outcome", desc))

        # H3: completeness contradiction — notes say partial but data says complete
        if pd.notna(completeness) and completeness >= 0.95:
            if any(phrase in notes for phrase in ["before disconnection", "before patient", "hung up", "remaining responses are empty"]):
                hypotheses.append((
                    "completeness",
                    "All responses were fully recorded with no missing data.",
                ))

        # H4: identity check for wrong_number
        if outcome == "wrong_number":
            hypotheses.append((
                "identity",
                "The person who answered the phone was not the intended patient.",
            ))

        # H5: data quality — check for fabrication/error language
        if any(kw in notes for kw in ["fabricat", "erroneously", "incorrectly", "recorded as", "differs"]):
            hypotheses.append((
                "data_quality",
                "All recorded responses accurately match what the patient actually said.",
            ))

        return hypotheses

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Must call fit() first"
        self._load_model()

        features = pd.DataFrame(index=df.index)
        all_scores = []

        for idx, row in df.iterrows():
            notes = row.get("validation_notes", "")
            if pd.isna(notes) or not notes:
                all_scores.append({
                    "nli_max_contradiction": 0.0,
                    "nli_answered_count_contradiction": 0.0,
                    "nli_outcome_contradiction": 0.0,
                    "nli_completeness_contradiction": 0.0,
                    "nli_num_contradictions": 0,
                    "nli_mean_entailment": 1.0,
                })
                continue

            hypotheses = self._build_hypotheses(row)
            contradiction_scores = {}
            entailment_scores = []

            for name, hypothesis in hypotheses:
                scores = self._score_pair(notes, hypothesis)
                contradiction_scores[name] = scores["contradiction"]
                entailment_scores.append(scores["entailment"])

            all_scores.append({
                "nli_max_contradiction": max(contradiction_scores.values()) if contradiction_scores else 0.0,
                "nli_answered_count_contradiction": contradiction_scores.get("answered_count", 0.0),
                "nli_outcome_contradiction": contradiction_scores.get("outcome", 0.0),
                "nli_completeness_contradiction": contradiction_scores.get("completeness", 0.0),
                "nli_num_contradictions": sum(1 for v in contradiction_scores.values() if v > 0.5),
                "nli_mean_entailment": float(np.mean(entailment_scores)) if entailment_scores else 1.0,
            })

        scores_df = pd.DataFrame(all_scores, index=df.index)
        return scores_df

    def unload(self):
        """Free model memory after feature extraction is done."""
        self.model = None
        self.tokenizer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

"""NLI-based contradiction detection between validation_notes and structured fields.

Uses cross-encoder/nli-deberta-v3-base to detect logical contradictions,
e.g. notes say "all 14 questions asked" but answered_count=11.
"""

import time

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

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        if self.model is None:
            self.device = self._get_device()
            log.info(f"Loading NLI model: {MODEL_NAME} (device={self.device})")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.to(self.device)
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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()
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

        # H1: answered count — skip for escalated (default values, not real data)
        if outcome != "escalated" and pd.notna(answered):
            if int(answered) < int(total):
                missing = int(total) - int(answered)
                hypotheses.append((
                    "answered_count",
                    f"Not all questions were answered. {missing} questions were skipped or left unanswered.",
                ))
            elif int(answered) == int(total):
                hypotheses.append((
                    "answered_count",
                    f"All {int(total)} questions were asked and every single one was answered.",
                ))

        # H2: outcome semantic check — skip escalated (handled separately)
        if outcome and outcome != "escalated":
            outcome_desc = {
                "completed": "The call was completed successfully with all questions answered.",
                "incomplete": "The call ended early before all questions could be asked.",
                "opted_out": "The patient refused to continue and opted out of the questionnaire.",
                "wrong_number": "The call reached a wrong person who is not the intended patient.",
                "voicemail": "The call went to voicemail and no live conversation occurred.",
                "scheduled": "The call was rescheduled for a later time.",
            }
            desc = outcome_desc.get(outcome, f"The call outcome was {outcome}.")
            hypotheses.append(("outcome", desc))

        # H3: completeness contradiction — notes say partial but data says complete
        if outcome != "escalated" and pd.notna(completeness) and completeness >= 0.95:
            if any(phrase in notes for phrase in ["before disconnection", "before patient", "hung up", "remaining responses are empty"]):
                hypotheses.append((
                    "completeness",
                    "All responses were fully recorded with no missing data.",
                ))

        # H4: identity check for wrong_number — works even when answered=0
        if outcome == "wrong_number":
            hypotheses.append((
                "identity",
                "The call was answered by a stranger who is not the patient.",
            ))

        # H5: data quality — fabrication/error language
        if any(kw in notes for kw in ["fabricat", "erroneously", "incorrectly", "recorded as", "differs"]):
            hypotheses.append((
                "data_quality",
                "All recorded responses accurately match what the patient actually said.",
            ))

        # H6: outcome corrected by validation AI
        if any(phrase in notes for phrase in ["corrected by validation", "outcome was corrected", "outcome corrected"]):
            hypotheses.append((
                "outcome",
                "The call outcome was correctly classified from the start without any corrections needed.",
            ))

        # H7: medical advice / dosage guidance (policy violation)
        if any(phrase in notes for phrase in ["dosage guidance", "dosage recommendation", "provided dosage", "medication recommendation"]):
            hypotheses.append((
                "outcome",
                "The agent did not provide any medical advice or medication dosage recommendations during the call.",
            ))

        return hypotheses

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Must call fit() first"
        self._load_model()

        log.info(f"Running NLI on {len(df)} calls (device={self.device})")
        t0 = time.time()

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

        elapsed = time.time() - t0
        log.info(f"NLI done: {len(df)} calls in {elapsed:.1f}s ({elapsed/len(df)*1000:.0f}ms/call)")

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

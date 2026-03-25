"""Tests for signal extractors."""

import pandas as pd
import numpy as np

from src.signals import heuristics, transcript_diff, number_checker, flow_checker
from src.signals.text_features import TextFeatureExtractor
from src.signals.outcome_predictor import OutcomePredictor


class TestHeuristics:
    def test_output_shape(self, sample_call):
        result = heuristics.extract(sample_call)
        assert len(result) == 1
        assert result.shape[1] == 12

    def test_normal_call_no_flags(self, sample_call):
        result = heuristics.extract(sample_call)
        assert result["rule_completed_low_completeness"].iloc[0] == 0
        assert result["rule_whisper_mismatch"].iloc[0] == 0
        assert result["rule_medical_advice"].iloc[0] == 0

    def test_skipped_questions_flag(self, ticket_call_skipped_questions):
        result = heuristics.extract(ticket_call_skipped_questions)
        assert result["rule_completed_low_completeness"].iloc[0] == 1

    def test_whisper_mismatch_flag(self, ticket_call_stt_error):
        result = heuristics.extract(ticket_call_stt_error)
        assert result["rule_whisper_mismatch"].iloc[0] == 1

    def test_composite_flags(self, ticket_call_skipped_questions):
        result = heuristics.extract(ticket_call_skipped_questions)
        assert result["rule_any_fired"].iloc[0] == 1
        assert result["rule_count_fired"].iloc[0] >= 1

    def test_medical_advice_detection(self):
        df = pd.DataFrame([{
            "outcome": "completed",
            "response_completeness": 1.0,
            "whisper_mismatch_count": 0,
            "whisper_status": "completed",
            "turn_count": 30,
            "form_submitted": True,
            "answered_count": 14,
            "transcript_text": "[AGENT]: I recommend you should take ibuprofen for that. [USER]: Ok.",
            "responses_json": "[]",
        }])
        result = heuristics.extract(df)
        assert result["rule_medical_advice"].iloc[0] == 1


class TestTranscriptDiff:
    def test_output_columns(self, sample_call):
        result = transcript_diff.extract(sample_call)
        assert "diff_wer" in result.columns
        assert "diff_cer" in result.columns
        assert "diff_seq_similarity" in result.columns
        assert "diff_len_ratio" in result.columns

    def test_identical_transcripts_low_wer(self, sample_call):
        result = transcript_diff.extract(sample_call)
        # Formatted vs whisper should be very similar
        assert result["diff_wer"].iloc[0] < 0.1
        assert result["diff_seq_similarity"].iloc[0] > 0.9

    def test_nan_transcript(self):
        df = pd.DataFrame([{
            "transcript_text": None,
            "whisper_transcript": None,
        }])
        result = transcript_diff.extract(df)
        assert result["diff_wer"].iloc[0] == 0.0


class TestNumberChecker:
    def test_output_columns(self, sample_call):
        result = number_checker.extract(sample_call)
        assert "num_mismatches" in result.columns
        assert "num_implausible" in result.columns
        assert "num_response_transcript_gap" in result.columns

    def test_normal_call_no_issues(self, sample_call):
        result = number_checker.extract(sample_call)
        assert result["num_implausible"].iloc[0] == 0

    def test_stt_error_detected(self, ticket_call_stt_error):
        result = number_checker.extract(ticket_call_stt_error)
        # "62" in response vs "262" in transcript — substring match
        assert result["num_mismatches"].iloc[0] >= 1

    def test_implausible_weight(self):
        df = pd.DataFrame([{
            "transcript_text": "[AGENT]: What's your current weight in pounds? [USER]: 10.",
            "responses_json": '[{"question": "What\'s your current weight in pounds?", "answer": "10"}]',
        }])
        result = number_checker.extract(df)
        assert result["num_implausible"].iloc[0] >= 1


class TestFlowChecker:
    def test_output_columns(self, sample_call):
        result = flow_checker.extract(sample_call)
        expected = ["flow_edit_distance", "flow_missing_states", "flow_actual_state_count",
                    "flow_question_states_found", "flow_question_coverage"]
        for col in expected:
            assert col in result.columns

    def test_complete_call_high_coverage(self, sample_call):
        result = flow_checker.extract(sample_call)
        assert result["flow_question_coverage"].iloc[0] > 0.5

    def test_short_call_low_coverage(self, ticket_call_skipped_questions):
        result = flow_checker.extract(ticket_call_skipped_questions)
        coverage = result["flow_question_coverage"].iloc[0]
        assert coverage < 0.5  # only a few questions asked

    def test_nan_transcript(self):
        df = pd.DataFrame([{"transcript_text": None}])
        result = flow_checker.extract(df)
        assert result["flow_actual_state_count"].iloc[0] == 0


class TestTextFeatures:
    def test_fit_transform(self):
        """Need multiple rows for TF-IDF min_df=2."""
        from src.data_loader import load_all
        train, _, _ = load_all()
        ext = TextFeatureExtractor(max_tfidf_features=5)
        ext.fit(train)
        result = ext.transform(train.head(5))
        assert "vn_word_count" in result.columns
        assert "transcript_word_count" in result.columns
        assert "text_user_talk_ratio" in result.columns

    def test_keyword_flags(self, ticket_call_skipped_questions):
        """Keyword flags work on single rows (no TF-IDF needed)."""
        from src.signals.text_features import _keyword_flags
        notes = "Agent marked call complete but 2+ required questions were never asked."
        flags = _keyword_flags(notes)
        assert flags["vn_kw_never_asked"] == 1
        assert flags["vn_kw_incomplete"] == 0

    def test_transform_before_fit_raises(self, sample_call):
        ext = TextFeatureExtractor()
        try:
            ext.transform(sample_call)
            assert False, "Should have raised"
        except AssertionError:
            pass


class TestOutcomePredictor:
    def test_fit_transform(self):
        from src.data_loader import load_all
        train, val, _ = load_all()
        pred = OutcomePredictor()
        pred.fit(train)
        result = pred.transform(val)
        assert "outcome_disagreement" in result.columns
        assert "outcome_pred_entropy" in result.columns
        assert "outcome_pred_confidence" in result.columns
        assert "outcome_actual_prob" in result.columns
        assert len(result) == len(val)
        # Confidence should be between 0 and 1
        assert result["outcome_pred_confidence"].between(0, 1).all()

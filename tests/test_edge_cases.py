"""Edge case tests — adversarial, empty, and boundary inputs."""

import pandas as pd
import numpy as np

from src.signals import transcript_diff, number_checker, flow_checker
from src.signals.text_features import TextFeatureExtractor
from src.data_loader import parse_responses


def _make_row(**overrides):
    """Create a minimal valid call row with overrides."""
    defaults = {
        "call_id": "edge-001",
        "outcome": "completed",
        "call_duration": 100,
        "attempt_number": 1,
        "direction": "outbound",
        "whisper_status": "completed",
        "whisper_mismatch_count": 0,
        "organization_id": "org_001",
        "product_id": "prod_001",
        "patient_state": "CA",
        "cycle_status": "active",
        "form_submitted": True,
        "patient_name_anon": "Test User",
        "question_count": 14,
        "answered_count": 0,
        "response_completeness": 0.0,
        "turn_count": 2,
        "user_turn_count": 0,
        "agent_turn_count": 2,
        "user_word_count": 0,
        "agent_word_count": 10,
        "avg_user_turn_words": 0.0,
        "avg_agent_turn_words": 5.0,
        "interruption_count": 0,
        "max_time_in_call": 100,
        "hour_of_day": 12,
        "day_of_week": "Monday",
        "transcript_text": "",
        "validation_notes": "",
        "responses_json": "[]",
        "whisper_transcript": "",
        "has_ticket": False,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# === TRANSCRIPT DIFF EDGE CASES ===

class TestTranscriptDiffEdgeCases:
    def test_empty_strings(self):
        df = _make_row(transcript_text="", whisper_transcript="")
        result = transcript_diff.extract(df)
        assert result["diff_wer"].iloc[0] == 0.0
        assert result["diff_len_ratio"].iloc[0] == 0.0

    def test_one_nan_one_valid(self):
        df = _make_row(transcript_text=None, whisper_transcript="hello world")
        result = transcript_diff.extract(df)
        assert result["diff_wer"].iloc[0] == 0.0

    def test_identical_texts(self):
        text = "[AGENT]: Hello there [USER]: Hi"
        whisper = "Hello there Hi"
        df = _make_row(transcript_text=text, whisper_transcript=whisper)
        result = transcript_diff.extract(df)
        assert result["diff_seq_similarity"].iloc[0] > 0.8

    def test_completely_different_texts(self):
        df = _make_row(
            transcript_text="[AGENT]: Alpha bravo charlie",
            whisper_transcript="Delta echo foxtrot",
        )
        result = transcript_diff.extract(df)
        assert result["diff_wer"].iloc[0] > 0.5
        assert result["diff_seq_similarity"].iloc[0] < 0.5


# === NUMBER CHECKER EDGE CASES ===

class TestNumberCheckerEdgeCases:
    def test_empty_responses_json(self):
        df = _make_row(responses_json="[]", transcript_text="[AGENT]: Hello [USER]: Hi")
        result = number_checker.extract(df)
        assert result["num_mismatches"].iloc[0] == 0

    def test_nan_responses_json(self):
        df = _make_row(responses_json=None, transcript_text="[AGENT]: Hello")
        result = number_checker.extract(df)
        assert result["num_mismatches"].iloc[0] == 0

    def test_non_numeric_answers_ignored(self):
        df = _make_row(
            responses_json='[{"question": "How have you been feeling overall?", "answer": "Pretty good"}]',
            transcript_text="[AGENT]: How are you? [USER]: Pretty good",
        )
        result = number_checker.extract(df)
        assert result["num_implausible"].iloc[0] == 0

    def test_weight_at_boundary(self):
        df = _make_row(
            responses_json='[{"question": "What\'s your current weight in pounds?", "answer": "50"}]',
            transcript_text="[AGENT]: current weight? [USER]: 50",
        )
        result = number_checker.extract(df)
        assert result["num_implausible"].iloc[0] == 0

    def test_weight_below_boundary(self):
        df = _make_row(
            responses_json='[{"question": "What\'s your current weight in pounds?", "answer": "10"}]',
            transcript_text="[AGENT]: current weight? [USER]: 10",
        )
        result = number_checker.extract(df)
        assert result["num_implausible"].iloc[0] >= 1

    def test_transposition_detection(self):
        df = _make_row(
            responses_json='[{"question": "What\'s your current weight in pounds?", "answer": "216"}]',
            transcript_text="[AGENT]: current weight? [USER]: 261",
        )
        result = number_checker.extract(df)
        assert result["num_mismatches"].iloc[0] >= 1


# === FLOW CHECKER EDGE CASES ===

class TestFlowCheckerEdgeCases:
    def test_empty_transcript(self):
        df = _make_row(transcript_text="")
        result = flow_checker.extract(df)
        assert result["flow_actual_state_count"].iloc[0] == 0
        assert result["flow_question_coverage"].iloc[0] == 0.0

    def test_only_greeting(self):
        df = _make_row(transcript_text="[AGENT]: Thanks for calling TrimRX.")
        result = flow_checker.extract(df)
        assert result["flow_actual_state_count"].iloc[0] >= 1
        assert result["flow_question_states_found"].iloc[0] == 0

    def test_single_turn(self):
        df = _make_row(transcript_text="[AGENT]: Hello")
        result = flow_checker.extract(df)
        assert result["flow_missing_states"].iloc[0] > 10


# === TEXT FEATURES EDGE CASES ===

class TestTextFeaturesEdgeCases:
    def test_empty_validation_notes(self):
        df = _make_row(validation_notes="")
        ext = TextFeatureExtractor()
        ext.fit(df)
        result = ext.transform(df)
        assert result["vn_word_count"].iloc[0] == 0

    def test_nan_validation_notes(self):
        df = _make_row(validation_notes=None)
        ext = TextFeatureExtractor()
        ext.fit(df)
        result = ext.transform(df)
        assert result["vn_word_count"].iloc[0] == 0


# === PARSE RESPONSES EDGE CASES ===

class TestParseResponsesEdgeCases:
    def test_malformed_json_array(self):
        assert parse_responses("[{broken}]") == []

    def test_single_response(self):
        result = parse_responses('[{"question": "Q1", "answer": "A1"}]')
        assert len(result) == 1

    def test_empty_answers(self):
        result = parse_responses('[{"question": "Q1", "answer": ""}]')
        assert len(result) == 1
        assert result[0]["answer"] == ""

    def test_unicode_in_response(self):
        result = parse_responses('[{"question": "Weight?", "answer": "70 kg"}]')
        assert result[0]["answer"] == "70 kg"

    def test_nested_quotes(self):
        result = parse_responses('[{"question": "How?", "answer": "I\'m fine"}]')
        assert "fine" in result[0]["answer"]


# === THRESHOLD EDGE CASES ===

class TestThresholdEdgeCases:
    def test_all_same_probability(self):
        from src.train import find_best_threshold
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        thresh, f1 = find_best_threshold(y_true, y_proba)
        assert isinstance(thresh, float)
        assert isinstance(f1, float)

    def test_single_sample(self):
        from src.train import find_best_threshold
        y_true = np.array([1])
        y_proba = np.array([0.8])
        thresh, f1 = find_best_threshold(y_true, y_proba)
        assert f1 > 0

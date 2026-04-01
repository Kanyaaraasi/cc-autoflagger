"""Tests for signal extractors."""

import pandas as pd
import numpy as np

from src.signals import transcript_diff, number_checker, flow_checker
from src.signals.text_features import TextFeatureExtractor
from src.signals.outcome_predictor import OutcomePredictor
from src.signals.embedding_features import EmbeddingFeatureExtractor


class TestTranscriptDiff:
    def test_output_columns(self, sample_call):
        result = transcript_diff.extract(sample_call)
        assert "diff_wer" in result.columns
        assert "diff_cer" in result.columns
        assert "diff_seq_similarity" in result.columns
        assert "diff_len_ratio" in result.columns

    def test_identical_transcripts_low_wer(self, sample_call):
        result = transcript_diff.extract(sample_call)
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
        assert coverage < 0.5

    def test_nan_transcript(self):
        df = pd.DataFrame([{"transcript_text": None}])
        result = flow_checker.extract(df)
        assert result["flow_actual_state_count"].iloc[0] == 0


class TestTextFeatures:
    def test_fit_transform(self):
        from src.data_loader import load_all
        train, _, _ = load_all()
        ext = TextFeatureExtractor()
        ext.fit(train)
        result = ext.transform(train.head(5))
        assert "vn_word_count" in result.columns
        assert "transcript_word_count" in result.columns
        assert "text_user_talk_ratio" in result.columns
        # No TF-IDF or keyword features
        tfidf_cols = [c for c in result.columns if c.startswith("tfidf_")]
        kw_cols = [c for c in result.columns if c.startswith("vn_kw_")]
        assert len(tfidf_cols) == 0, f"TF-IDF features should be removed: {tfidf_cols}"
        assert len(kw_cols) == 0, f"Keyword features should be removed: {kw_cols}"

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
        assert result["outcome_pred_confidence"].between(0, 1).all()


class TestEmbeddingFeatures:
    def test_fit_transform(self):
        from src.data_loader import load_all
        train, _, _ = load_all()
        ext = EmbeddingFeatureExtractor(vn_components=5, transcript_components=5)
        y = train["has_ticket"].astype(int).values
        ext.fit(train, y)
        result = ext.transform(train.head(5))
        # Should have PCA features + cosine similarity
        vn_cols = [c for c in result.columns if c.startswith("emb_vn_pca_")]
        tr_cols = [c for c in result.columns if c.startswith("emb_tr_pca_")]
        assert len(vn_cols) == 5
        assert len(tr_cols) == 5
        assert "emb_vn_positive_similarity" in result.columns
        assert len(result) == 5

    def test_cosine_similarity_range(self):
        from src.data_loader import load_all
        train, _, _ = load_all()
        ext = EmbeddingFeatureExtractor(vn_components=3, transcript_components=3)
        y = train["has_ticket"].astype(int).values
        ext.fit(train, y)
        result = ext.transform(train)
        sim = result["emb_vn_positive_similarity"]
        assert sim.between(-1.01, 1.01).all(), f"Cosine similarity out of range: {sim.describe()}"

    def test_transform_before_fit_raises(self, sample_call):
        ext = EmbeddingFeatureExtractor()
        try:
            ext.transform(sample_call)
            assert False, "Should have raised"
        except AssertionError:
            pass

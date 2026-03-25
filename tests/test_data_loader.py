"""Tests for data_loader module."""

import pandas as pd
from src.data_loader import load_all, parse_responses


class TestLoadAll:
    def test_returns_three_dataframes(self):
        train, val, test = load_all()
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_sizes(self):
        train, val, test = load_all()
        assert len(train) == 689
        assert len(val) == 144
        assert len(test) == 159

    def test_train_has_target(self):
        train, _, _ = load_all()
        assert "has_ticket" in train.columns
        assert train["has_ticket"].dtype == bool

    def test_train_ticket_count(self):
        train, _, _ = load_all()
        assert train["has_ticket"].sum() == 59

    def test_required_columns_exist(self):
        train, _, _ = load_all()
        required = [
            "call_id", "outcome", "call_duration", "transcript_text",
            "validation_notes", "responses_json", "whisper_transcript",
            "response_completeness", "whisper_mismatch_count",
        ]
        for col in required:
            assert col in train.columns, f"Missing column: {col}"


class TestParseResponses:
    def test_valid_json(self):
        json_str = '[{"question": "Weight?", "answer": "210"}]'
        result = parse_responses(json_str)
        assert len(result) == 1
        assert result[0]["answer"] == "210"

    def test_nan_input(self):
        assert parse_responses(float("nan")) == []

    def test_empty_string(self):
        assert parse_responses("") == []

    def test_invalid_json(self):
        assert parse_responses("not json") == []

    def test_none_input(self):
        assert parse_responses(None) == []

    def test_14_questions(self):
        train, _, _ = load_all()
        # Pick a completed call with full responses
        completed = train[train["response_completeness"] == 1.0].iloc[0]
        responses = parse_responses(completed["responses_json"])
        assert len(responses) == 14

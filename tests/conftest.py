"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_call():
    """A single normal completed call (no ticket)."""
    return pd.DataFrame([{
        "call_id": "test-001",
        "outcome": "completed",
        "call_duration": 180,
        "attempt_number": 1,
        "direction": "outbound",
        "whisper_status": "completed",
        "whisper_mismatch_count": 0,
        "organization_id": "org_001",
        "product_id": "prod_001",
        "patient_state": "CA",
        "cycle_status": "active",
        "form_submitted": True,
        "patient_name_anon": "John Doe",
        "question_count": 14,
        "answered_count": 14,
        "response_completeness": 1.0,
        "turn_count": 30,
        "user_turn_count": 14,
        "agent_turn_count": 16,
        "user_word_count": 80,
        "agent_word_count": 160,
        "avg_user_turn_words": 5.7,
        "avg_agent_turn_words": 10.0,
        "interruption_count": 1,
        "max_time_in_call": 180,
        "hour_of_day": 10,
        "day_of_week": "Monday",
        "transcript_text": (
            "[AGENT]: Thanks for calling TrimRX. Am I speaking with John Doe? "
            "[USER]: Yes, that's me. "
            "[AGENT]: Are you interested in getting your Semaglutide refill? "
            "[USER]: Yes. "
            "[AGENT]: How have you been feeling overall? "
            "[USER]: Pretty good. "
            "[AGENT]: What's your current weight in pounds? "
            "[USER]: 210. "
            "[AGENT]: What's your height in feet and inches? "
            "[USER]: 5'10. "
            "[AGENT]: How much weight have you lost this past month? "
            "[USER]: About 5 pounds. "
            "[AGENT]: Any side effects from your medication? "
            "[USER]: No. "
            "[AGENT]: Satisfied with your rate of weight loss? "
            "[USER]: Yes. "
            "[AGENT]: What's your goal weight? "
            "[USER]: 190. "
            "[AGENT]: Any requests about your dosage? "
            "[USER]: No. "
            "[AGENT]: Have you started any new medications? "
            "[USER]: No. "
            "[AGENT]: Any new medical conditions? "
            "[USER]: No. "
            "[AGENT]: Any new allergies? "
            "[USER]: No. "
            "[AGENT]: Any surgeries since last check-in? "
            "[USER]: No. "
            "[AGENT]: Any questions for your doctor? "
            "[USER]: No. "
            "[AGENT]: Has your shipping address changed? "
            "[USER]: No. "
            "[AGENT]: Great, thank you! Take care!"
        ),
        "validation_notes": "All 14 mapped questions were asked and answered; outcome remains COMPLETED_QUESTIONNAIRE.",
        "responses_json": '[{"question": "How have you been feeling overall?", "answer": "Pretty good"}, '
            '{"question": "What\'s your current weight in pounds?", "answer": "210"}, '
            '{"question": "What\'s your height in feet and inches?", "answer": "5\'10"}, '
            '{"question": "How much weight have you lost this past month in pounds?", "answer": "5"}, '
            '{"question": "Any side effects from your medication this month?", "answer": "No"}, '
            '{"question": "Satisfied with your rate of weight loss?", "answer": "Yes"}, '
            '{"question": "What\'s your goal weight in pounds?", "answer": "190"}, '
            '{"question": "Any requests about your dosage?", "answer": "No"}, '
            '{"question": "Have you started any new medications or supplements since last month?", "answer": "No"}, '
            '{"question": "Do you have any new medical conditions since your last check-in?", "answer": "No"}, '
            '{"question": "Any new allergies?", "answer": "No"}, '
            '{"question": "Any surgeries since your last check-in?", "answer": "No"}, '
            '{"question": "Any questions for your doctor?", "answer": "No"}, '
            '{"question": "Has your shipping address changed?", "answer": "No"}]',
        "whisper_transcript": (
            "Thanks for calling TrimRX. Am I speaking with John Doe? "
            "Yes, that's me. Are you interested in getting your Semaglutide refill? "
            "Yes. How have you been feeling overall? Pretty good. "
            "What's your current weight in pounds? 210. "
            "What's your height in feet and inches? 5'10. "
            "How much weight have you lost this past month? About 5 pounds. "
            "Any side effects from your medication? No. "
            "Satisfied with your rate of weight loss? Yes. "
            "What's your goal weight? 190. "
            "Any requests about your dosage? No. "
            "Have you started any new medications? No. "
            "Any new medical conditions? No. "
            "Any new allergies? No. "
            "Any surgeries since last check-in? No. "
            "Any questions for your doctor? No. "
            "Has your shipping address changed? No. "
            "Great, thank you! Take care!"
        ),
        "has_ticket": False,
    }])


@pytest.fixture
def ticket_call_skipped_questions():
    """A ticket case: completed but questions were skipped."""
    return pd.DataFrame([{
        "call_id": "test-ticket-001",
        "outcome": "completed",
        "call_duration": 60,
        "attempt_number": 1,
        "direction": "outbound",
        "whisper_status": "completed",
        "whisper_mismatch_count": 0,
        "organization_id": "org_001",
        "product_id": "prod_001",
        "patient_state": "TX",
        "cycle_status": "active",
        "form_submitted": True,
        "patient_name_anon": "Jane Smith",
        "question_count": 14,
        "answered_count": 5,
        "response_completeness": 0.36,
        "turn_count": 12,
        "user_turn_count": 5,
        "agent_turn_count": 7,
        "user_word_count": 30,
        "agent_word_count": 80,
        "avg_user_turn_words": 6.0,
        "avg_agent_turn_words": 11.4,
        "interruption_count": 0,
        "max_time_in_call": 60,
        "hour_of_day": 14,
        "day_of_week": "Tuesday",
        "transcript_text": (
            "[AGENT]: Hi, am I speaking with Jane Smith? "
            "[USER]: Yes. "
            "[AGENT]: How have you been feeling overall? "
            "[USER]: Fine. "
            "[AGENT]: What's your current weight? "
            "[USER]: 180. "
            "[AGENT]: Any side effects? "
            "[USER]: No. "
            "[AGENT]: Great, we're all done. Take care!"
        ),
        "validation_notes": "Agent marked call complete but 2+ required questions were never asked.",
        "responses_json": '[{"question": "How have you been feeling overall?", "answer": "Fine"}, '
            '{"question": "What\'s your current weight in pounds?", "answer": "180"}, '
            '{"question": "What\'s your height in feet and inches?", "answer": ""}, '
            '{"question": "How much weight have you lost this past month in pounds?", "answer": ""}, '
            '{"question": "Any side effects from your medication this month?", "answer": "No"}, '
            '{"question": "Satisfied with your rate of weight loss?", "answer": ""}, '
            '{"question": "What\'s your goal weight in pounds?", "answer": ""}, '
            '{"question": "Any requests about your dosage?", "answer": ""}, '
            '{"question": "Have you started any new medications or supplements since last month?", "answer": ""}, '
            '{"question": "Do you have any new medical conditions since your last check-in?", "answer": ""}, '
            '{"question": "Any new allergies?", "answer": ""}, '
            '{"question": "Any surgeries since your last check-in?", "answer": ""}, '
            '{"question": "Any questions for your doctor?", "answer": ""}, '
            '{"question": "Has your shipping address changed?", "answer": ""}]',
        "whisper_transcript": (
            "Hi, am I speaking with Jane Smith? Yes. "
            "How have you been feeling overall? Fine. "
            "What's your current weight? 180. "
            "Any side effects? No. "
            "Great, we're all done. Take care!"
        ),
        "has_ticket": True,
    }])


@pytest.fixture
def ticket_call_stt_error():
    """A ticket case: STT mishearing on weight value."""
    return pd.DataFrame([{
        "call_id": "test-ticket-002",
        "outcome": "completed",
        "call_duration": 200,
        "attempt_number": 1,
        "direction": "outbound",
        "whisper_status": "completed",
        "whisper_mismatch_count": 1,
        "organization_id": "org_001",
        "product_id": "prod_001",
        "patient_state": "NY",
        "cycle_status": "active",
        "form_submitted": True,
        "patient_name_anon": "Bob Wilson",
        "question_count": 14,
        "answered_count": 14,
        "response_completeness": 1.0,
        "turn_count": 32,
        "user_turn_count": 14,
        "agent_turn_count": 18,
        "user_word_count": 90,
        "agent_word_count": 170,
        "avg_user_turn_words": 6.4,
        "avg_agent_turn_words": 9.4,
        "interruption_count": 0,
        "max_time_in_call": 200,
        "hour_of_day": 9,
        "day_of_week": "Wednesday",
        "transcript_text": (
            "[AGENT]: Am I speaking with Bob Wilson? "
            "[USER]: Yes. "
            "[AGENT]: What's your current weight in pounds? "
            "[USER]: 262. "
            "[AGENT]: Got it. Take care!"
        ),
        "validation_notes": "Q2 recorded weight erroneously as 62; original response was clearly 262. | WHISPER VERIFICATION: Q2 weight differs.",
        "responses_json": '[{"question": "What\'s your current weight in pounds?", "answer": "62"}]',
        "whisper_transcript": "Am I speaking with Bob Wilson? Yes. What's your current weight in pounds? 262. Got it. Take care!",
        "has_ticket": True,
    }])

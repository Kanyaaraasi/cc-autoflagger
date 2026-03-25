"""Conversation flow analysis — detect deviations from expected call structure."""

import re

import pandas as pd

# Expected call flow: greeting → identity → medication confirmation → check-in → questions → closing
EXPECTED_FLOW = [
    "greeting",
    "identity",
    "medication",
    "checkin_consent",
    "q_feeling",
    "q_weight",
    "q_height",
    "q_weight_lost",
    "q_side_effects",
    "q_satisfaction",
    "q_goal_weight",
    "q_dosage",
    "q_new_meds",
    "q_new_conditions",
    "q_allergies",
    "q_surgeries",
    "q_doctor_questions",
    "q_address",
    "closing",
]

# Patterns to identify each state from agent utterances
STATE_PATTERNS = {
    "greeting": r"(thanks for calling|hello|hi,?\s)",
    "identity": r"am i speaking with",
    "medication": r"(interested in getting|refill)",
    "checkin_consent": r"(2 minutes|quick check-in|do you have)",
    "q_feeling": r"feeling overall",
    "q_weight": r"current weight",
    "q_height": r"height in feet",
    "q_weight_lost": r"weight have you lost",
    "q_side_effects": r"side effects",
    "q_satisfaction": r"satisfied with your rate",
    "q_goal_weight": r"goal weight",
    "q_dosage": r"(requests about your dosage|dosage)",
    "q_new_meds": r"(new medications|supplements)",
    "q_new_conditions": r"(new medical conditions|medical conditions)",
    "q_allergies": r"new allergies",
    "q_surgeries": r"surgeries",
    "q_doctor_questions": r"questions for your doctor",
    "q_address": r"shipping address",
    "closing": r"(take care|goodbye|have a great|thank you for your time)",
}


def _extract_agent_turns(transcript: str) -> list[str]:
    if pd.isna(transcript):
        return []
    return re.findall(r"\[AGENT\]:\s*(.*?)(?=\[USER\]:|$)", transcript, re.DOTALL)


def _tag_states(agent_turns: list[str]) -> list[str]:
    """Tag each agent turn with a state from the flow."""
    states = []
    for turn in agent_turns:
        turn_lower = turn.lower().strip()
        matched = None
        for state, pattern in STATE_PATTERNS.items():
            if re.search(pattern, turn_lower):
                matched = state
                break
        if matched and (not states or states[-1] != matched):
            states.append(matched)
    return states


def _edit_distance(seq1: list[str], seq2: list[str]) -> int:
    """Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def extract(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    edit_distances = []
    missing_states = []
    actual_state_counts = []
    question_states_found = []

    for _, row in df.iterrows():
        agent_turns = _extract_agent_turns(row.get("transcript_text", ""))
        actual_flow = _tag_states(agent_turns)
        actual_state_counts.append(len(actual_flow))

        # Edit distance from expected flow
        ed = _edit_distance(actual_flow, EXPECTED_FLOW)
        edit_distances.append(ed)

        # Count missing states
        expected_set = set(EXPECTED_FLOW)
        actual_set = set(actual_flow)
        missing = len(expected_set - actual_set)
        missing_states.append(missing)

        # Count question states found
        q_states = [s for s in actual_flow if s.startswith("q_")]
        question_states_found.append(len(q_states))

    features["flow_edit_distance"] = edit_distances
    features["flow_missing_states"] = missing_states
    features["flow_actual_state_count"] = actual_state_counts
    features["flow_question_states_found"] = question_states_found
    features["flow_question_coverage"] = [q / 14.0 for q in question_states_found]

    return features

"""Exploratory data analysis for Problem 1."""

import pandas as pd

from .config import TARGET
from .data_loader import load_all


def main():
    train, val, test = load_all()

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Train: {len(train)} calls, {train[TARGET].sum()} tickets ({train[TARGET].mean():.1%})")
    print(f"Val:   {len(val)} calls, {val[TARGET].sum()} tickets ({val[TARGET].mean():.1%})")
    print(f"Test:  {len(test)} calls (labels hidden)")
    print(f"Columns: {len(train.columns)}")

    print("\n" + "=" * 60)
    print("OUTCOME x TICKET CROSS-TAB (Train)")
    print("=" * 60)
    ct = pd.crosstab(train["outcome"], train[TARGET], margins=True)
    ct.columns = ["no_ticket", "ticket", "total"]
    ct["ticket_rate"] = (ct["ticket"] / ct["total"]).apply(lambda x: f"{x:.1%}")
    print(ct.to_string())

    print("\n" + "=" * 60)
    print("KEY FEATURE DISTRIBUTIONS: TICKET vs NO-TICKET (Train)")
    print("=" * 60)
    features_to_compare = [
        "call_duration", "response_completeness", "answered_count",
        "whisper_mismatch_count", "turn_count", "interruption_count",
        "user_word_count", "agent_word_count",
    ]
    ticket = train[train[TARGET]]
    no_ticket = train[~train[TARGET]]

    for feat in features_to_compare:
        if feat in train.columns:
            t_mean = ticket[feat].mean()
            nt_mean = no_ticket[feat].mean()
            t_med = ticket[feat].median()
            nt_med = no_ticket[feat].median()
            print(f"\n  {feat}:")
            print(f"    Ticket:    mean={t_mean:.2f}, median={t_med:.2f}")
            print(f"    No ticket: mean={nt_mean:.2f}, median={nt_med:.2f}")

    print("\n" + "=" * 60)
    print("WHISPER STATUS x TICKET (Train)")
    print("=" * 60)
    ws_ct = pd.crosstab(train["whisper_status"], train[TARGET], margins=True)
    ws_ct.columns = ["no_ticket", "ticket", "total"]
    ws_ct["ticket_rate"] = (ws_ct["ticket"] / ws_ct["total"]).apply(lambda x: f"{x:.1%}")
    print(ws_ct.to_string())

    print("\n" + "=" * 60)
    print("SAMPLE TICKET CASES — validation_notes")
    print("=" * 60)
    ticket_samples = train[train[TARGET]].head(10)
    for _, row in ticket_samples.iterrows():
        print(f"\n  [{row['outcome']}] {row['call_id'][:8]}...")
        print(f"    completeness={row['response_completeness']}, whisper_mismatch={row['whisper_mismatch_count']}")
        notes = str(row.get("validation_notes", ""))[:200]
        print(f"    notes: {notes}")
        initial = str(row.get("ticket_initial_notes", ""))[:200]
        if initial and initial != "nan":
            print(f"    ticket_reason: {initial}")

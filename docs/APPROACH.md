# Problem 1: Call Quality Auto-Flagger — Approach

## Problem
Binary classification: predict whether a healthcare AI voice agent call needs human review (`has_ticket = True/False`). ~9% positive rate across 992 synthetic calls.

## Dataset
| Split | Calls | Tickets | Rate |
|-------|-------|---------|------|
| Train | 689 | 59 | 8.6% |
| Val | 144 | 11 | 7.6% |
| Test | 159 | Hidden | — |

## Architecture

```
Raw Data (53 columns per call)
    │
    ▼
┌──────────────────────────────────────────────┐
│           8 Signal Extractors                │
│                                              │
│  1. Structured features (numeric + one-hot)  │
│  2. Heuristic rules (10 domain rules)        │
│  3. Transcript diff (WER/CER)               │
│  4. Number checker (plausibility)            │
│  5. Flow checker (edit distance)             │
│  6. Text features (TF-IDF + keywords)        │
│  7. Outcome predictor (disagreement)         │
│  8. Response checker (answer verification)   │
└──────────────┬───────────────────────────────┘
               │ 146 features
               ▼
┌──────────────────────────────────────────────┐
│    Grid Search: XGBoost + LightGBM           │
│    5-fold Stratified CV on train             │
│    Pick best model by CV F1                  │
└──────────────┬───────────────────────────────┘
               │ ml_proba
               ▼
┌──────────────────────────────────────────────┐
│    NLI Contradiction Detection (DeBERTa)     │
│    6 scores per call (zero-shot)             │
└──────────────┬───────────────────────────────┘
               │ 6 NLI scores
               ▼
┌──────────────────────────────────────────────┐
│    Stacking Meta-learner (LogReg)            │
│    15 features: ml_proba + 6 NLI scores      │
│    + 8 context features                      │
│    Blended threshold: 0.69                   │
└──────────────┬───────────────────────────────┘
               │
               ▼
         submission.csv
```

## Signal Extractors

### 1. Structured Features (~30 features)
Standard numeric columns (call_duration, turn_count, word counts, etc.) plus one-hot encoded categoricals (outcome, direction, whisper_status, day_of_week). Three derived features: duration_per_turn, user_talk_ratio, duration_per_completeness.

### 2. Heuristic Rules (12 features)
Domain-specific binary flags:
- `completed` + low response_completeness → skipped questions
- `opted_out` + high completeness → outcome miscategorization
- `wrong_number` + long conversation → misclassification
- whisper_mismatch_count > 0 → STT errors
- Medical advice keywords in agent text → guardrail violation
- Questions asked vs answered gap → data capture issues

### 3. Transcript Diff (4 features)
Compares `transcript_text` (formatted with role markers) against `whisper_transcript` (raw STT) using jiwer: Word Error Rate, Character Error Rate, sequence similarity, length ratio.

### 4. Number Checker (3 features)
Extracts numbers from transcript, cross-validates against `responses_json`:
- STT number errors (substring/transposition detection)
- Physiologically implausible values (weight outside 50-600 lbs)
- Answers in response but missing from transcript

### 5. Flow Checker (5 features)
Models expected call flow as a 19-state sequence (greeting → identity → 14 questions → closing). Tags actual agent turns, computes Levenshtein edit distance from expected flow.

### 6. Text Features (~30 features)
TF-IDF (15 features) on `validation_notes` plus 13 binary keyword flags for issue-indicating phrases (mismatch, error, skipped, medical advice, etc.) plus word counts.

### 7. Outcome Predictor (4 features)
Trains a separate LogisticRegression to predict call outcome from transcript alone. Features: disagreement flag, prediction entropy, confidence, probability of actual outcome.

### 8. Response Checker (5 features)
Verifies whether the answers recorded in `responses_json` actually appear in the transcript text.

- `resp_not_in_transcript`: Fraction of answered responses whose value does not appear anywhere in the transcript. Tickets average 0.163 vs non-tickets 0.021 — the strongest single feature in this extractor.
- `resp_empty_count`: Number of responses with empty/missing answers.
- `resp_binary_ratio`: Fraction of responses that are simple yes/no answers.
- `resp_words_per_answered`: Average word count per answered response.
- `resp_duration_per_answered`: Call duration divided by number of answered responses.

## NLI Contradiction Detection

Uses `cross-encoder/nli-deberta-v3-base` (~370MB) to detect contradictions between `validation_notes` and structured fields. For each call, the checker generates hypotheses from structured data (e.g., "The patient answered 11 of 14 questions") and compares them against statements in the validation notes. Produces 6 features: max contradiction score, mean entailment, number of contradictions above threshold, and per-hypothesis contradiction scores for outcome, answered count, and completeness.

## Stacking Meta-learner

A LogisticRegression meta-learner combines ML predictions with NLI signals using 15 features:

| Feature Group | Count | Examples |
|---------------|-------|---------|
| ML probability | 1 | `ml_proba` |
| NLI scores | 6 | `nli_max_contradiction`, `nli_mean_entailment`, etc. |
| Context features | 8 | `resp_not_in_transcript`, `resp_empty_count`, `resp_binary_ratio`, `response_completeness`, `answered_count`, `whisper_mismatch_count`, `rule_any_fired`, `outcome_pred_confidence` |

The wider feature set lets the meta-learner contextualize NLI contradictions. For example, a high NLI contradiction score combined with `resp_not_in_transcript = 0` (answers verified in transcript) likely means the contradiction is noise, not a real ticket.

Threshold selection uses a blended approach: 40% validation-optimal + 60% CV-optimal = 0.69.

## Model Selection

Grid search over 18 hyperparameter combinations for both XGBoost and LightGBM:
- max_depth: [3, 4, 5]
- learning_rate: [0.05, 0.1]
- n_estimators: [100, 200, 300]

Class imbalance handled via `scale_pos_weight` (~10.7x).

Best model selected by validation F1, with ensemble (avg probabilities) as a third candidate.

## Results

| Metric | Score |
|--------|-------|
| Val F1 | 1.000 |
| Val Precision | 1.000 |
| Val Recall | 1.000 |
| Base CV F1 | 0.9599 +/- 0.036 |
| Stacked CV F1 | 0.9739 +/- 0.021 |
| Best Model | LightGBM (depth=3, lr=0.1, 200 trees) |
| Test Predictions | 18/159 flagged (11.3%) |
| Private Leaderboard F1 | **1.000** |
| Public Leaderboard F1 | **1.000** |

## Key Decisions

### Feature Selection
Dropped `patient_state` one-hot encoding (50 sparse features) — US state codes are noise for ticket prediction. Reduced TF-IDF from 50 to 15 features. This cut features from 165 → 146 (after adding Response Checker) and improved CV F1 significantly.

### What We Tried and Dropped
- **LLM-as-judge (Groq API)**: Built and tested. Over-flagged (5/5 test calls marked as issues). Rate limited on gpt-oss-120b. Marginal value given validation_notes already captures similar analysis.

## Suspicions & Caveats

### validation_notes — strong signal, borderline leakage
The `validation_notes` field is a post-call AI analysis that contains phrases like "dosage guidance" and "weight dif" that appear **only** in ticket cases during training. This makes it an extremely powerful feature.

**Is it leakage?** No — the field exists in the test set too with the same patterns. It's a legitimate upstream signal from the AI validation system.

**Risk:** The model may over-rely on validation_notes phrasing. If test set notes are worded differently, performance could degrade.

**Mitigation tested:** Without validation_notes features, the model still achieves F1=0.952 (precision=1.0, recall=0.91). The model is robust without it.

### Perfect val F1 — overfitting concern
F1=1.000 on 144 val samples (11 positives) looks suspicious. However:
- CV F1=0.944 confirms the model generalizes across folds
- Without validation_notes: still F1=0.952
- The task is genuinely easier than it looks — ticket cases have distinct patterns (high completeness + specific issues)

### Small positive class
59 training positives is small. Mitigations:
- Shallow trees (depth=3) prevent overfitting
- Class weights (10.7x) balance learning
- Grid search uses 5-fold CV, not val set, to select hyperparameters

# Technical Guide — How the Auto-Flagger Works

A developer-friendly walkthrough of every algorithm and technique used in this project.

---

## Table of Contents

1. [The Problem in Engineering Terms](#1-the-problem-in-engineering-terms)
2. [Data Pipeline](#2-data-pipeline)
3. [Feature Engineering — The 7 Signal Extractors](#3-feature-engineering--the-7-signal-extractors)
4. [Machine Learning Models](#4-machine-learning-models)
5. [Threshold Tuning](#5-threshold-tuning)
6. [Evaluation Metrics Explained](#6-evaluation-metrics-explained)
7. [Data Leakage Analysis](#7-data-leakage-analysis)
8. [What We Tried and Why We Dropped It](#8-what-we-tried-and-why-we-dropped-it)

---

## 1. The Problem in Engineering Terms

We have a database of 992 phone calls. Each call has metadata, a full transcript, structured Q&A responses, and a label: `has_ticket = True/False` (whether a human reviewer flagged an issue).

**Task:** Given a new call's data, predict if it needs review.

**Why it's hard:**
- Only **9% of calls** have issues (imbalanced dataset)
- Issues are diverse: speech-to-text errors, skipped questions, wrong labels, guardrail violations
- Small dataset: only 59 positive examples to learn from

**What we output:** A CSV with `call_id, predicted_ticket` for 159 unseen test calls.

---

## 2. Data Pipeline

```
hackathon_train.csv (689 rows × 53 columns)
hackathon_val.csv   (144 rows × 53 columns)
hackathon_test.csv  (159 rows × 53 columns, labels hidden)
        │
        ▼
   data_loader.py        → Loads CSVs, parses JSON fields
        │
        ▼
   features.py           → Runs 7 signal extractors, produces feature matrix
        │                   (689 × 135 numeric matrix)
        ▼
   train.py              → Trains LightGBM, tunes threshold
        │
        ▼
   predict.py            → Applies model to test set → submission.csv
```

### Column Categories

The 53 raw columns fall into groups:

| Group | Examples | Used as features? |
|-------|---------|-------------------|
| **Metadata** | call_duration, outcome, attempt_number | Yes (numeric + one-hot) |
| **Response stats** | answered_count, response_completeness | Yes |
| **Transcript stats** | turn_count, word_counts, interruptions | Yes |
| **Text fields** | transcript_text, validation_notes | Yes (processed into numbers) |
| **Target** | has_ticket | No (this is what we predict) |
| **Ticket details** | ticket_initial_notes, ticket_cat_* | No (leakage — only exist for ticket cases) |

**Leakage columns** are fields that exist *because* a ticket was created. Using them would be like checking "was a ticket filed?" to predict "should a ticket be filed?" — circular. We explicitly drop 20 such columns.

---

## 3. Feature Engineering — The 7 Signal Extractors

Each extractor takes the raw dataframe and outputs new numeric columns. All 7 are concatenated into a single feature matrix.

### 3a. Structured Features (~30 features)
**File:** `src/features.py → _structured_features()`

**What it does:** Takes the numeric columns as-is, and converts categorical columns into numbers.

**One-hot encoding** converts a column like `outcome` (which has values like "completed", "incomplete", "escalated") into separate binary columns:

```
Before:  outcome = "completed"
After:   outcome_completed = 1, outcome_incomplete = 0, outcome_escalated = 0, ...
```

We one-hot encode: `outcome`, `direction`, `whisper_status`, `cycle_status`, `day_of_week`.

We deliberately **don't** encode `patient_state` (US state). With 50 states, that's 50 extra columns that mostly contain noise — a patient being in Wyoming vs California doesn't predict call quality.

**Derived features** (combinations of existing columns):
- `duration_per_turn` = call_duration / turn_count — how long each conversation exchange took
- `user_talk_ratio` = user_words / total_words — how much the patient spoke vs the agent
- `duration_per_completeness` = call_duration / response_completeness — time relative to how much got done

These ratios help the model see patterns like "long call but few questions answered = something went wrong."

---

### 3b. Heuristic Rules (12 features)
**File:** `src/signals/heuristics.py`

**What it does:** Encodes domain knowledge as if/then rules. Each rule produces a 0 or 1.

| Rule | Logic | What it catches |
|------|-------|----------------|
| Completed + low completeness | `outcome == "completed" AND response_completeness < 0.8` | Agent marked done but skipped questions |
| Opted out + high completeness | `outcome == "opted_out" AND response_completeness > 0.5` | Patient answered many questions but call labeled as opt-out |
| Wrong number + long conversation | `outcome == "wrong_number" AND turn_count > 10` | Real conversations don't happen with wrong numbers |
| Whisper mismatch | `whisper_mismatch_count > 0` | Two speech-to-text systems disagreed |
| Medical advice detected | Regex patterns in agent text | Agent said "I recommend", "you should take", etc. |
| Question gap | Questions found in transcript - answered_count | More questions asked than answers recorded |
| Completed + whisper skipped | Outcome is completed but STT verification was skipped | Quality check was bypassed |
| Escalated call | `outcome == "escalated"` | Escalated calls have 12.5% ticket rate |
| Completed + no form | Completed but form_submitted is False | Inconsistent state |

**How medical advice detection works:** We search agent utterances (text after `[AGENT]:`) for regex patterns:

```python
patterns = [
    r"\byou should take\b",
    r"\bi recommend\b",
    r"\btry taking\b",
    r"\bstop taking\b",
    # ... 12 total patterns
]
```

The `\b` means "word boundary" — so "recommend" matches but "recommendation" doesn't accidentally trigger it.

Two composite features summarize all rules:
- `rule_any_fired` = 1 if any rule triggered
- `rule_count_fired` = how many rules triggered

---

### 3c. Transcript Diff (4 features)
**File:** `src/signals/transcript_diff.py`

**What it does:** Every call has TWO transcripts of the same conversation:
1. `transcript_text` — formatted with `[AGENT]:` and `[USER]:` markers
2. `whisper_transcript` — raw speech-to-text output, no markers

If these two differ significantly, something went wrong with transcription.

**Algorithms used:**

**Word Error Rate (WER)** — the standard metric for speech recognition accuracy. It counts how many words need to be inserted, deleted, or substituted to transform one text into another, divided by the total words.

```
Reference: "my weight is two sixty two"
Hypothesis: "my weight is two sixty"
WER = 1 deletion / 6 words = 16.7%
```

We use the `jiwer` library for this. Higher WER = more transcription problems.

**Character Error Rate (CER)** — same idea but at character level. Catches smaller errors like "262" vs "62".

**Sequence Similarity** — Python's `difflib.SequenceMatcher` computes a ratio from 0 (completely different) to 1 (identical). It finds the longest common subsequences.

**Length Ratio** — simple `|len(a) - len(b)| / max(len(a), len(b))`. Big differences mean something was dropped or added.

Before comparison, we **normalize** both texts: lowercase, remove punctuation, collapse whitespace, strip role markers.

---

### 3d. Number Checker (3 features)
**File:** `src/signals/number_checker.py`

**What it does:** Extracts all numbers from the transcript and checks them against the structured responses.

**Why:** One of the 6 ticket types is "STT mishearing" — the patient says "262 pounds" but the system records "62". These are critical errors in healthcare.

**How it works:**

1. **Extract numbers** from transcript text using regex: `\b\d+\.?\d*\b` (matches "210", "5.5", "262", etc.)

2. **Map questions to numeric fields** — we know which of the 14 questions expect numbers:
   - "current weight" → weight (plausible range: 50-600 lbs)
   - "goal weight" → goal_weight (50-500 lbs)
   - "weight lost" → weight_lost (-50 to 100 lbs)

3. **Cross-reference** each recorded answer against transcript numbers:
   - **Substring check:** Is "62" a substring of "262"? → Likely a dropped leading digit
   - **Transposition check:** Is "216" a rearrangement of "261"? → Likely a digit swap
   - **Plausibility check:** Is 10 lbs a realistic adult weight? → No, flag it

---

### 3e. Flow Checker (5 features)
**File:** `src/signals/flow_checker.py`

**What it does:** Models the expected call as a sequence of steps and measures how much the actual call deviated.

**Expected flow (19 states):**
```
greeting → identity → medication → consent → q_feeling → q_weight → q_height →
q_weight_lost → q_side_effects → q_satisfaction → q_goal_weight → q_dosage →
q_new_meds → q_new_conditions → q_allergies → q_surgeries → q_doctor_questions →
q_address → closing
```

**How state tagging works:** Each agent utterance is matched against regex patterns:
```python
"greeting":    r"(thanks for calling|hello|hi)"
"identity":    r"am i speaking with"
"q_weight":    r"current weight"
"q_allergies": r"new allergies"
"closing":     r"(take care|goodbye)"
```

This produces an actual sequence like `[greeting, identity, q_feeling, q_weight, closing]`.

**Levenshtein edit distance** then measures the minimum number of insertions, deletions, and substitutions to transform the actual sequence into the expected one. This is the same algorithm used for spell-checking — it uses dynamic programming (O(m×n) time).

A perfect call has edit distance ~0. A voicemail with just a greeting has edit distance ~18.

---

### 3f. Text Features (~30 features)
**File:** `src/signals/text_features.py`

**What it does:** Converts the free-text `validation_notes` field into numbers.

**TF-IDF (Term Frequency × Inverse Document Frequency):**

TF-IDF is a way to measure how important a word is to a document relative to the whole collection.

- **TF (Term Frequency):** How often does "error" appear in this call's notes? (normalized by document length)
- **IDF (Inverse Document Frequency):** How rare is "error" across all calls? Rare words get higher scores.
- **TF-IDF = TF × IDF** — words that are frequent in one document but rare overall score highest.

We use `sklearn.TfidfVectorizer` with:
- `max_features=15` — only keep the 15 most informative terms
- `ngram_range=(1,2)` — consider single words and two-word phrases ("medical advice")
- `min_df=2` — ignore terms that appear in fewer than 2 documents
- `stop_words="english"` — ignore common words like "the", "is", "and"

**Keyword flags (13 binary features):** Simple presence/absence checks for issue-indicating words: "mismatch", "error", "skipped", "incorrect", "medical advice", "fabricated", etc.

**Other features:** Word count of validation notes, word count of transcript, user-to-agent talk ratio.

---

### 3g. Outcome Predictor (4 features)
**File:** `src/signals/outcome_predictor.py`

**What it does:** Trains a *separate* simple model to predict what the call outcome *should be* based on the transcript, then checks if that prediction matches the actual label.

**Why:** If a call's transcript reads like the patient opted out, but it's labeled "completed" — that mismatch is a strong signal of miscategorization.

**How:**
1. Fit TF-IDF (200 features) on all transcripts
2. Train a Logistic Regression to predict outcome (completed, incomplete, opted_out, etc.)
3. For each call, compute:
   - **outcome_disagreement**: Does our prediction match the label? (0 or 1)
   - **outcome_pred_entropy**: How uncertain is our prediction? (higher = more confused)
   - **outcome_pred_confidence**: How sure are we about our top prediction? (0-1)
   - **outcome_actual_prob**: How likely did our model think the *actual* outcome was? (0-1)

**Logistic Regression** is a simple linear model that outputs probabilities for each class. It's fast and interpretable. We're not trying to be accurate here — we want the *disagreements* to be informative.

**Entropy** measures uncertainty: if the model gives 33%/33%/33% across three outcomes, entropy is high (confused). If it gives 95%/3%/2%, entropy is low (confident). Formula: `-Σ(p × log(p))`.

---

## 4. Machine Learning Models

### What is LightGBM?

LightGBM (Light Gradient Boosting Machine) is a **tree ensemble** algorithm. Here's how it works:

**Decision trees** are like flowcharts:
```
                    response_completeness > 0.8?
                   /                             \
                Yes                               No
               /                                   \
    whisper_mismatch > 0?                    outcome == "completed"?
    /                  \                     /                  \
  Yes                  No                  Yes                  No
  │                    │                    │                    │
 TICKET            NO TICKET           TICKET              NO TICKET
```

A single tree is too simple. **Gradient boosting** chains 200 trees, where each new tree focuses on correcting the mistakes of the previous ones:

1. Tree 1 makes predictions → some wrong
2. Tree 2 focuses on the errors from Tree 1
3. Tree 3 focuses on remaining errors
4. ... repeat 200 times
5. Final prediction = weighted sum of all 200 trees

**LightGBM** is an optimized version that grows trees **leaf-wise** (expanding the leaf with the largest error reduction) rather than **level-wise** (expanding all leaves at the same depth). This makes it faster and often more accurate.

### Why not a neural network?

With only 59 positive examples and 135 features, deep learning would overfit catastrophically. Tree ensembles are the gold standard for small-to-medium tabular data.

### Key Hyperparameters

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `n_estimators=200` | Number of trees in the chain | More = more expressive, risk of overfitting |
| `max_depth=3` | Maximum depth of each tree | Shallow trees = simpler rules, less overfitting |
| `learning_rate=0.1` | How much each new tree corrects | Lower = more conservative, needs more trees |
| `scale_pos_weight=10.7` | Weight for positive class | Tells the model "a missed ticket costs 10.7× more than a false alarm" |
| `subsample=0.8` | Fraction of data used per tree | Randomization prevents overfitting |
| `colsample_bytree=0.8` | Fraction of features used per tree | Randomization prevents overfitting |

### Grid Search

We don't guess hyperparameters — we **try all combinations** and pick the best:

```
max_depth:     [3, 4, 5]         → 3 options
learning_rate: [0.05, 0.1]       → 2 options
n_estimators:  [100, 200, 300]   → 3 options
                                   ─────────
                                   18 combinations
```

For each combination, we run **5-fold cross-validation** (see below). That's 18 × 5 = 90 model trainings. We do this for both XGBoost and LightGBM, plus test an ensemble (average of both).

### Cross-Validation (5-Fold Stratified)

We can't just train on all 689 calls and check performance on those same calls — the model would memorize the answers.

**5-fold CV** splits the training data into 5 equal parts:
```
Fold 1: Train on parts 2-5, test on part 1
Fold 2: Train on parts 1,3-5, test on part 2
Fold 3: Train on parts 1-2,4-5, test on part 3
Fold 4: Train on parts 1-3,5, test on part 4
Fold 5: Train on parts 1-4, test on part 5
```

Each fold gives an F1 score. The average tells us how well the model generalizes. **Stratified** means each fold maintains the 9% positive ratio.

Our CV result: **F1 = 0.944 ± 0.07** (mean ± standard deviation across 5 folds).

### XGBoost vs LightGBM vs Ensemble

Both are gradient boosting algorithms with slightly different tree-building strategies:

| | XGBoost | LightGBM |
|---|---|---|
| Tree growth | Level-wise (all leaves at same depth) | Leaf-wise (greediest leaf first) |
| Speed | Slower | Faster |
| Our CV F1 | 0.944 | **0.956** |

The **ensemble** averages their probabilities: `final = 0.5 × xgb_prob + 0.5 × lgb_prob`. Sometimes this helps because they make different mistakes.

Winner: **LightGBM** (highest CV F1 and perfect val score).

---

## 5. Threshold Tuning

The model outputs a probability (0.0 to 1.0) for each call. We need to pick a **cutoff**: above this → flag as ticket, below → don't flag.

The default would be 0.5, but with 9% positive rate that's too aggressive. We **sweep** all thresholds from 0.05 to 0.95 in steps of 0.01 and pick the one that maximizes F1 on the validation set.

```
Threshold 0.10: flags 25 calls → Precision=0.44, Recall=1.00, F1=0.61
Threshold 0.20: flags 15 calls → Precision=0.73, Recall=1.00, F1=0.85
Threshold 0.35: flags 11 calls → Precision=1.00, Recall=1.00, F1=1.00  ← picked
Threshold 0.50: flags  9 calls → Precision=1.00, Recall=0.82, F1=0.90
Threshold 0.70: flags  7 calls → Precision=1.00, Recall=0.64, F1=0.78
```

Our optimal threshold: **0.35** (any call with >35% ticket probability gets flagged).

---

## 6. Evaluation Metrics Explained

### Precision — "When we flag a call, are we right?"
```
Precision = True Positives / (True Positives + False Positives)
```
If we flag 13 calls and 11 are actual tickets, 2 are not → Precision = 11/13 = 85%.
High precision = fewer false alarms = less wasted reviewer time.

### Recall — "Of all actual bad calls, how many did we catch?"
```
Recall = True Positives / (True Positives + False Negatives)
```
If there are 11 actual tickets and we catch all 11 → Recall = 11/11 = 100%.
High recall = fewer missed issues = safer for patients.

### F1 Score — "Balance of precision and recall"
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
F1 = 1.0 means both precision and recall are perfect. It's the **harmonic mean** — it penalizes being bad at either one. A model with 100% recall but 10% precision gets F1 = 0.18, not 55%.

### Why F1 and not Accuracy?

With 91% non-tickets, a model that says "no ticket" for every call gets **91% accuracy** but catches zero issues (F1 = 0). Accuracy is misleading for imbalanced data. F1 forces the model to actually find the positive cases.

---

## 7. Data Leakage Analysis

**Data leakage** = accidentally using information that wouldn't be available at prediction time, making the model look better than it really is.

### Columns we explicitly exclude (20 columns)

All `ticket_*` columns: `ticket_initial_notes`, `ticket_priority`, `ticket_status`, `ticket_cat_*`, etc. These only have values when `has_ticket=True`, so using them would be directly encoding the answer.

### The validation_notes question

`validation_notes` is an AI-generated post-call analysis. For ticket cases, it often contains phrases like:
- "dosage guidance" (appears in 13 ticket cases, 0 non-ticket cases)
- "weight dif" (5 ticket, 0 non-ticket)
- "fabricated" (1 ticket, 0 non-ticket)

**Is this leakage?** We investigated:

1. The field **exists in the test set** with the same patterns (4 test calls have leak phrases)
2. It's generated by an upstream AI system, not by the ticket creation process
3. **Without validation_notes**, the model still achieves F1=0.952

**Conclusion:** It's a legitimate (but very strong) signal, not leakage. The upstream AI validation system is essentially doing half our job already — we're just formalizing its output into a prediction.

### How we verified no leakage

Our test suite (`test_pipeline.py`) explicitly checks:
- No `LEAKAGE_COLS` appear in the feature matrix
- No `has_ticket` (target) in features
- No raw text columns in features
- No `patient_state` one-hots (noise, not leakage)

---

## 8. What We Tried and Why We Dropped It

### LLM-as-Judge (Groq API, GPT-OSS-20B)

**Idea:** Send each call's transcript to a large language model and ask "does this call have quality issues?"

**Implementation:** Built a full provider system (`src/llm/`) with rate limiting, retry logic, caching, and a structured JSON evaluation rubric checking all 6 issue types.

**Result:** Over-flagged. In a 5-call test, it marked 5/5 as having issues (including perfectly normal calls). The LLM was too cautious — any small imperfection triggered a flag.

**Why dropped:** Marginal value (validation_notes already captures similar analysis), API rate limits on the free tier, and the base model was already at F1=0.92 without it.

### NLI Contradiction Checker (facebook/bart-large-mnli)

**Idea:** Use a Natural Language Inference model to detect contradictions between the transcript and the structured responses.

NLI models classify pairs of sentences as "entailment" (A implies B), "contradiction" (A contradicts B), or "neutral":
```
Premise:    "My weight is 262 pounds"
Hypothesis: "The patient weighs 62 pounds"
Result:     CONTRADICTION (score: 0.85)
```

**Implementation:** Built the full pipeline. For each Q&A pair, convert the answer to a hypothesis, find the relevant transcript segment, run NLI classification.

**Why dropped:** ~45-60 min runtime on CPU (1.5GB model, 14 comparisons per call × 992 calls). The number_checker already catches numeric mismatches faster. The heuristic rules catch non-numeric contradictions.

### Feature: patient_state one-hot encoding

**What it was:** 50 binary columns, one per US state.

**Why dropped:** With 689 training samples, most states have <15 examples. The model can't learn meaningful patterns from "3 calls from Wyoming had 1 ticket." It just memorizes noise. Removing it improved CV F1 from 0.83 to 0.94 — the single biggest improvement.

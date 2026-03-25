# CareCaller Ticket — Call Quality Auto-Flagger

> Automatically detect which AI voice agent calls need human review.

## What This Does

CareCaller is a system where **AI agents call patients** for medication refill check-ins. They ask 14 health questions (weight, side effects, allergies, etc.) and record the answers.

Sometimes things go wrong:
- The AI **mishears** a number (patient says "262 lbs", system records "62")
- The AI **skips questions** but marks the call as "completed"
- The AI **gives medical advice** when it shouldn't ("you should take ibuprofen")
- The call gets **mislabeled** (patient opted out but call says "completed")

About **9% of calls** have these issues and need human review. This tool **automatically flags** those calls so humans don't have to listen to all 992 calls.

## How It Works (Plain English)

### Step 1: Look at each call from 7 different angles

Think of it like 7 different inspectors, each checking for something specific:

| Inspector | What it checks | Example |
|-----------|---------------|---------|
| **Rules** | Common-sense red flags | "Call marked complete but only 5/14 questions answered" |
| **Transcript Comparison** | Do the two transcripts match? | Two different speech-to-text systems disagreed |
| **Number Checker** | Are the health numbers realistic? | Weight recorded as 10 lbs (impossible for an adult) |
| **Flow Checker** | Did the call follow the right steps? | Skipped from greeting straight to goodbye |
| **Text Analysis** | What do the validation notes say? | Notes mention "error" or "mismatch" |
| **Outcome Predictor** | Does the label match the conversation? | Transcript sounds like opt-out, but labeled "completed" |
| **Basic Stats** | Call duration, word counts, etc. | Unusually short call marked as completed |

Each inspector produces a set of numbers (features) describing what it found.

### Step 2: Feed all findings into a machine learning model

All 135 numbers from the 7 inspectors go into a **LightGBM model** (a type of decision tree ensemble). The model learned from 689 example calls where we know the right answer.

### Step 3: The model decides: flag or not?

The model outputs a probability (0-100%) that the call needs review. If it's above 35%, we flag it.

## Results

On 144 validation calls (where we know the answer):
- **Found all 11 bad calls** (100% recall)
- **Zero false alarms** (100% precision)
- **F1 score: 1.000** (perfect on validation set)

On the 159 test calls: flagged **17 calls** (10.7%) for review.

## Quick Start

```bash
# Install dependencies
uv sync

# Run everything (extract features → train model → generate predictions)
uv run pipeline

# Output: outputs/submission.csv
```

That's it. Takes about 30 seconds.

### All Commands

```bash
uv run pipeline     # Full pipeline: extract → train → predict
uv run eda          # Explore the data (stats, distributions)
uv run extract      # Extract features only (saves to outputs/)
uv run train        # Train model only (saves to models/)
uv run predict      # Generate submission.csv from trained model
```

### Run Tests

```bash
uv run pytest tests/ -v -s      # 79 tests (unit + edge cases + end-to-end)
```

## Project Structure

```
carecaller-ticket/
├── main.py                         # Standalone pipeline script
├── pyproject.toml                  # Dependencies + CLI scripts
├── roadmap.md                      # Architecture diagram + project status
│
├── src/
│   ├── cli.py                      # CLI entry points (uv run <cmd>)
│   ├── config.py                   # File paths, column lists
│   ├── data_loader.py              # Load train/val/test CSVs
│   ├── features.py                 # Combine all inspectors into one feature table
│   ├── train.py                    # Train models, tune threshold, evaluate
│   ├── predict.py                  # Generate submission.csv
│   ├── logger.py                   # Logging setup (console + file)
│   ├── eda.py                      # Exploratory data analysis
│   │
│   └── signals/                    # The 7 "inspectors"
│       ├── heuristics.py           # Rule-based flags (10 rules)
│       ├── transcript_diff.py      # Compare two transcripts (WER)
│       ├── number_checker.py       # Validate health numbers
│       ├── flow_checker.py         # Check conversation structure
│       ├── text_features.py        # Analyze validation notes
│       └── outcome_predictor.py    # Predict outcome independently
│
├── tests/                          # 79 tests
│   ├── conftest.py                 # Test fixtures (sample calls)
│   ├── test_data_loader.py         # Data loading tests
│   ├── test_signals.py             # Signal extractor tests
│   ├── test_pipeline.py            # Pipeline + threshold tests
│   ├── test_edge_cases.py          # Adversarial + boundary inputs
│   └── test_end_to_end.py          # Full pipeline + F1 validation
│
├── docs/
│   ├── APPROACH.md                 # Methodology + decisions + caveats
│   └── TECHNICAL.md                # Developer guide (algorithms explained)
│
├── outputs/                        # Generated (gitignored)
│   ├── submission.csv              # Final predictions
│   ├── X_train.parquet             # Feature matrices
│   └── pipeline.log                # Run log
│
└── models/                         # Generated (gitignored)
    ├── model.pkl                   # Trained LightGBM
    └── config.json                 # Threshold + column order + hyperparams
```

## What's in the Data?

Each call has:
- **Metadata**: outcome, duration, attempt number
- **Transcript**: full `[AGENT]: ... [USER]: ...` conversation
- **Whisper transcript**: raw speech-to-text (no speaker labels)
- **Responses**: 14 Q&A pairs (the health questionnaire answers)
- **Validation notes**: an AI's post-call analysis of the call

The model uses all of these except the ticket-related fields (that would be cheating).

## Important Notes

### On "validation_notes" (our strongest signal)

The `validation_notes` field is written by an upstream AI system *after* the call. For problem calls, it says things like "Agent provided dosage guidance" or "recorded weight erroneously as 62." This is extremely helpful but raises a question: **is this cheating?**

**No** — the field exists in the test set too with the same patterns. It's a legitimate part of the pipeline. But we verified the model works well even without it (F1=0.95 vs 1.00).

### On the perfect validation score

A perfect F1 score on validation sounds too good to be true. We checked:
- Cross-validation across 5 different train splits: F1 = 0.94 (still strong, not perfect)
- Without validation_notes: F1 = 0.95
- The issues are genuinely distinctive — bad calls have clear patterns

### What we tried and removed

- **LLM-as-judge**: Used GPT-OSS via Groq to evaluate each call. Over-flagged everything. Removed.
- **NLI contradiction checker**: Would compare transcript against recorded answers using a language model. Too slow (45+ min). Removed.
- Both added complexity without improving the score.

## Submission Format

```csv
call_id,predicted_ticket
4a90d9a9-...,False
c138772b-...,True
...
```

159 rows, one per test call.

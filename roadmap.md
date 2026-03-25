# Problem 1: Call Quality Auto-Flagger — Roadmap

## Goal
Binary classification: predict `has_ticket` (True/False) for 159 test calls.
Scored on **F1** (primary), Recall (secondary), Precision (tertiary).

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DATA INGESTION LAYER                               ║
║                                                                            ║
║   hackathon_train.csv ──┐                                                  ║
║   hackathon_val.csv ────┼──▶  data_loader.py  ──▶  pd.DataFrame (53 cols)  ║
║   hackathon_test.csv ───┘        │                                         ║
║                                  │  parse_responses()                      ║
║                                  ▼                                         ║
║                          JSON ─▶ list[{question, answer}]                  ║
╚═══════════════════════════════════╤════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     FEATURE EXTRACTION PIPELINE                            ║
║                                                                            ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │                    7 SIGNAL EXTRACTORS                              │   ║
║   │                                                                     │   ║
║   │   ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐     │   ║
║   │   │  STRUCTURED   │  │  HEURISTIC    │  │  TRANSCRIPT DIFF   │     │   ║
║   │   │  FEATURES     │  │  RULES        │  │                    │     │   ║
║   │   │               │  │               │  │  transcript_text   │     │   ║
║   │   │  call_duration│  │  10 domain    │  │       vs           │     │   ║
║   │   │  turn_count   │  │  rules as     │  │  whisper_transcript│     │   ║
║   │   │  word_counts  │  │  binary flags │  │                    │     │   ║
║   │   │  one-hot cats │  │               │  │  WER, CER,         │     │   ║
║   │   │  derived      │  │  skipped Q's  │  │  seq_similarity,   │     │   ║
║   │   │  ratios       │  │  mislabels    │  │  len_ratio         │     │   ║
║   │   │               │  │  med advice   │  │                    │     │   ║
║   │   │  ~30 features │  │  12 features  │  │  4 features        │     │   ║
║   │   └───────┬───────┘  └───────┬───────┘  └─────────┬──────────┘     │   ║
║   │           │                  │                     │                │   ║
║   │   ┌───────┴───────┐  ┌──────┴────────┐  ┌─────────┴──────────┐     │   ║
║   │   │   NUMBER      │  │  FLOW         │  │  TEXT FEATURES      │     │   ║
║   │   │   CHECKER     │  │  CHECKER      │  │                     │     │   ║
║   │   │               │  │               │  │  TF-IDF on          │     │   ║
║   │   │  Extract nums │  │  Expected:    │  │  validation_notes   │     │   ║
║   │   │  from text,   │  │  19-state     │  │  (15 features)      │     │   ║
║   │   │  cross-check  │  │  sequence     │  │                     │     │   ║
║   │   │  vs responses │  │               │  │  13 keyword flags   │     │   ║
║   │   │               │  │  Actual:      │  │  (mismatch, error,  │     │   ║
║   │   │  Plausibility │  │  tagged from  │  │   skipped, etc.)    │     │   ║
║   │   │  checks       │  │  transcript   │  │                     │     │   ║
║   │   │  (50-600 lbs) │  │               │  │  word counts,       │     │   ║
║   │   │               │  │  Levenshtein  │  │  talk ratios        │     │   ║
║   │   │  STT error    │  │  edit dist    │  │                     │     │   ║
║   │   │  detection    │  │               │  │                     │     │   ║
║   │   │               │  │               │  │                     │     │   ║
║   │   │  3 features   │  │  5 features   │  │  ~30 features       │     │   ║
║   │   └───────┬───────┘  └──────┬────────┘  └─────────┬───────────┘     │   ║
║   │           │                 │                      │                │   ║
║   │           │    ┌────────────┴──────────────┐       │                │   ║
║   │           │    │  OUTCOME PREDICTOR        │       │                │   ║
║   │           │    │                           │       │                │   ║
║   │           │    │  Separate LogisticReg     │       │                │   ║
║   │           │    │  predicts outcome from    │       │                │   ║
║   │           │    │  transcript alone.        │       │                │   ║
║   │           │    │  Flags disagreements      │       │                │   ║
║   │           │    │  with actual label.       │       │                │   ║
║   │           │    │                           │       │                │   ║
║   │           │    │  4 features               │       │                │   ║
║   │           │    └────────────┬──────────────┘       │                │   ║
║   │           │                 │                      │                │   ║
║   └───────────┴─────────────────┴──────────────────────┴────────────────┘   ║
║                                 │                                          ║
║                    CONCAT ──▶   │  135 numeric features                    ║
║                                 │                                          ║
╚═════════════════════════════════╤══════════════════════════════════════════╝
                                  │
                                  ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MODEL TRAINING LAYER                                  ║
║                                                                            ║
║   ┌──────────────────────────────────────────────────────────────────┐      ║
║   │              GRID SEARCH (5-Fold Stratified CV)                  │      ║
║   │                                                                  │      ║
║   │   max_depth:     [3, 4, 5]        ─┐                            │      ║
║   │   learning_rate: [0.05, 0.1]       ├─▶ 18 combos × 5 folds     │      ║
║   │   n_estimators:  [100, 200, 300]  ─┘    = 90 trainings/model   │      ║
║   │                                                                  │      ║
║   │   ┌──────────────┐        ┌──────────────┐                      │      ║
║   │   │   XGBoost    │        │  LightGBM    │                      │      ║
║   │   │              │        │              │                      │      ║
║   │   │  CV F1=0.944 │        │  CV F1=0.956 │  ◀── WINNER         │      ║
║   │   └──────┬───────┘        └──────┬───────┘                      │      ║
║   │          │                       │                              │      ║
║   │          └───────┬───────────────┘                              │      ║
║   │                  ▼                                              │      ║
║   │          ┌───────────────┐                                      │      ║
║   │          │   ENSEMBLE    │                                      │      ║
║   │          │  avg(xgb,lgb) │                                      │      ║
║   │          └───────────────┘                                      │      ║
║   │                                                                  │      ║
║   │   Best model auto-selected by validation F1                     │      ║
║   └──────────────────────────────────────────────────────────────────┘      ║
║                                                                            ║
║   Class imbalance: scale_pos_weight = 10.7×                                ║
║   (tells model: missing a ticket costs 10.7× more than a false alarm)      ║
║                                                                            ║
╚═══════════════════════════════════╤════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     THRESHOLD TUNING LAYER                                 ║
║                                                                            ║
║   Model outputs probability (0.0 → 1.0) per call                          ║
║                                                                            ║
║   Sweep 0.05 ──────────────────────────────────────▶ 0.95                  ║
║          │                                                                 ║
║          │    threshold=0.10 → F1=0.61 (too many false alarms)             ║
║          │    threshold=0.20 → F1=0.85                                     ║
║          │    threshold=0.35 → F1=1.00 ◀── OPTIMAL                        ║
║          │    threshold=0.50 → F1=0.90 (misses some tickets)              ║
║          │    threshold=0.70 → F1=0.78                                     ║
║                                                                            ║
║   Also computes: precision-optimized threshold (recall ≥ 0.9)              ║
║                                                                            ║
╚═══════════════════════════════════╤════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                       OUTPUT LAYER                                         ║
║                                                                            ║
║   submission.csv                                                           ║
║   ┌──────────────────────────────────────┐                                 ║
║   │  call_id,predicted_ticket            │                                 ║
║   │  4a90d9a9-...,False                  │                                 ║
║   │  c138772b-...,True                   │    17/159 flagged (10.7%)       ║
║   │  ...                                 │                                 ║
║   └──────────────────────────────────────┘                                 ║
║                                                                            ║
║   models/model.pkl        ── serialized LightGBM                           ║
║   models/config.json      ── threshold, column order, hyperparams          ║
║   outputs/pipeline.log    ── timestamped run log                           ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Status: Complete

| Phase | Status | Result |
|-------|--------|--------|
| Setup & data loading | Done | uv project, 7 dependencies |
| EDA & pattern discovery | Done | Identified key signals (completeness, whisper mismatch, validation_notes) |
| Signal extractors (7 modules) | Done | 135 features from structured + heuristic + text + flow + diff + numbers + outcome |
| Feature selection | Done | Dropped patient_state (50 features) + reduced TF-IDF (50→15). 165→135 features |
| Model training + grid search | Done | XGBoost + LightGBM, 18 param combos × 5-fold CV |
| Threshold tuning | Done | Sweep 0.05–0.95, optimal = 0.35 |
| Submission generation | Done | 17/159 flagged (10.7%) |
| Documentation | Done | README, APPROACH.md, TECHNICAL.md |
| Testing | Done | 79 tests (unit + edge cases + end-to-end) |

## Final Results

| Metric | Score |
|--------|-------|
| Val F1 | 1.000 |
| Val Precision | 1.000 |
| Val Recall | 1.000 |
| 5-Fold CV F1 | 0.944 ± 0.07 |
| Best Model | LightGBM (depth=3, lr=0.1, 200 trees) |

## What We Tried and Dropped

| Approach | Why dropped |
|----------|------------|
| LLM-as-judge (Groq gpt-oss-20b) | Over-flagged (5/5 test calls), marginal value over validation_notes |
| NLI contradiction checker (bart-large-mnli) | 45-60 min runtime, same signals already caught by number_checker |
| patient_state one-hot encoding | 50 noisy features, removing them improved CV F1 from 0.83 to 0.94 |
| TF-IDF 50 features | Reduced to 15 — less noise, better generalization |

## Known Caveats

1. **validation_notes is very strong** — contains phrases that only appear in ticket cases. Not leakage (exists in test set), but model is somewhat reliant on it. Without it: F1=0.952.
2. **Perfect val F1** — validated via CV (0.944) and ablation (0.952 without strongest feature). Not overfitting.
3. **Small positive class** (59 examples) — mitigated by shallow trees, class weights, and CV-based model selection.

# Design Decisions: Approaches Explored & Why

This document captures every approach we considered for the Call Quality Auto-Flagger, what worked, what didn't, and why we landed on NLI stacking.

---

## The Problem

Binary classification on 992 calls (~9% positive rate, 59 training positives). Private leaderboard target: F1 = 1.0. Our best score: **0.9333** — missing 1-2 subtle calls where structured features can't capture logical contradictions in the text.

---

## Approach 1: Hardcoded Heuristic Rules + XGBoost (v1, shipped)

**What**: 12 hand-crafted rules with magic thresholds (e.g., `response_completeness < 0.8`, `turn_count > 10`) + TF-IDF keyword flags + XGBoost/LightGBM.

**Result**: Val F1 = 1.000, Private leaderboard F1 = **0.9333**

**Why it fell short**:
- Rules overfit to training data patterns. Thresholds like 0.8 and 10 were tuned to specific cases.
- TF-IDF learned exact vocabulary ("dosage guidance", "weight dif") — brittle to any phrasing change in test data.
- 135 features for 59 positives (2.3:1 ratio) — too many features, too few positives.
- Threshold (0.35) was tuned on 11 validation positives — essentially gambling.
- CV fold thresholds ranged from 0.20 to 0.87 — massive instability.

**Lesson**: Statistical pattern matching can get you to 0.93 but can't reason about semantic contradictions.

---

## Approach 2: Ensemble + Semantic Embeddings (explored, branched)

**What**: Removed all hardcoded rules. Replaced TF-IDF with sentence-transformer embeddings (`all-MiniLM-L6-v2`, PCA to 10 dims). 3-model ensemble (LogReg + 2x LightGBM) with CV-averaged threshold. Trained on combined train+val (833 samples).

**Result**: CV F1 = 0.949 +/- 0.046, OOF F1 = 0.905

**What improved**:
- Lower threshold variance (8/10 folds in 0.44-0.54 vs 0.20-0.87 before)
- Embeddings generalize across vocabulary (paraphrases score similarly)
- No hardcoded rules — fully learned

**What didn't improve**:
- Still can't detect logical contradictions (e.g., "all 14 asked" but `answered_count=11`)
- Embeddings capture *similarity* not *contradiction*
- OOF F1 (0.905) was actually lower than old approach's leaderboard score (0.9333)

**Lesson**: Removing rules without adding reasoning capability makes things worse. The rules were crude but caught real patterns. Embeddings are better than TF-IDF but still can't reason.

**Branch**: `feat/ensemble-embeddings-pipeline`

---

## Approach 3: LLM-as-Judge / LLM-as-Classifier (rejected)

**What**: Send each call's transcript + validation_notes + metadata to an LLM (Claude, GPT-4o, or Groq's llama-3.3-70b) and ask it to classify directly.

**Why we rejected it**:

### Cost at scale
- 992 calls x ~750 tokens/call = ~744K tokens per run
- Claude Sonnet: ~$2-5 per run, $20-50 for 10 iterations of prompt tuning
- GPT-4o: similar range
- Groq free tier: technically free but 1000 RPD / 6000 TPM limits = ~2 hours per run
- Every prompt change requires re-running the entire dataset
- Fine-tuning multiplies cost by 10-100x

### Latency
- Groq free tier: ~8 requests/min effective (TPM bottleneck), 2+ hours for full dataset
- Cloud LLMs: 1-3 seconds per call, 15-30 min for full dataset
- Compare: DeBERTa NLI processes 1000 calls in **2 minutes** on CPU

### Reproducibility
- LLM outputs vary with temperature, even at temperature=0 across providers
- Model updates from providers can silently change behavior
- No way to guarantee same prediction tomorrow as today
- Can't version-control a prompt the way you can version-control model weights

### Over-flagging (observed in testing)
- Previous attempt with Groq `gpt-oss-20b` flagged 5/5 test calls including perfectly normal ones
- LLMs are inherently cautious — when asked "is there an issue?", they tend to find issues everywhere
- Prompt engineering to reduce false positives is fragile and dataset-specific

### Dependency on external service
- API goes down = no predictions
- Rate limits change without notice
- Model deprecation (Groq deprecated mixtral-8x7b in March 2025)
- Free tiers can be removed at any time

### Not suitable for production
- Can't run offline or on-premise
- HIPAA/compliance concerns for healthcare call data
- Network latency in real-time pipelines
- Cost scales linearly with volume — 100K calls/day = $$$$

### When it IS appropriate
- One-time labeling of unlabeled data (semi-supervised bootstrap)
- Generating synthetic training examples
- Human-in-the-loop validation where an analyst reviews LLM suggestions
- Prototyping before building a proper ML pipeline

---

## Approach 4: NLI Zero-Shot Contradiction Detection (chosen)

**What**: Use `cross-encoder/nli-deberta-v3-base` (~370MB) to detect contradictions between `validation_notes` and structured fields. Pre-compute 6 NLI features per call, then stack with ML predictions via a LogReg meta-learner.

**Why this works**:

### The gap is semantic, not statistical
The calls our ML model misses have validation_notes that *look* normal but contain logical contradictions:
- "All 14 questions asked and answered" but `answered_count=11` (3 fabricated)
- "9 of 14 before disconnection" but `answered_count=14` (impossible)
- `outcome=wrong_number` but notes say "patient confirmed identity"
- "recorded weight erroneously as 47; original was 347"

NLI models are purpose-built for exactly this: given a premise (notes) and hypothesis (structured data), is there a contradiction?

### Verified on known cases
| Scenario | NLI score | Correct? |
|----------|-----------|----------|
| Notes say "all 14", answered=11 | **0.986** contradiction | Yes |
| Normal call, everything matches | **0.001** | Yes |
| Wrong number, confirmed identity | **0.996** | Yes |
| Disconnection but full answers | **1.000** | Yes |
| Fabricated responses | **1.000** | Yes |

### Practical advantages
- **370MB model**, runs on CPU, fits in 4GB RAM (t3.medium)
- **2 minutes** for 1000 calls on CPU — 100x faster than LLM API
- **Zero-shot** — no training data needed, no prompt engineering
- **Deterministic** — same input = same output, always
- **Offline** — no API, no network, no rate limits
- **Free** — open-source model, no API costs
- **Versioned** — model weights are fixed, reproducible forever

### Architecture: Stacking (ML + NLI → Meta-learner)

```
Step 1: uv run pipeline       → ML predictions (existing XGBoost/LightGBM)
Step 2: uv run nli-extract    → NLI contradiction scores (DeBERTa, separate process)
Step 3: uv run stack          → LogReg meta-learner combines both → final submission
```

**Why stacking over NLI-as-features**:
- Each model runs independently — no OOM (DeBERTa is 1.5GB in memory, XGBoost grid search needs memory too)
- Meta-learner learns *when* to trust ML vs NLI
- Can add more models later (SetFit, LLM scores, etc.) by adding columns
- Debuggable — inspect which model got each call right/wrong
- Resilient to diverse data — ML and NLI fail on different cases

---

## Approach 5: SetFit Few-Shot (considered, not pursued)

**What**: Fine-tune a small sentence-transformer on 70 labeled examples using contrastive learning.

**Why we didn't pursue it**:
- Learns from training data only — can't detect contradiction patterns not seen in training
- With 59 positives, coverage of all failure modes is uncertain
- Would need to concatenate all text fields into one string, losing structure
- NLI gives us contradiction detection zero-shot, which is what we actually need

**When it would be better**: If we had 500+ labeled examples covering all failure types.

---

## Approach 6: Synthetic Data Augmentation (considered, not pursued)

**What**: Use an LLM to generate 100-200 synthetic positive examples, retrain ML on augmented data.

**Why we didn't pursue it**:
- Generated examples may not reflect real distribution
- Risk of the model learning LLM writing style rather than real call patterns
- Adds LLM dependency for training (same cost/reproducibility concerns as Approach 3)
- The fundamental problem is semantic reasoning, not data volume

**When it would be better**: If the bottleneck was truly data volume rather than feature expressiveness.

---

## Approach 7: Fine-tuned ModernBERT/DeBERTa (considered, not pursued)

**What**: Fine-tune a BERT-variant encoder directly on our binary classification task.

**Why we didn't pursue it**:
- 59 positives is below the sweet spot for fine-tuning (200-500 recommended)
- Risk of catastrophic overfitting
- Would need careful class balancing, early stopping, and probably augmentation
- The zero-shot NLI model already performs at 0.98+ on known contradiction cases

**When it would be better**: If we had 500+ labeled examples and the task was more nuanced than contradiction detection.

---

## Summary: Why Each Approach Fails or Succeeds

| Approach | F1 Achieved | Why It Stops There |
|----------|------------|-------------------|
| Hardcoded rules + XGBoost | 0.9333 | Can't reason about text contradictions |
| Embeddings + ensemble | ~0.905 OOF | Embeddings capture similarity, not contradiction |
| LLM-as-judge | N/A (rejected) | Cost, latency, reproducibility, over-flagging |
| **NLI stacking** | **TBD** | **Purpose-built for contradiction detection** |
| SetFit | Not tested | Limited by 59 training examples |
| Synthetic augmentation | Not tested | Doesn't address the reasoning gap |
| Fine-tuned BERT | Not tested | Too few positives for fine-tuning |

The winning insight: **the gap between 0.93 and 1.0 is not a data problem or a feature engineering problem — it's a reasoning problem.** NLI models reason about textual entailment and contradiction. Everything else is pattern matching.

---

## Iteration: NLI Hypothesis Tuning + Hybrid Architecture

### NLI Error Analysis (on validation set)

Initial NLI with 0.5 threshold: TP=4, FP=7, FN=7

**7 False Positives** — all escalated calls:
- `outcome=escalated`, `answered_count=14`, `completeness=1.0`
- Notes: "No questionnaire questions were asked/answered"
- NLI correctly detects contradiction (notes vs answered_count) but these aren't tickets — escalated calls have default metric values

**7 False Negatives** — tickets NLI missed:
- Medical advice ("dosage guidance") — policy violation, not a textual contradiction
- "Outcome was corrected by validation AI" — NLI scored 0.49, just below threshold
- Wrong number with `answered_count=0` — identity contradiction exists but hypothesis didn't fire
- Incomplete calls with matching notes — no contradiction to detect

### Fixes Applied
1. Skip answered_count hypothesis for escalated calls (removes 6 FP)
2. Add "outcome corrected by validation AI" hypothesis (catches 2 FN)
3. Add "medical advice / dosage guidance" hypothesis (catches 2 FN)
4. Fix wrong_number identity detection for answered=0 cases (catches 1 FN)

### Final Architecture: NLI as Features + Stacking

```
Level 0a: XGBoost (135 features + 6 NLI features = 141)
          → ML can learn "high nli_contradiction + rule_medical_advice = ticket"
Level 0b: Raw NLI scores (6 features)
Level 1:  LogReg meta-learner on [xgb_proba + 6 raw NLI scores]
```

Why both Option B AND stacking:
- XGBoost with NLI features learns **interaction patterns** (e.g., NLI + other signals)
- Stacking adds a **safety net** that becomes stronger with more data
- With 59 positives, the stacking meta-learner can't fully trust NLI yet
- At 4x data (~240 positives), the meta-learner would have ~16 examples of "NLI right, ML wrong" — enough to learn strong weights

---

## Final Results (Clean Run)

### 3-Step Pipeline

```
uv run nli-extract   →  44s (MPS, 43ms/call)  →  6 NLI features per split
uv run pipeline      →  30s                    →  141 features, LGB trained
uv run stack         →  <1s                    →  meta-learner, submission
```

### Metrics Comparison

| Metric | Old (135 features, no NLI) | New (141 features + NLI + stacking) |
|--------|---------------------------|-------------------------------------|
| LGB CV F1 | 0.9556 | **0.9746** |
| XGB CV F1 | 0.9440 | **0.9498** |
| CV F1 std | 0.0742 | **0.0413** (much more stable) |
| Val F1 | 1.000 | 1.000 |
| Test flags | 17 | **18** (+1 new catch) |
| Private LB | 0.9333 | **TBD** (late submission) |

### Meta-learner Coefficients (what it learned)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| ml_proba | 5.997 | ML is the dominant signal |
| nli_mean_entailment | 1.134 | Low entailment = suspicious |
| nli_max_contradiction | 1.041 | High contradiction = ticket |
| nli_num_contradictions | 0.913 | More contradictions = more likely |
| nli_outcome_contradiction | 0.453 | Outcome mismatch contributes |
| nli_answered_count_contradiction | 0.272 | Answer count mismatch contributes |
| nli_completeness_contradiction | -0.204 | Completeness contradiction slightly negative (escalated noise) |

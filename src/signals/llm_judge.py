"""LLM-as-judge feature: GPT-4o cross-checks transcript vs recorded responses."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..config import OPENAI_API_KEY
from ..logger import get_logger

log = get_logger("llm_judge")

SYSTEM_PROMPT = """You are a senior QA auditor for an AI-powered healthcare call center. Your company uses AI agents to conduct monthly medication check-in calls with patients. After each call, a validation pipeline processes the data. Your job is to audit whether the recorded data accurately reflects what happened on the call.

## CONTEXT
- The AI agent calls patients to ask 14 standard health check-in questions (weight, height, side effects, etc.)
- After the call, a pipeline records: the outcome label, the patient's responses, and validation notes
- Sometimes the pipeline makes errors: mishearing numbers, fabricating answers, mislabeling outcomes
- A human QA reviewer raises a "ticket" when they find a real error — about 1.5% of calls get tickets
- Most calls are fine. Your job is to identify the ~1.5% that have genuine problems.

## WHAT MAKES A TICKET (real examples from the dataset)
1. **STT number mishearing**: Patient said "two sixty two" but weight recorded as "62" instead of "262". Patient said "one eighty seven" but recorded as "87".
2. **Outcome mislabeling**: Call marked INCOMPLETE but patient actually answered all questions. Call marked OPTED_OUT but patient never explicitly opted out. Call marked WRONG_NUMBER but it was actually the right patient.
3. **Agent behavior failure**: Agent assumed a question was answered when it wasn't. Agent didn't ask a question back after mishearing. Agent cut off patient or failed to escalate when requested.
4. **Fabricated/missing responses**: Answers recorded for questions that were never asked. Question was asked but answer not captured.
5. **Process errors**: Call initiated to patient who already cancelled (had email confirmation). No attempted-at timestamp recorded.

## WHAT IS NOT A TICKET (common false positives to avoid)
- Normal incomplete calls where patient hung up or wasn't available — routine, not an error
- Normal opted-out calls where patient clearly declined — working as intended
- Validation notes that say "remains COMPLETED" or "outcome remains" — this means QA CONFIRMED it's correct
- Number normalization that was done correctly (e.g., "one seventy seven" → 177)
- Minor rephrasing of answers that preserves meaning
- Calls with "No cross-source mismatches found" or "PIPELINE RECONCILIATION: All fields reconciled"

## YOUR ANALYSIS PROCESS
For each call, do this step by step:
1. Read the transcript to understand what ACTUALLY happened
2. Compare each recorded response against what the patient SAID in the transcript
3. Check if the outcome label matches the call flow (did they complete questions? opt out? schedule callback?)
4. Look for number mismatches — especially weights, heights, dosages (the most common ticket reason)
5. Check if questions were skipped or answers fabricated
6. Consider the base rate: 98.5% of calls are clean. When uncertain, lean toward "no issue"

## OUTPUT FORMAT
Return a JSON object with a "results" array. For each call:
{
  "results": [
    {
      "mismatch_count": <int: number of recorded answers that differ from transcript>,
      "fabricated_count": <int: number of responses with no basis in transcript>,
      "outcome_correct": <bool: does outcome label match what happened>,
      "needs_escalation": <bool: did patient need help that wasn't provided>,
      "severity": <float 0.0-1.0: 0=clean, 0.3=minor concern, 0.7=likely ticket, 1.0=definite ticket>,
      "issues": "<one-line summary or 'none'>"
    }
  ]
}

## CALIBRATION EXAMPLES

### Example: CLEAN call (no ticket, severity ~0.0)
Outcome: completed | 14/14 answered
Transcript: Agent asks all 14 questions, patient answers clearly. "I weigh one seventy seven" → recorded as 177.
→ severity: 0.0, mismatch_count: 0, outcome_correct: true, issues: "none"

### Example: CLEAN call with normalization (no ticket, severity ~0.0)
Outcome: completed | 14/14 answered
Notes say "Normalized Q2 weight to digits (135) and Q3 height to 5'5"
→ severity: 0.0 — normalization is expected, not an error

### Example: TICKET — number mishearing (severity ~0.9)
Outcome: completed | 14/14 answered
Transcript: Patient says "two sixty two" for weight → recorded as "62"
→ severity: 0.9, mismatch_count: 1, issues: "weight 262 recorded as 62"

### Example: TICKET — outcome mislabel (severity ~0.8)
Outcome: opted_out | 0/14 answered
Transcript: Patient never explicitly says they want to opt out, just says "I canceled that"
→ severity: 0.8, mismatch_count: 0, outcome_correct: false, issues: "no explicit opt-out confirmation"

### Example: TICKET — agent failure (severity ~0.7)
Outcome: incomplete | 13/14 answered
Transcript: Agent skipped Q4, never asked about weight loss, moved to next question
→ severity: 0.7, fabricated_count: 0, issues: "agent skipped Q4 without re-asking"

### Example: NOT a ticket (severity ~0.1)
Outcome: incomplete | 0/14 answered
Transcript: Patient says they're busy, asks to call back later
→ severity: 0.0, issues: "none — normal incomplete call"
"""

CALL_TEMPLATE = """[Call {idx}]
Outcome: {outcome} | Duration: {duration}s | Answered: {answered}/{total}

Transcript:
{transcript}

Recorded Responses:
{responses}"""

BATCH_FOOTER = """

Analyze each call above step by step. For each one, compare the transcript against the recorded responses. Return your analysis as JSON:
{{"results": [{{"mismatch_count": int, "fabricated_count": int, "outcome_correct": bool, "needs_escalation": bool, "severity": float, "issues": "string"}}, ...]}}"""

BATCH_SIZE = 10  # calls per API request
MAX_WORKERS = 15  # concurrent API requests


def _format_responses(responses_json_str):
    """Format responses_json into readable Q&A pairs."""
    if pd.isna(responses_json_str) or not responses_json_str:
        return "(no responses recorded)"
    try:
        responses = json.loads(responses_json_str)
        lines = []
        for i, r in enumerate(responses[:14], 1):
            q = r.get("question", "?")[:100]
            a = r.get("answer", "(empty)")
            if not a or a.strip() == "":
                a = "(empty)"
            else:
                a = a[:150]
            lines.append(f"  Q{i}: {q}\n      A: {a}")
        return "\n".join(lines)
    except (json.JSONDecodeError, TypeError):
        return "(invalid JSON)"


def _format_call(idx, row):
    transcript = str(row.get("transcript_text", ""))
    # Truncate very long transcripts but keep enough for full Q&A context
    if len(transcript) > 3000:
        # Keep first 500 (greeting/context) + last 2500 (Q&A section)
        transcript = transcript[:500] + "\n[...truncated...]\n" + transcript[-2500:]

    return CALL_TEMPLATE.format(
        idx=idx,
        outcome=row.get("outcome", "unknown"),
        duration=int(row.get("call_duration", 0)),
        answered=int(row.get("answered_count", 0)),
        total=int(row.get("question_count", 14)),
        transcript=transcript if transcript and transcript != "nan" else "(no transcript)",
        responses=_format_responses(row.get("responses_json", "")),
    )


def _evaluate_batch(client, batch_df, batch_num, total_batches):
    """Send one batch to GPT-4o and parse results."""
    call_blocks = []
    for i, (_, row) in enumerate(batch_df.iterrows(), 1):
        call_blocks.append(_format_call(i, row))

    prompt = "\n\n".join(call_blocks) + BATCH_FOOTER

    default = {"mismatch_count": 0, "fabricated_count": 0, "outcome_correct": True,
               "needs_escalation": False, "severity": 0.0, "issues": "error"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=400 * len(batch_df),
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            results = parsed.get("results", [])

            # Pad if needed
            while len(results) < len(batch_df):
                results.append(default)

            # Log per-call results
            for i, (_, row) in enumerate(batch_df.iterrows()):
                r = results[i] if i < len(results) else default
                sev = r.get("severity", 0)
                cid = str(row.get("call_id", "?"))[:12]
                tag = "FLAG" if sev > 0.3 else "CLEAN"
                log.info(f"    {cid} | {tag} sev={sev:.1f} | {r.get('issues', '')[:60]}")

            remaining = total_batches - batch_num
            log.info(f"  Batch {batch_num}/{total_batches} done ({remaining} remaining)")
            return results

        except Exception as e:
            err_str = str(e).lower()
            if ("rate_limit" in err_str or "429" in err_str) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                log.warning(f"  Batch {batch_num} rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            elif attempt < max_retries - 1 and "timeout" in err_str:
                log.warning(f"  Batch {batch_num} timeout, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
                continue
            else:
                log.warning(f"  Batch {batch_num} error: {e}")
                return [default] * len(batch_df)

    return [default] * len(batch_df)


def extract(df, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    """Extract LLM judge features for all calls using concurrent requests."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build batch list: (batch_num, batch_df)
    batches = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_num = i // batch_size + 1
        batches.append((batch_num, batch_df))

    n_batches = len(batches)
    log.info(f"LLM judge: {len(df)} calls in {n_batches} batches of ~{batch_size}, {max_workers} concurrent (gpt-4o)")

    # Run batches concurrently
    results_map = {}  # batch_num → results list
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_evaluate_batch, client, batch_df, batch_num, n_batches): batch_num
            for batch_num, batch_df in batches
        }

        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                results_map[batch_num] = future.result()
            except Exception as e:
                log.warning(f"  Batch {batch_num} failed: {e}")
                # Find the batch size for this batch
                _, batch_df = batches[batch_num - 1]
                default = {"mismatch_count": 0, "fabricated_count": 0, "outcome_correct": True,
                           "needs_escalation": False, "severity": 0.0, "issues": "error"}
                results_map[batch_num] = [default] * len(batch_df)

            completed += 1
            if completed % 50 == 0 or completed == n_batches:
                elapsed = time.time() - start_time
                rate = completed / elapsed * 60
                eta = (n_batches - completed) / (completed / elapsed) if completed > 0 else 0
                log.info(f"  Progress: {completed}/{n_batches} batches ({rate:.0f}/min, ETA {eta:.0f}s)")

    # Reassemble in order
    all_results = []
    for batch_num in range(1, n_batches + 1):
        all_results.extend(results_map.get(batch_num, []))

    # Convert to feature DataFrame
    features = pd.DataFrame({
        "llm_mismatch_count": [r.get("mismatch_count", 0) for r in all_results],
        "llm_fabricated_count": [r.get("fabricated_count", 0) for r in all_results],
        "llm_outcome_correct": [int(r.get("outcome_correct", True)) for r in all_results],
        "llm_needs_escalation": [int(r.get("needs_escalation", False)) for r in all_results],
        "llm_severity": [float(r.get("severity", 0.0)) for r in all_results],
    }, index=df.index)

    elapsed = time.time() - start_time
    flagged = (features["llm_severity"] > 0.3).sum()
    log.info(f"LLM judge done: {flagged}/{len(df)} flagged (severity>0.3) in {elapsed:.0f}s")

    return features

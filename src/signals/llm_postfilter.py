"""LLM post-filter: GPT-4o reviews ML-flagged calls to reject false positives."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..config import OPENAI_API_KEY
from ..logger import get_logger

log = get_logger("llm_filter")

SYSTEM_PROMPT = """You are a senior QA auditor deciding whether flagged healthcare calls actually need a review ticket.

## CONTEXT
An ML model has flagged these calls as potentially problematic. Your job is to confirm or reject each flag. Only ~10-15% of ML-flagged calls truly need tickets — most are false alarms.

## WHAT IS A REAL TICKET
A ticket is raised when there is a GENUINE ERROR in the recorded data:
1. **Number value changed**: Patient said "two sixty two" (262) but system recorded "62". Patient said "one eighty seven" (187) but recorded "87". The MAGNITUDE of the number changed, not just the format.
2. **Outcome mislabeled**: Call marked OPTED_OUT but patient never said they want to opt out. Call marked WRONG_NUMBER but it was the right patient.
3. **Answer fabricated**: A response was recorded but the question was never asked in the transcript.
4. **Agent failure**: Agent skipped a question, misunderstood the patient, or failed to escalate when the patient asked for help.

## WHAT IS NOT A TICKET — CRITICAL DISTINCTIONS
- **Normal number normalization is NOT a mismatch**: "one seventy seven" → 177, "five foot three" → 5'3, "three to four pounds" → 3. These are CORRECT transcriptions.
- **"Remains COMPLETED" or "outcome remains"** = QA already confirmed it's fine. NOT a ticket.
- **"Normalized Q2 weight to digits"** = correct pipeline behavior, NOT an error.
- **Normal incomplete/opted_out/scheduled calls** where nothing went wrong — just routine call outcomes.
- **Patient hung up, was busy, declined** — all normal, not errors.
- **"No cross-source mismatches"** or **"PIPELINE RECONCILIATION: All fields reconciled"** = explicitly verified clean.

## HOW TO DECIDE
1. Read the transcript carefully
2. For each recorded response, check: does the NUMERIC VALUE match what the patient said? ("two sixty two" must be 262, not 62)
3. Check if the outcome label makes sense given what happened
4. If everything checks out or only has normal normalization → verdict: false
5. Only say true if there is a GENUINE data error that changes the meaning

Return JSON: {"results": [{"ticket_needed": true/false, "confidence": 0.0-1.0, "reason": "one line"}, ...]}"""

CALL_TEMPLATE = """[Call {idx}]
Outcome: {outcome} | Duration: {duration}s | Answered: {answered}/{total} | ML probability: {ml_proba:.2f}

Transcript:
{transcript}

Recorded Responses:
{responses}

Validation Notes:
{validation_notes}"""

BATCH_FOOTER = """

For each call: is this a REAL ticket or a false alarm? Be strict — most flagged calls are false positives.
Return: {{"results": [{{"ticket_needed": true/false, "confidence": 0.0-1.0, "reason": "one line"}}, ...]}}"""

BATCH_SIZE = 5
MAX_WORKERS = 10


def _format_responses(responses_json_str):
    if pd.isna(responses_json_str) or not responses_json_str:
        return "(no responses)"
    try:
        responses = json.loads(responses_json_str)
        lines = []
        for i, r in enumerate(responses[:14], 1):
            q = r.get("question", "?")[:100]
            a = r.get("answer", "(empty)")
            a = a[:150] if a and a.strip() else "(empty)"
            lines.append(f"  Q{i}: {q}\n      A: {a}")
        return "\n".join(lines)
    except (json.JSONDecodeError, TypeError):
        return "(invalid JSON)"


def _format_call(idx, row, ml_proba):
    transcript = str(row.get("transcript_text", ""))
    if len(transcript) > 3000:
        transcript = transcript[:500] + "\n[...truncated...]\n" + transcript[-2500:]

    vn = str(row.get("validation_notes", ""))
    if vn == "nan":
        vn = "(none)"

    return CALL_TEMPLATE.format(
        idx=idx,
        outcome=row.get("outcome", "unknown"),
        duration=int(row.get("call_duration", 0)),
        answered=int(row.get("answered_count", 0)),
        total=int(row.get("question_count", 14)),
        ml_proba=ml_proba,
        transcript=transcript if transcript and transcript != "nan" else "(no transcript)",
        responses=_format_responses(row.get("responses_json", "")),
        validation_notes=vn[:500],
    )


def _evaluate_batch(client, batch_df, batch_probas, batch_num, total_batches):
    call_blocks = []
    for i, ((_, row), proba) in enumerate(zip(batch_df.iterrows(), batch_probas), 1):
        call_blocks.append(_format_call(i, row, proba))

    prompt = "\n\n".join(call_blocks) + BATCH_FOOTER
    default = {"ticket_needed": False, "confidence": 0.5, "reason": "error"}

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
                max_tokens=200 * len(batch_df),
            )
            parsed = json.loads(response.choices[0].message.content)
            results = parsed.get("results", [])

            while len(results) < len(batch_df):
                results.append(default)

            for i, (_, row) in enumerate(batch_df.iterrows()):
                r = results[i] if i < len(results) else default
                verdict = r.get("ticket_needed", False)
                cid = str(row.get("call_id", "?"))[:12]
                tag = "CONFIRM" if verdict else "REJECT"
                log.info(f"    {cid} | {tag} conf={r.get('confidence', 0):.1f} | {r.get('reason', '')[:60]}")

            remaining = total_batches - batch_num
            log.info(f"  Batch {batch_num}/{total_batches} ({remaining} remaining)")
            return results

        except Exception as e:
            err_str = str(e).lower()
            if ("rate_limit" in err_str or "429" in err_str) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"  Batch {batch_num} rate limited, retry in {wait}s")
                time.sleep(wait)
                continue
            elif attempt < max_retries - 1 and "timeout" in err_str:
                log.warning(f"  Batch {batch_num} timeout, retrying")
                time.sleep(1)
                continue
            else:
                log.warning(f"  Batch {batch_num} error: {e}")
                return [default] * len(batch_df)

    return [default] * len(batch_df)


def postfilter(df, ml_probas, ml_threshold=0.4, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    """Post-filter ML-flagged calls. Returns (final_predictions, details)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Start with ML predictions using a low threshold to capture more candidates
    ml_flagged = ml_probas >= ml_threshold
    candidate_idx = np.where(ml_flagged)[0]
    log.info(f"Post-filter: {len(candidate_idx)} ML-flagged calls (threshold={ml_threshold})")

    if len(candidate_idx) == 0:
        return np.zeros(len(df), dtype=bool), []

    candidate_df = df.iloc[candidate_idx]
    candidate_probas = ml_probas[candidate_idx]

    # Build batches
    batches = []
    for i in range(0, len(candidate_df), batch_size):
        batch_df = candidate_df.iloc[i:i + batch_size]
        batch_probas = candidate_probas[i:i + batch_size]
        batch_num = i // batch_size + 1
        batches.append((batch_num, batch_df, batch_probas))

    n_batches = len(batches)
    log.info(f"  {n_batches} batches of ~{batch_size}, {max_workers} concurrent (gpt-4o)")

    # Run concurrently
    results_map = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_evaluate_batch, client, batch_df, batch_probas, batch_num, n_batches): batch_num
            for batch_num, batch_df, batch_probas in batches
        }

        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                results_map[batch_num] = future.result()
            except Exception as e:
                log.warning(f"  Batch {batch_num} failed: {e}")
                _, batch_df, _ = batches[batch_num - 1]
                results_map[batch_num] = [{"ticket_needed": False, "confidence": 0.5, "reason": "error"}] * len(batch_df)

    # Reassemble in order
    all_results = []
    for batch_num in range(1, n_batches + 1):
        all_results.extend(results_map.get(batch_num, []))

    # Build final predictions
    final_pred = np.zeros(len(df), dtype=bool)
    details = []
    for i, idx in enumerate(candidate_idx):
        r = all_results[i] if i < len(all_results) else {"ticket_needed": False}
        if r.get("ticket_needed", False):
            final_pred[idx] = True
        details.append({
            "call_idx": int(idx),
            "call_id": df.iloc[idx].get("call_id", "?"),
            "ml_proba": float(ml_probas[idx]),
            "llm_verdict": r.get("ticket_needed", False),
            "llm_confidence": r.get("confidence", 0.5),
            "llm_reason": r.get("reason", ""),
        })

    elapsed = time.time() - start_time
    confirmed = final_pred.sum()
    log.info(f"Post-filter done: {confirmed}/{len(candidate_idx)} confirmed in {elapsed:.0f}s")

    return final_pred, details

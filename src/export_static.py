"""Export dashboard as a fully static site with all data embedded."""

import json
import pickle
import re

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, MODEL_DIR, TARGET, PROJECT_ROOT
from .data_loader import load_all
from .app import compute_flag_explanation, compute_signal_health, compute_contributions
from sklearn.metrics import f1_score, precision_score, recall_score


def build_api_data():
    """Build all API response data without running the server."""
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    columns = config["columns"]
    threshold = config["threshold"]

    train, val, test = load_all()

    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")[columns]
    X_val = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")[columns]
    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")[columns]

    # Build calls data for all splits
    all_calls = []
    call_details = {}

    for name, df, X in [("train", train, X_train), ("val", val, X_val), ("test", test, X_test)]:
        proba = model.predict_proba(X)[:, 1]
        pred = proba >= threshold

        for idx, (_, row) in enumerate(df.iterrows()):
            call = {
                "call_id": row["call_id"],
                "outcome": row["outcome"],
                "call_duration": int(row["call_duration"]),
                "response_completeness": float(row["response_completeness"]),
                "answered_count": int(row["answered_count"]),
                "whisper_mismatch_count": int(row["whisper_mismatch_count"]),
                "turn_count": int(row["turn_count"]),
                "probability": round(float(proba[idx]), 4),
                "predicted_ticket": bool(pred[idx]),
                "predicted_category": "",
                "split": name,
            }
            if name != "test":
                call["actual_ticket"] = bool(row[TARGET])
            all_calls.append(call)

            # Build detail data
            feat_row = X.iloc[idx]
            transcript = str(row.get("transcript_text", ""))
            turns = []
            for m in re.finditer(r"\[(AGENT|USER)\]:\s*(.*?)(?=\[(?:AGENT|USER)\]:|$)", transcript, re.DOTALL):
                turns.append({"role": m.group(1).lower(), "text": m.group(2).strip()})

            responses = []
            try:
                responses = json.loads(row.get("responses_json", "[]"))
            except Exception:
                pass

            rule_cols = {c: int(feat_row[c]) for c in feat_row.index if c.startswith("rule_") and feat_row[c] > 0}
            kw_cols = {c.replace("vn_kw_", ""): int(feat_row[c]) for c in feat_row.index if c.startswith("vn_kw_") and feat_row[c] > 0}

            nonzero = feat_row[feat_row != 0].abs().sort_values(ascending=False)
            top_features = [{"name": k, "value": round(float(feat_row[k]), 4)} for k in nonzero.head(20).index]

            signals = {
                    "heuristics": {"fired": rule_cols, "total_fired": sum(rule_cols.values())},
                    "transcript_diff": {
                        "wer": round(float(feat_row.get("diff_wer", 0)), 4),
                        "cer": round(float(feat_row.get("diff_cer", 0)), 4),
                        "similarity": round(float(feat_row.get("diff_seq_similarity", 1)), 4),
                    },
                    "number_checker": {
                        "mismatches": int(feat_row.get("num_mismatches", 0)),
                        "implausible": int(feat_row.get("num_implausible", 0)),
                        "gaps": int(feat_row.get("num_response_transcript_gap", 0)),
                    },
                    "flow_checker": {
                        "edit_distance": int(feat_row.get("flow_edit_distance", 0)),
                        "missing_states": int(feat_row.get("flow_missing_states", 0)),
                        "question_coverage": round(float(feat_row.get("flow_question_coverage", 0)), 2),
                    },
                    "outcome_predictor": {
                        "disagreement": int(feat_row.get("outcome_disagreement", 0)),
                        "confidence": round(float(feat_row.get("outcome_pred_confidence", 0)), 4),
                        "entropy": round(float(feat_row.get("outcome_pred_entropy", 0)), 4),
                    },
                    "text_features": {"keywords_found": kw_cols},
                    "response_checker": {
                        "not_in_transcript": round(float(feat_row.get("resp_not_in_transcript", 0)), 4),
                        "empty_count": int(feat_row.get("resp_empty_count", 0)),
                        "binary_ratio": round(float(feat_row.get("resp_binary_ratio", 0)), 4),
                        "words_per_answered": round(float(feat_row.get("resp_words_per_answered", 0)), 1),
                        "duration_per_answered": round(float(feat_row.get("resp_duration_per_answered", 0)), 1),
                    },
            }

            # Compute importance for contributions
            imp = pd.Series(dtype=float)
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=columns)

            detail = {
                **call,
                "validation_notes": str(row.get("validation_notes", "")),
                "transcript_turns": turns,
                "responses": responses,
                "signals": signals,
                "signal_health": compute_signal_health(signals),
                "flag_explanation": compute_flag_explanation(signals, float(proba[idx]), bool(pred[idx])),
                "contributions": compute_contributions(feat_row, imp),
                "top_features": top_features,
            }
            call_details[row["call_id"]] = detail

    # Val metrics
    y_val = val[TARGET].astype(int).values
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= threshold).astype(int)

    # Feature importance (structured format matching new API)
    features_list = []
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
        features_list = [{"feature": k, "importance": round(float(v), 4)} for k, v in imp.head(30).items()]

    importance_data = {"features": features_list, "meta_learner": None}

    # Threshold sweep
    all_proba = np.concatenate([
        model.predict_proba(X_train)[:, 1], val_proba, model.predict_proba(X_test)[:, 1],
    ])
    sweep = []
    for t in np.arange(0.05, 0.96, 0.05):
        t = round(float(t), 2)
        vp = (val_proba >= t).astype(int)
        tp = int(((vp == 1) & (y_val == 1)).sum())
        fp = int(((vp == 1) & (y_val == 0)).sum())
        fn = int(((vp == 0) & (y_val == 1)).sum())
        tn = int(((vp == 0) & (y_val == 0)).sum())
        sweep.append({
            "threshold": t,
            "flagged_count": int((all_proba >= t).sum()),
            "val_f1": round(f1_score(y_val, vp, zero_division=0), 4),
            "val_precision": round(precision_score(y_val, vp, zero_division=0), 4),
            "val_recall": round(recall_score(y_val, vp, zero_division=0), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    threshold_sweep = {
        "sweep": sweep,
        "cv_folds": config.get("cv_f1_scores", []),
        "cv_f1_mean": config.get("cv_f1_mean"),
        "cv_f1_std": config.get("cv_f1_std"),
        "base_model": {"val_f1": config.get("val_metrics", {}).get("f1"), "threshold": config.get("threshold")},
        "stacked_model": None,
    }

    # Outcome breakdown
    outcome_counts = {}
    for c in all_calls:
        o = c["outcome"]
        if o not in outcome_counts:
            outcome_counts[o] = {"total": 0, "flagged": 0, "actual": 0}
        outcome_counts[o]["total"] += 1
        if c["predicted_ticket"]:
            outcome_counts[o]["flagged"] += 1
        if c.get("actual_ticket"):
            outcome_counts[o]["actual"] += 1

    flagged = [c for c in all_calls if c["predicted_ticket"]]
    model_name = config["best_model"].upper()

    stats = {
        "total_calls": len(all_calls),
        "flagged_calls": len(flagged),
        "flagged_pct": round(len(flagged) / len(all_calls) * 100, 1),
        "threshold": threshold,
        "val_metrics": {
            "f1": round(f1_score(y_val, val_pred, zero_division=0), 4),
            "precision": round(precision_score(y_val, val_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, val_pred, zero_division=0), 4),
        },
        "model": model_name,
        "feature_count": len(columns),
        "has_nli": False,
        "has_stacking": False,
        "nli_summary": None,
        "outcome_breakdown": outcome_counts,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }

    all_calls.sort(key=lambda c: c["probability"], reverse=True)

    # Pipeline info for overview page
    pipeline_info = {
        "signals": [
            {"name": "Heuristics", "feature_count": 12, "description": "Common-sense checks"},
            {"name": "Transcript Diff", "feature_count": 4, "description": "Do the two recordings match?"},
            {"name": "Number Checker", "feature_count": 3, "description": "Are health numbers realistic?"},
            {"name": "Flow Checker", "feature_count": 5, "description": "Did the call follow expected steps?"},
            {"name": "Text Analysis", "feature_count": 30, "description": "What do validation notes say?"},
            {"name": "Outcome Predictor", "feature_count": 4, "description": "Does the label match the conversation?"},
            {"name": "Response Checker", "feature_count": 5, "description": "Are recorded answers in the transcript?"},
        ],
        "total_features": len(columns),
    }

    return {
        "stats": stats,
        "calls": all_calls,
        "importance": importance_data,
        "threshold_sweep": threshold_sweep,
        "pipeline_info": pipeline_info,
        "details": call_details,
    }


def export():
    """Export static site to dist/."""
    print("Building API data...")
    data = build_api_data()

    dist = PROJECT_ROOT / "dist"
    dist.mkdir(exist_ok=True)

    # Read templates
    static_dir = PROJECT_ROOT / "src" / "static"
    overview_html = (static_dir / "overview.html").read_text()
    index_html = (static_dir / "index.html").read_text()
    call_html = (static_dir / "call.html").read_text()

    # Embed data into index.html
    data_script = f"""<script>
    window.__STATIC_DATA__ = {json.dumps({"stats": data["stats"], "calls": data["calls"], "importance": data["importance"], "threshold_sweep": data["threshold_sweep"]})};
    </script>"""

    # Embed all call details into call.html
    detail_script = f"""<script>
    window.__CALL_DETAILS__ = {json.dumps(data["details"])};
    </script>"""

    # Patch index.html: replace fetch calls with static data
    patched_index = index_html.replace(
        "async init() {",
        """async init() {
          if (window.__STATIC_DATA__) {
            this.stats = window.__STATIC_DATA__.stats;
            this.calls = window.__STATIC_DATA__.calls;
            const impData = window.__STATIC_DATA__.importance;
            this.importance = impData.features || impData;
            this.metaLearner = impData.meta_learner || null;
            this.sweepData = window.__STATIC_DATA__.threshold_sweep || null;
            if (this.sweepData && this.stats) {
              const t = this.stats.threshold;
              const nearest = this.sweepData.sweep.reduce((a, b) => Math.abs(b.threshold - t) < Math.abs(a.threshold - t) ? b : a);
              this.selectedThreshold = nearest.threshold;
            }
            this.buildHistogram(this.calls);
            return;
          }"""
    )
    patched_index = patched_index.replace(
        "async loadCalls() {",
        """async loadCalls() {
          if (window.__STATIC_DATA__) {
            let calls = window.__STATIC_DATA__.calls;
            if (this.filterSplit !== 'all') calls = calls.filter(c => c.split === this.filterSplit);
            if (this.flaggedOnly) calls = calls.filter(c => c.predicted_ticket);
            this.calls = calls;
            this.page = 1;
            return;
          }"""
    )
    patched_index = patched_index.replace("</body>", data_script + "\n</body>")

    # Patch call.html
    patched_call = call_html.replace(
        """async init() {
          const callId = window.location.pathname.split('/call/')[1];
          const data = await fetch('/api/call/' + callId).then(r => r.json());
          if (!data.error) {
            this.call = { ...empty, ...data, signals: { ...empty.signals, ...data.signals }, signal_health: data.signal_health || {} };
            this.loaded = true;
          }
        },""",
        """async init() {
          const callId = new URLSearchParams(window.location.search).get('id') || window.location.pathname.split('/call/')[1];
          if (window.__CALL_DETAILS__ && callId && window.__CALL_DETAILS__[callId]) {
            const data = window.__CALL_DETAILS__[callId];
            const empty = this.call;
            this.call = { ...empty, ...data, signals: { ...empty.signals, ...data.signals }, signal_health: data.signal_health || {} };
            this.loaded = true;
          }
        },"""
    )
    patched_call = patched_call.replace("</body>", detail_script + "\n</body>")

    # --- Patch overview.html ---
    pipeline_script = f"""<script>
    window.__PIPELINE_INFO__ = {json.dumps(data.get("pipeline_info", {}))};
    </script>"""
    patched_overview = overview_html
    # Fix links for static mode
    patched_overview = patched_overview.replace('href="/dashboard?flagged=1"', 'href="dashboard.html?flagged=1"')
    patched_overview = patched_overview.replace('href="/dashboard"', 'href="dashboard.html"')
    patched_overview = patched_overview.replace('href="/"', 'href="index.html"')
    patched_overview = patched_overview.replace("'/dashboard'", "'dashboard.html'")
    patched_overview = patched_overview.replace("'/call/'", "'call.html?id='")
    patched_overview = patched_overview.replace("</body>", pipeline_script + "\n</body>")

    # Fix dashboard links for static mode
    patched_index = patched_index.replace(
        "window.location.href = '/call/' + call.call_id",
        "window.location.href = 'call.html?id=' + call.call_id"
    )
    patched_index = patched_index.replace('href="/"', 'href="index.html"')
    patched_index = patched_index.replace('href="/dashboard"', 'href="dashboard.html"')

    # Fix call detail links for static mode
    patched_call = patched_call.replace('href="/"', 'href="index.html"')
    patched_call = patched_call.replace('href="/dashboard"', 'href="dashboard.html"')

    # Fix favicon links for static mode (relative path)
    for page in [patched_overview, patched_index, patched_call]:
        pass  # handled below
    patched_overview = patched_overview.replace('href="/favicon.ico"', 'href="favicon.svg"')
    patched_index = patched_index.replace('href="/favicon.ico"', 'href="favicon.svg"')
    patched_call = patched_call.replace('href="/favicon.ico"', 'href="favicon.svg"')

    # Write output: overview → index.html (landing), dashboard → dashboard.html, call → call.html
    (dist / "index.html").write_text(patched_overview)
    (dist / "dashboard.html").write_text(patched_index)
    (dist / "call.html").write_text(patched_call)

    # Write favicon
    (dist / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23e54666">'
        '<path d="M16.5 3C19.538 3 22 5.5 22 9c0 7-7.5 11-10 12.5C9.5 20 2 16 2 9c0-3.5 2.462-6 5.5-6'
        'C9.36 3 11.048 3.695 12 4.628 12.952 3.695 14.64 3 16.5 3Z"/></svg>'
    )

    overview_size = len(patched_overview) / 1024
    dashboard_size = len(patched_index) / 1024
    call_size = len(patched_call) / 1024
    print(f"Exported to dist/")
    print(f"  index.html (overview):  {overview_size:.0f} KB")
    print(f"  dashboard.html:         {dashboard_size:.0f} KB")
    print(f"  call.html:              {call_size:.0f} KB")
    print(f"  Total:                  {(overview_size + dashboard_size + call_size):.0f} KB")


def main():
    export()

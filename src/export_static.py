"""Export dashboard as a fully static site with all data embedded.

Starts the API server, fetches all data, bakes it into the HTML templates,
and writes a self-contained static site to dist/.
"""

import json
import pickle
import re

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, MODEL_DIR, TARGET, PROJECT_ROOT
from .data_loader import load_all
from sklearn.metrics import f1_score, precision_score, recall_score


def _ensemble_predict_proba(models, scalers, subsets, X):
    """Average probabilities from ensemble models."""
    proba = np.zeros(len(X))
    n = 0
    for subset_name, cols in subsets.items():
        model = models[subset_name]
        X_sub = X[cols]
        if subset_name in scalers:
            X_sub = pd.DataFrame(
                scalers[subset_name].transform(X_sub),
                columns=cols, index=X_sub.index,
            )
        proba += model.predict_proba(X_sub)[:, 1]
        n += 1
    return proba / n


def build_api_data():
    """Build all API response data without running the server."""
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        payload = pickle.load(f)
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    models = payload["models"]
    scalers = payload.get("scalers", {})
    subsets = config["subsets"]
    threshold = config["threshold"]
    all_columns = sorted(set(c for cols in subsets.values() for c in cols))

    train, val, test = load_all()

    X_combined = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")[all_columns]
    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")[all_columns]

    X_train_split = X_combined.iloc[:len(train)].reset_index(drop=True)
    X_val_split = X_combined.iloc[len(train):len(train)+len(val)].reset_index(drop=True)

    # Build calls data for all splits
    all_calls = []
    call_details = {}

    for name, df, X in [("train", train, X_train_split), ("val", val, X_val_split), ("test", test, X_test)]:
        proba = _ensemble_predict_proba(models, scalers, subsets, X)
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

            nonzero = feat_row[feat_row != 0].abs().sort_values(ascending=False)
            top_features = [{"name": k, "value": round(float(feat_row[k]), 4)} for k in nonzero.head(20).index]

            detail = {
                **call,
                "validation_notes": str(row.get("validation_notes", "")),
                "transcript_turns": turns,
                "responses": responses,
                "signals": {
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
                    },
                    "embeddings": {
                        "vn_positive_similarity": round(float(feat_row.get("emb_vn_positive_similarity", 0)), 4),
                    },
                },
                "top_features": top_features,
            }
            call_details[row["call_id"]] = detail

    # Val metrics
    y_val = val[TARGET].astype(int).values
    val_proba = _ensemble_predict_proba(models, scalers, subsets, X_val_split)
    val_pred = (val_proba >= threshold).astype(int)

    # Feature importance from first LightGBM model
    importance_list = []
    for subset_name in ["text_light", "embedding"]:
        model = models.get(subset_name)
        if hasattr(model, "feature_importances_"):
            cols = subsets[subset_name]
            imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
            importance_list = [{"feature": k, "importance": round(float(v), 4)} for k, v in imp.head(30).items()]
            break

    # Outcome breakdown
    outcome_counts = {}
    for c in all_calls:
        o = c["outcome"]
        if o not in outcome_counts:
            outcome_counts[o] = {"total": 0, "flagged": 0}
        outcome_counts[o]["total"] += 1
        if c["predicted_ticket"]:
            outcome_counts[o]["flagged"] += 1

    flagged = [c for c in all_calls if c["predicted_ticket"]]

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
        "model": config["best_model"],
        "feature_count": len(all_columns),
        "outcome_breakdown": outcome_counts,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }

    all_calls.sort(key=lambda c: c["probability"], reverse=True)

    return {
        "stats": stats,
        "calls": all_calls,
        "importance": importance_list,
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
    index_html = (static_dir / "index.html").read_text()
    call_html = (static_dir / "call.html").read_text()

    # Embed data into index.html
    data_script = f"""<script>
    window.__STATIC_DATA__ = {json.dumps({"stats": data["stats"], "calls": data["calls"], "importance": data["importance"]})};
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
            this.importance = window.__STATIC_DATA__.importance;
            this.buildHistogram(this.calls);
            return;
          }"""
    )
    # Patch loadCalls to filter from static data
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
    # Inject data script before </body>
    patched_index = patched_index.replace("</body>", data_script + "\n</body>")

    # Patch call.html: replace entire init with static lookup
    patched_call = call_html.replace(
        """async init() {
          const callId = window.location.pathname.split('/call/')[1];
          const data = await fetch('/api/call/' + callId).then(r => r.json());
          if (!data.error) {
            this.call = { ...empty, ...data, signals: { ...empty.signals, ...data.signals } };
            this.loaded = true;
          }
        },""",
        """async init() {
          const callId = new URLSearchParams(window.location.search).get('id') || window.location.pathname.split('/call/')[1];
          if (window.__CALL_DETAILS__ && callId && window.__CALL_DETAILS__[callId]) {
            const data = window.__CALL_DETAILS__[callId];
            const empty = this.call;
            this.call = { ...empty, ...data, signals: { ...empty.signals, ...data.signals } };
            this.loaded = true;
          }
        },"""
    )
    patched_call = patched_call.replace("</body>", detail_script + "\n</body>")

    # For static hosting, call detail uses ?id= param instead of /call/{id} path
    patched_index = patched_index.replace(
        "window.location.href = '/call/' + call.call_id",
        "window.location.href = 'call.html?id=' + call.call_id"
    )
    # Back link
    patched_call = patched_call.replace('href="/"', 'href="index.html"')

    (dist / "index.html").write_text(patched_index)
    (dist / "call.html").write_text(patched_call)

    # Size report
    index_size = len(patched_index) / 1024
    call_size = len(patched_call) / 1024
    print(f"Exported to dist/")
    print(f"  index.html: {index_size:.0f} KB")
    print(f"  call.html:  {call_size:.0f} KB")
    print(f"  Total:      {(index_size + call_size):.0f} KB")


def main():
    export()

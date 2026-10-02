"""Upstream typed-decisions benchmark evaluation, ported from the Kaggle notebook.

Source: notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
  cell 12  -> predict()   (test-split inference with laya.Agent)
  cell 14  -> score()     (metric definitions, unchanged)
  cell 18  -> report()    (metrics.json shape + per-workflow accuracy)

Metric definitions are copied as-is so results compare with the published
table. The notebook computes one set of metrics, using whatever temperatures
the checkpoint's rl_agent_config.json carries (fitted ones for a fine-tuned
checkpoint); there is no separate raw-logit pass, so none is added here.

The data is the upstream LocalLLaMA/typed-decisions test split, not this
project's RVL-CDIP test set.

    python scripts/eval_upstream.py --model-dir ./laya_base --out metrics_base.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

DATASET_ID = "LocalLLaMA/typed-decisions"
WORKFLOWS = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]


def predict(model_dir, device, limit=None):
    import laya
    from datasets import load_dataset

    print("Loading test split for evaluation...")
    ds_test = load_dataset(DATASET_ID, "all", split="test")
    if limit is not None:
        ds_test = ds_test.select(range(min(limit, len(ds_test))))

    agent = laya.Agent(model_dir, device=device)

    predictions = []
    latencies_ms = []
    print(f"Evaluating {len(ds_test)} test cases on {device}...")
    t0_eval = time.time()
    for row in ds_test:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])

        t0 = time.perf_counter()
        res = agent.predict(state, questions)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt_ms)

        predictions.append({
            "id": row["id"],
            "workflow": row["workflow"],
            "pred": res["answers"],
            "gold": gold,
            "questions": questions,
            "latency_ms": dt_ms,
        })
    print(f"Evaluated all {len(ds_test)} cases in {time.time() - t0_eval:.1f}s!")
    return predictions, latencies_ms


def score(predictions, latencies_ms):
    from laya.common import ece_score

    accuracies = []
    soft_accuracies = []
    brier_scores = []
    kl_divs = []
    tv_distances = []
    score_maes = []
    within_one = []
    all_confs = []
    all_corrects = []

    for item in predictions:
        pred_answers = item["pred"]
        gold_answers = item["gold"]
        questions = item["questions"]

        for qid, qdef in questions.items():
            p_ans = pred_answers[qid]
            g_ans = gold_answers[qid]
            q_type = qdef["type"]

            if q_type == "choice":
                keys = list(qdef["criteria"].keys())
                pred_choice = p_ans["choice"]
                gold_label = str(g_ans["label"])

                is_corr = float(pred_choice == gold_label)
                accuracies.append(is_corr)
                all_corrects.append(is_corr)

                p_probs = np.array([p_ans["probabilities"].get(k, 1e-6) for k in keys])
                g_probs = np.array([g_ans["probabilities"].get(k, 1e-6) for k in keys])
                p_probs /= p_probs.sum()
                g_probs /= g_probs.sum()

                all_confs.append(float(p_probs.max()))
                soft_accuracies.append(float((p_probs * g_probs).sum()))
                brier_scores.append(float(((p_probs - g_probs) ** 2).sum()))
                tv_distances.append(float(0.5 * np.abs(p_probs - g_probs).sum()))
                kl_divs.append(float((g_probs * np.log(np.clip(g_probs / p_probs, 1e-12, 1e4))).sum()))

            elif q_type == "noul":
                p_val = p_ans["noul"]
                g_val = g_ans.get("noul", g_ans.get("probabilities", {}).get("true", 0.5))
                gold_label = str(g_ans["label"]).lower()

                pred_label = "true" if p_val >= 0.5 else "false"
                is_corr = float(pred_label == gold_label)
                accuracies.append(is_corr)
                all_corrects.append(is_corr)
                all_confs.append(float(max(p_val, 1.0 - p_val)))

                p_dist = np.array([1.0 - p_val, p_val])
                g_dist = np.array([1.0 - g_val, g_val])

                soft_accuracies.append(float((p_dist * g_dist).sum()))
                brier_scores.append(float(((p_dist - g_dist) ** 2).sum()))
                tv_distances.append(float(0.5 * np.abs(p_dist - g_dist).sum()))
                kl_divs.append(float((g_dist * np.log(np.clip(g_dist / p_dist, 1e-12, 1e4))).sum()))

            elif q_type == "score":
                p_score = p_ans["score"]
                g_score = g_ans.get("score", 0.0)
                score_maes.append(abs(p_score - g_score))
                within_one.append(float(abs(p_score - g_score) <= 1.0))

                # Use argmax of probabilities for discrete prediction
                n_levels = len(qdef.get("criteria", []))
                p_probs_score = np.array([p_ans["probabilities"].get(str(i), 0.0) for i in range(n_levels)])
                if p_probs_score.sum() > 0:
                    p_probs_score /= p_probs_score.sum()
                    p_lvl = int(np.argmax(p_probs_score))
                    all_confs.append(float(p_probs_score.max()))
                else:
                    p_lvl = int(round(p_score))
                    all_confs.append(0.5)

                g_lvl = int(g_ans.get("label", int(round(g_score))))
                is_corr = float(p_lvl == g_lvl)
                accuracies.append(is_corr)
                all_corrects.append(is_corr)

    return {
        "accuracy": round(float(np.mean(accuracies)), 4),
        "soft_accuracy": round(float(np.mean(soft_accuracies)), 4),
        "brier_score": round(float(np.mean(brier_scores)), 4),
        "ece": round(float(ece_score(np.array(all_confs), np.array(all_corrects))), 4),
        "score_mae": round(float(np.mean(score_maes) if score_maes else 0.0), 4),
        "within_1_level": round(float(np.mean(within_one) if within_one else 0.0), 4),
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "kl_divergence": round(float(np.mean(kl_divs)), 4),
        "total_variation": round(float(np.mean(tv_distances)), 4),
    }


def per_workflow(predictions):
    out = {}
    for wf in WORKFLOWS:
        wf_items = [item for item in predictions if item["workflow"] == wf]
        wf_corr = []
        for item in wf_items:
            for qid, qdef in item["questions"].items():
                p_ans = item["pred"][qid]
                g_ans = item["gold"][qid]
                if qdef["type"] == "choice":
                    wf_corr.append(float(p_ans["choice"] == str(g_ans["label"])))
                elif qdef["type"] == "noul":
                    pred_label = "true" if p_ans["noul"] >= 0.5 else "false"
                    wf_corr.append(float(pred_label == str(g_ans["label"]).lower()))
                elif qdef["type"] == "score":
                    p_probs = [p_ans["probabilities"].get(str(i), 0.0) for i in range(len(qdef.get("criteria", [])))]
                    p_lvl = int(np.argmax(p_probs)) if sum(p_probs) > 0 else int(round(p_ans["score"]))
                    g_lvl = int(g_ans.get("label", int(round(g_ans.get("score", 0.0)))))
                    wf_corr.append(float(p_lvl == g_lvl))
        if wf_corr:
            out[wf] = {"n_decisions": len(wf_corr), "accuracy": round(float(np.mean(wf_corr)), 4)}
    return out


def main():
    parser = argparse.ArgumentParser(description="Upstream typed-decisions benchmark evaluation")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True, help="path of the metrics JSON to write")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases (dry run)")
    args = parser.parse_args()

    predictions, latencies_ms = predict(args.model_dir, args.device, args.limit)
    report = {
        "benchmark": DATASET_ID,
        "model_dir": str(args.model_dir),
        "n_cases": len(predictions),
        "n_decisions": sum(len(item["questions"]) for item in predictions),
        "limit": args.limit,
        "metrics": score(predictions, latencies_ms),
        "per_workflow": per_workflow(predictions),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics to {out}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

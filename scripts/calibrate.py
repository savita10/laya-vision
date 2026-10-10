"""Phase 4: fit one temperature per view on the val monitor pool.

Scores the val images OUTSIDE the fixed 2,000-image comparison set (the pool
train_vision.py monitored on; the comparison ids and test are never touched)
with a train_vision.py checkpoint, for each view (combined / image / text),
and fits one scalar temperature per view by NLL with upstream's fitter
(laya.calibrate.fit_one_temperature: LBFGS on log T, clamped to
[TEMP_MIN, TEMP_MAX] = [0.5, 5.0]).

Per view, before (T = 1) and after: accuracy, NLL, Brier, ECE two ways
  ece_15bin                    vision.metrics: laya.common.ece_score on max(p)
  ece_15bin_upstream_harness   as scripts/eval_upstream.py computes it from
                               laya.Agent answers (4-decimal rounded probabilities)
"after" is measured on the pool the temperature was fitted on, so it is
optimistic; "after_crossfit" fits on one half of the pool and scores the
other (both ways, pooled) as an out-of-sample estimate. The held-out number
that is reported is eval_vision.py --temperatures on the comparison set.

Outputs, in --out-dir:
  calibration.json           temperatures (eval_vision.py / final_eval.py --temperatures),
                             metrics before/after/after_crossfit, reliability bins, provenance
  reliability_<view>.png     15 bins, before vs after
  calibration_logits.npz     the pool's raw logits per view, ids and labels (refits need no GPU)

    python scripts/calibrate.py --base-ckpt <drive>/step5/laya_finetuned --run-dir <drive>/phase3/runs/run02 \\
        --shards-root <drive>/phase2/shards --records <drive>/data/rvl_v1/val.jsonl --out-dir <drive>/phase3/calibration
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

import plots
import vision
from route_a import DEFAULT_COMPARISON_SET, load_comparison_set

KEEP = ("accuracy", "nll", "brier", "ece_15bin", "ece_15bin_upstream_harness", "mean_confidence")
CROSSFIT_SEED = 20261010


def parse_args():
    p = argparse.ArgumentParser(description="Fit per-view temperatures on the val monitor pool")
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--run-dir", required=True, help="train_vision.py --out-dir (run02)")
    p.add_argument("--checkpoint", default="final", help="final, latest, or epoch_N inside --run-dir")
    p.add_argument("--shards-root", required=True)
    p.add_argument("--records", required=True, help="val.jsonl")
    p.add_argument("--comparison-set", default=str(DEFAULT_COMPARISON_SET))
    p.add_argument("--views", default="combined,image,text")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="dry run: first N pool images")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def fit_temperature(logits, labels):
    from laya.calibrate import MIN_TYPE_N, fit_one_temperature

    onehot = np.eye(logits.shape[1], dtype=np.float32)[labels]
    return fit_one_temperature(list(zip(logits, onehot)), min_n=MIN_TYPE_N)


def summary(probs, labels):
    m = vision.metrics(probs, labels)
    return {k: m[k] for k in KEEP}


def crossfit(logits, labels, seed=CROSSFIT_SEED):
    """Fit on one half, score the other, both ways; metrics on the pooled held-out probabilities."""
    order = np.random.RandomState(seed).permutation(len(labels))
    halves = [order[: len(order) // 2], order[len(order) // 2:]]
    probs = np.zeros_like(logits)
    temps = []
    for fit_rows, score_rows in (halves, halves[::-1]):
        t = fit_temperature(logits[fit_rows], labels[fit_rows])
        temps.append(t)
        probs[score_rows] = vision.softmax_t(logits[score_rows], t)
    return {**summary(probs, labels), "temperatures": temps, "seed": seed}


def main():
    args = parse_args()
    vision.check_laya_version()
    from laya.common import TEMP_MAX, TEMP_MIN

    views = args.views.split(",")
    if any(v not in vision.MODALITIES for v in views):
        raise SystemExit(f"--views must be from {vision.MODALITIES}")
    device = torch.device(args.device)
    amp = None if args.amp == "none" else args.amp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = Path(args.records)
    records = vision.read_records(records_path, "val")
    comparison, _ = load_comparison_set(Path(args.comparison_set), records_path, records)
    comparison_ids = set(comparison["ids"])

    model, tok, cfg, base_sha, run_info, max_len, tokens = vision.load_run(args.base_ckpt, args.run_dir,
                                                                            args.checkpoint)
    model.to(device).eval()
    builder = vision.SequenceBuilder(tok, vision.doc_type_question(records), max_len, cfg.get("head_max_len", 256),
                                     tokens)
    data, prov = vision.load_split_data(args.shards_root, records_path, "val", builder, args.cache_dir, tokens)
    pool = [r for r, i in enumerate(data.ids) if i not in comparison_ids]
    if args.limit is not None:
        pool = pool[:args.limit]
    data = data.subset(pool)
    assert not comparison_ids & set(data.ids)
    diff = vision.equivalence_check(model, data, builder, device, amp)
    print(f"equivalence check passed (max |diff| {diff:.2e}); monitor pool {len(data)} val images "
          f"(comparison set excluded)", flush=True)

    labels = data.labels
    result = {
        "complete": args.limit is None,
        "fitted_on": {"split": "val", "pool": "val outside the comparison set", "n": len(data),
                      "ids_sha256": hashlib.sha256("\n".join(data.ids).encode()).hexdigest(), "data": prov},
        "checkpoint": {**{k: v for k, v in run_info.items() if k != "train_identity"}, "visual_tokens": tokens,
                       "max_len": max_len, "base_model_sha256": base_sha},
        "fitter": f"laya.calibrate.fit_one_temperature (laya {vision.LAYA_VERSION}): NLL, LBFGS on log T, "
                  f"clamped to [{TEMP_MIN}, {TEMP_MAX}]",
        "ece": {"bins": 15, "ece_15bin": "laya.common.ece_score(max p, correct)",
                "ece_15bin_upstream_harness": "scripts/eval_upstream.py on laya.Agent output: probabilities "
                                              "rounded to 4 dp, renormalised, then the same ece_score"},
        "amp": args.amp, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "script_git_hash": vision.script_git_hash(), "date": time.strftime("%Y-%m-%d"),
        "temperatures": {}, "views": {},
    }
    saved = {"ids": np.array(data.ids), "labels": labels}
    for view in views:
        logits, _, _, truncated = vision.predict_logits(model, data, builder, view, device, args.batch_size, amp)
        saved[view] = logits
        t = fit_temperature(logits, labels)
        before, after = vision.softmax_t(logits), vision.softmax_t(logits, t)
        v = {
            "temperature": t,
            "temperature_at_clamp_bound": t in (TEMP_MIN, TEMP_MAX),
            "before": summary(before, labels),
            "after": summary(after, labels),
            "after_crossfit": crossfit(logits, labels),
            "reliability_before": vision.reliability_bins(before, labels),
            "reliability_after": vision.reliability_bins(after, labels),
            "text_truncated": truncated,
        }
        result["temperatures"][view] = t
        result["views"][view] = v
        plots.reliability_diagram(out_dir / f"reliability_{view}.png",
                                  [("before (T = 1)", v["reliability_before"], v["before"]["ece_15bin"]),
                                   (f"after (T = {t:.3f})", v["reliability_after"], v["after"]["ece_15bin"])],
                                  f"Laya-Vision {view} view, val monitor pool (n={len(data)})")
        b, a, c = v["before"], v["after"], v["after_crossfit"]
        print(f"{view:9s} T {t:.3f}{' (at clamp bound)' if v['temperature_at_clamp_bound'] else ''} | acc "
              f"{b['accuracy']:.4f} | NLL {b['nll']:.3f} -> {a['nll']:.3f} | Brier {b['brier']:.4f} -> "
              f"{a['brier']:.4f} | ECE {b['ece_15bin']:.4f} -> {a['ece_15bin']:.4f} (cross-fit "
              f"{c['ece_15bin']:.4f}) | harness ECE {b['ece_15bin_upstream_harness']:.4f} -> "
              f"{a['ece_15bin_upstream_harness']:.4f}", flush=True)
    np.savez_compressed(out_dir / "calibration_logits.npz", **saved)
    vision.write_json_atomic(out_dir / "calibration.json", result)
    print(f"Wrote {out_dir / 'calibration.json'}, reliability_<view>.png and calibration_logits.npz")


if __name__ == "__main__":
    main()

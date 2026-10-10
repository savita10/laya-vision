"""Evaluate Laya-Vision on the fixed 2,000-image val comparison set, three views.

Views (no modality dropout at evaluation):
  combined  [visual | OCR text]
  image     [visual] only
  text      [OCR text] only; the sequence is exactly upstream Laya's for that text
Per view: accuracy, NLL, Brier, 15-bin ECE two ways (laya.common.ece_score on
max p, and as scripts/eval_upstream.py computes it from laya.Agent output),
per-class and per-neighbourhood accuracy, a 16x16 confusion matrix (in the
results JSON, rows = true class, plus <out>.confusion_<view>.png), text
truncation count, and Laya-Vision forward cost (GPU-seconds/image at
--batch-size, plus batch-1 latency p50/p95).

--temperatures calibration.json (scripts/calibrate.py, fitted on the val
monitor pool, never on these ids) adds a "calibrated" block per view:
softmax(logits / T_view). Raw metrics stay at the top level of each view.
A scalar T > 0 does not change the argmax, so accuracy, per-class accuracy
and the confusion matrix are the same raw or calibrated.

The comparison set is splits/val_comparison_2000_seed11.json; its val.jsonl
sha256 is checked as in route_a.py. Val only: test is never loaded (shards.py
refuses it and --records must be val.jsonl).

--run-dir omitted evaluates the Gate-0 base with no training: only the text
view is meaningful then (zero-shot Laya on OCR text), so only it is allowed.

    python scripts/eval_vision.py --base-ckpt <drive>/step5/laya_finetuned --run-dir <drive>/phase3/runs/r1 \\
        --shards-root <drive>/phase2/shards --records <drive>/data/rvl_v1/val.jsonl --out <drive>/phase3/eval_r1.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import plots
import rvlcdip
import vision
from route_a import DEFAULT_COMPARISON_SET, load_comparison_set

REPO = Path(__file__).resolve().parent.parent
ROUTE_A_RESULTS = REPO / "docs" / "route_a_results.json"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Laya-Vision on the val comparison set")
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--run-dir", default=None, help="train_vision.py --out-dir; omit for the untrained base")
    p.add_argument("--checkpoint", default="final", help="final, latest, or epoch_N inside --run-dir")
    p.add_argument("--temperatures", default=None, help="calibration.json from calibrate.py (same checkpoint)")
    p.add_argument("--shards-root", required=True)
    p.add_argument("--records", required=True, help="val.jsonl")
    p.add_argument("--comparison-set", default=str(DEFAULT_COMPARISON_SET))
    p.add_argument("--views", default="combined,image,text")
    p.add_argument("--out", required=True, help="results JSON; predictions go next to it (.predictions.jsonl)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="dry run: first N comparison images")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latency-n", type=int, default=200, help="batch-1 forwards timed per view")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    vision.check_laya_version()
    views = args.views.split(",")
    if any(v not in vision.MODALITIES for v in views):
        raise SystemExit(f"--views must be from {vision.MODALITIES}")
    if args.run_dir is None and views != ["text"]:
        raise SystemExit("Without --run-dir the projector is untrained; use --views text for the base model.")
    device = torch.device(args.device)
    amp = None if args.amp == "none" else args.amp

    records_path = Path(args.records)
    records = vision.read_records(records_path, "val")
    comparison, selected = load_comparison_set(Path(args.comparison_set), records_path, records)

    model, tok, cfg, base_sha, run_info, builder_max_len, tokens = vision.load_run(args.base_ckpt, args.run_dir,
                                                                                    args.checkpoint)
    temperatures, calibration = {}, None
    if args.temperatures is not None:
        if args.run_dir is None:
            raise SystemExit("--temperatures needs --run-dir (they were fitted on a trained checkpoint).")
        temperatures, cal = vision.load_temperatures(args.temperatures, run_info["trainable_sha256"], tokens)
        missing = [v for v in views if v not in temperatures]
        if missing:
            raise SystemExit(f"{args.temperatures} has no temperature for views {missing}.")
        calibration = {"path": str(args.temperatures), "sha256": vision.sha256_file(args.temperatures),
                       "temperatures": temperatures, "fitted_on": cal["fitted_on"]["pool"],
                       "fitted_n": cal["fitted_on"]["n"]}
    model.to(device).eval()

    question = vision.doc_type_question(records)
    builder = vision.SequenceBuilder(tok, question, builder_max_len, cfg.get("head_max_len", 256), tokens)
    data, prov = vision.load_split_data(args.shards_root, records_path, "val", builder,
                                        args.cache_dir, tokens)
    rows = [data.index[r["id"]] for r in selected if r["id"] in data.index]
    if len(rows) != len(selected):
        raise SystemExit(f"{len(selected) - len(rows)} comparison images have no shard row (decode errors?).")
    if args.limit is not None:
        rows = rows[:args.limit]
    data = data.subset(rows)
    diff = vision.equivalence_check(model, data, builder, device, amp)
    print(f"equivalence check passed (max |diff| {diff:.2e}); {len(data)} comparison images", flush=True)

    majority = max(comparison["counts_per_class"].values()) / comparison["n"]
    results = {"complete": args.limit is None, "n": len(data), "views": {}, "run": run_info,
               "base_model_sha256": base_sha, "data": prov, "max_len": builder_max_len,
               "text_budget": {"combined": builder.text_budget(True), "text": builder.text_budget(False)},
               "visual_tokens": tokens, "amp": args.amp,
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "script_git_hash": vision.script_git_hash(), "date": time.strftime("%Y-%m-%d"),
               "majority_class_floor": majority, "classes": rvlcdip.CLASSES,
               "calibration": calibration or "none (raw probabilities)"}
    if ROUTE_A_RESULTS.exists():
        with open(ROUTE_A_RESULTS, encoding="utf-8") as f:
            ra = json.load(f)
        results["route_a"] = {"accuracy": ra["accuracy"], "gpu_seconds_per_image": ra["gpu_seconds_per_image"]["total"]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_path = out.with_suffix(".predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8", newline="\n") as pf:
        for view in views:
            logits, prow, seconds, truncated = vision.predict_logits(model, data, builder, view, device,
                                                                     args.batch_size, amp)
            labels = data.labels[prow]
            probs = vision.softmax_t(logits)
            m = vision.metrics(probs, labels)
            m["reliability"] = vision.reliability_bins(probs, labels)
            cm = vision.confusion_matrix(probs, labels)
            m["confusion_matrix"] = cm.tolist()
            if view in temperatures:
                t = temperatures[view]
                cal_probs = vision.softmax_t(logits, t)
                m["calibrated"] = {"temperature": t, **vision.metrics(cal_probs, labels),
                                   "reliability": vision.reliability_bins(cal_probs, labels)}
            m["text_truncated"] = truncated
            m["gpu_seconds_per_image_batched"] = seconds / len(prow)
            m["latency_batch1"] = vision.batch1_latency(model, data, builder, view, device, amp, args.latency_n)
            results["views"][view] = m
            plots.confusion_png(out.with_name(f"{out.stem}.confusion_{view}.png"), cm, rvlcdip.CLASSES,
                                f"Laya-Vision {view} view, val comparison set (n={len(prow)}, acc "
                                f"{m['accuracy']:.3f})")
            for r, z, p in zip(prow, logits, probs):
                pf.write(json.dumps({"id": data.ids[r], "view": view, "label": rvlcdip.CLASSES[data.labels[r]],
                                     "pred": rvlcdip.CLASSES[int(p.argmax())],
                                     "probs": [round(float(x), 6) for x in p],
                                     "logits": [round(float(x), 4) for x in z]}) + "\n")
            cal_txt = (f"  calibrated (T {m['calibrated']['temperature']:.3f}): nll {m['calibrated']['nll']:.3f} "
                       f"ece {m['calibrated']['ece_15bin']:.3f}" if "calibrated" in m else "")
            print(f"{view:9s} acc {m['accuracy']:.4f}  nll {m['nll']:.3f}  ece {m['ece_15bin']:.3f} "
                  f"(harness {m['ece_15bin_upstream_harness']:.3f}){cal_txt}  truncated {truncated}  "
                  f"{1000 * m['gpu_seconds_per_image_batched']:.1f} ms/img batched, "
                  f"p50 {m['latency_batch1']['p50_ms']:.0f} ms batch-1", flush=True)
    vision.write_json_atomic(out, results)

    print(f"\nmajority-class floor {majority:.4f}" + (f"; Route A {results['route_a']['accuracy']:.4f} at "
                                                     f"{results['route_a']['gpu_seconds_per_image']:.1f} GPU-s/img"
                                                     if "route_a" in results else ""))
    print("Laya-Vision cost excludes precompute (SigLIP + Tesseract, Phase 2; DECISIONS.md 2026-10-08).")
    print(f"Wrote {out}, {pred_path} and {out.stem}.confusion_<view>.png")


if __name__ == "__main__":
    main()

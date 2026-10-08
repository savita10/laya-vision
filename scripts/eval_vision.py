"""Evaluate Laya-Vision on the fixed 2,000-image val comparison set, three views.

Views (no modality dropout at evaluation):
  combined  [visual | OCR text]
  image     [visual] only
  text      [OCR text] only; the sequence is exactly upstream Laya's for that text
Per view: accuracy, NLL, Brier, 15-bin ECE (laya.common.ece_score), per-class
and per-neighbourhood accuracy, text truncation count, and Laya-Vision forward
cost (GPU-seconds/image at --batch-size, plus batch-1 latency p50/p95). Raw
probabilities: temperature fitting is Phase 4.

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
import statistics
import time
from pathlib import Path

import numpy as np
import torch

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


def batch1_latency(model, data, builder, view, device, amp, n):
    times = []
    for row in range(min(n, len(data))):
        _, _, seconds, _ = vision.predict(model, data, builder, view, device, 1, amp, rows=[row])
        times.append(seconds)
    times = times[1:] or times  # first call warms up kernels
    q = statistics.quantiles(times, n=20) if len(times) >= 2 else times * 19
    return {"n": len(times), "p50_ms": 1000 * statistics.median(times), "p95_ms": 1000 * q[18]}


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

    model, tok, cfg, base_sha = vision.load_base(args.base_ckpt)
    run_info = {"run_dir": None, "checkpoint": None}
    builder_max_len = vision.MAX_LEN
    tokens = vision.N_VISUAL
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        with open(run_dir / "run_config.json", encoding="utf-8") as f:
            run_cfg = json.load(f)
        ident = run_cfg["identity"]
        if ident["base_model_sha256"] != base_sha:
            raise SystemExit("--base-ckpt is not the checkpoint this run was trained from (sha256 differs).")
        builder_max_len, tokens = ident["max_len"], ident["tokens"]
        ckpt = run_dir / args.checkpoint / "trainable.safetensors"
        vision.load_trainable(model, ckpt)
        run_info = {"run_dir": str(run_dir), "checkpoint": args.checkpoint,
                    "trainable_sha256": vision.sha256_file(ckpt), "train_identity": ident,
                    "train_git_hash": run_cfg.get("script_git_hash")}
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
               "majority_class_floor": majority, "calibration": "none (raw probabilities; temperature is P4)"}
    if ROUTE_A_RESULTS.exists():
        with open(ROUTE_A_RESULTS, encoding="utf-8") as f:
            ra = json.load(f)
        results["route_a"] = {"accuracy": ra["accuracy"], "gpu_seconds_per_image": ra["gpu_seconds_per_image"]["total"]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_path = out.with_suffix(".predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8", newline="\n") as pf:
        for view in views:
            probs, prow, seconds, truncated = vision.predict(model, data, builder, view, device, args.batch_size, amp)
            m = vision.metrics(probs, data.labels[prow])
            m["text_truncated"] = truncated
            m["gpu_seconds_per_image_batched"] = seconds / len(prow)
            m["latency_batch1"] = batch1_latency(model, data, builder, view, device, amp, args.latency_n)
            results["views"][view] = m
            for r, p in zip(prow, probs):
                pf.write(json.dumps({"id": data.ids[r], "view": view, "label": rvlcdip.CLASSES[data.labels[r]],
                                     "pred": rvlcdip.CLASSES[int(p.argmax())],
                                     "probs": [round(float(x), 6) for x in p]}) + "\n")
            print(f"{view:9s} acc {m['accuracy']:.4f}  nll {m['nll']:.3f}  ece {m['ece_15bin']:.3f}  "
                  f"truncated {truncated}  {1000 * m['gpu_seconds_per_image_batched']:.1f} ms/img batched, "
                  f"p50 {m['latency_batch1']['p50_ms']:.0f} ms batch-1", flush=True)
    vision.write_json_atomic(out, results)

    print(f"\nmajority-class floor {majority:.4f}" + (f"; Route A {results['route_a']['accuracy']:.4f} at "
                                                     f"{results['route_a']['gpu_seconds_per_image']:.1f} GPU-s/img"
                                                     if "route_a" in results else ""))
    print("Laya-Vision cost excludes precompute (SigLIP + Tesseract, Phase 2; DECISIONS.md 2026-10-08).")
    print(f"Wrote {out} and {pred_path}")


if __name__ == "__main__":
    main()

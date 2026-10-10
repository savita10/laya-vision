"""Phase 4 final evaluation: the ONLY script permitted to read test labels (CLAUDE.md).

Protocol: docs/phase4_protocol.md, written and committed before any test run.
One touch: the run writes <out-dir>/final_results.json and refuses to start if
that file, or a started marker from an earlier attempt, exists. --force needs
--reason; the reason, and the sha256 of whatever it supersedes, are written
into the new output, and the earlier file is kept (renamed), never overwritten.

Scores run02_final (--run-dir, --checkpoint) on the full test split, three
views, raw and calibrated (--temperatures from calibrate.py, fitted on the
val monitor pool; the text view at T = 1.0, vision.UNCALIBRATED_VIEWS). Per view: accuracy, NLL, Brier, ECE (ours and the upstream
harness's), mean confidence, per-class and per-neighbourhood accuracy, 16x16
confusion matrix (JSON + PNG), reliability bins, and cost: GPU-seconds/image
batched, batched latency p50/p95 (per batch and per image) and batch-1
latency p50/p95. Laya-Vision cost excludes precompute (SigLIP + Tesseract).

Route A comparison: the same metrics on the pre-registered 1,000-image
stratified test subsample (seed 17), splits/test_comparison_1000_seed17.json,
drawn once by route_a.py --make-comparison-set (recorded exception,
DECISIONS.md 2026-10-10). Route A's test outputs (--route-a-outputs, from
route_a.py's one test run) are scored here with the same metric code, and
their label fields must agree with test.jsonl.

--dry-run runs the whole pipeline on the val comparison set instead (records
must be val.jsonl; test is not read; --limit allowed), writing to
<out-dir>/dry_run/. It never counts as the touch.

    python scripts/final_eval.py --dry-run --records <drive>/data/rvl_v1/val.jsonl --limit 200 ...
    python scripts/final_eval.py --base-ckpt <drive>/step5/laya_finetuned --run-dir <drive>/phase3/runs/run02 \\
        --temperatures <drive>/phase4/calibration/calibration.json --shards-root <drive>/phase2/shards \\
        --records <drive>/data/rvl_v1/test.jsonl --route-a-outputs <drive>/phase4/route_a_test/route_a_outputs.jsonl \\
        --out-dir <drive>/phase4/final
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import plots
import rvlcdip
import shards
import vision
from route_a import DEFAULT_COMPARISON_SET, TEST_COMPARISON_N, TEST_COMPARISON_SEED, TEST_COMPARISON_SET, \
    load_comparison_set

REPO = Path(__file__).resolve().parent.parent
PROTOCOL = REPO / "docs" / "phase4_protocol.md"
SUBSAMPLE_SEED = TEST_COMPARISON_SEED
SUBSAMPLE_N = TEST_COMPARISON_N


def parse_args():
    p = argparse.ArgumentParser(description="Final, one-touch test evaluation of Laya-Vision")
    p.add_argument("--base-ckpt")
    p.add_argument("--run-dir")
    p.add_argument("--checkpoint", default="final")
    p.add_argument("--temperatures", help="calibration.json from calibrate.py")
    p.add_argument("--views", default="combined,image,text")
    p.add_argument("--shards-root")
    p.add_argument("--records", required=True, help="test.jsonl (val.jsonl with --dry-run)")
    p.add_argument("--route-a-subsample", default=str(TEST_COMPARISON_SET))
    p.add_argument("--route-a-outputs", default=None, help="Route A route_a_outputs.jsonl on the test subsample")
    p.add_argument("--out-dir")
    p.add_argument("--force", action="store_true", help="run again although a result or started marker exists")
    p.add_argument("--reason", default=None, help="required with --force; written into the output")
    p.add_argument("--dry-run", action="store_true", help="whole pipeline on the val comparison set; test untouched")
    p.add_argument("--comparison-set", default=str(DEFAULT_COMPARISON_SET))
    p.add_argument("--limit", type=int, default=None, help="--dry-run only: first N images")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latency-n", type=int, default=200)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_subsample(path, records_path):
    """The pre-registered test subsample (route_a.py --make-comparison-set); checked by file hash only."""
    with open(path, encoding="utf-8") as f:
        sub = json.load(f)
    if sub.get("split") != "test" or sub["seed"] != SUBSAMPLE_SEED or sub["n"] != SUBSAMPLE_N:
        raise SystemExit(f"{path} is not the pre-registered test subsample ({SUBSAMPLE_N} test images, "
                         f"seed {SUBSAMPLE_SEED}).")
    if sub["source_sha256"] != vision.sha256_file(records_path):
        raise SystemExit(f"{records_path} is not the test.jsonl the Route A subsample was drawn from.")
    return sub


def one_touch_guard(out_dir, force, reason):
    """Refuse a second touch. Returns provenance of what a forced run supersedes."""
    result, marker = out_dir / "final_results.json", out_dir / "final_eval.started.json"
    previous = [p for p in (result, marker) if p.exists()]
    if not previous:
        return None
    if not force:
        raise SystemExit(f"{previous[0]} exists: the test set has been touched. final_eval.py runs once; "
                         "re-running needs --force --reason '<why>', and the reason is recorded in the output.")
    if not reason or not reason.strip():
        raise SystemExit("--force needs a non-empty --reason.")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    superseded = []
    for p in previous:
        kept = p.with_name(f"{p.stem}.superseded_{stamp}{p.suffix}")
        superseded.append({"file": p.name, "kept_as": kept.name, "sha256": vision.sha256_file(p)})
        p.rename(kept)
    return {"reason": reason.strip(), "superseded": superseded}


def load_split_for_eval(shards_root, split, records, builder, cache_dir, tokens, allow_test):
    """VisionData for the given records (shards + OCR + labels). Records whose image failed
    to decode have no shard row; they are returned separately as unscorable ids."""
    split_dir = Path(shards_root) / split
    emb_cfg, ids, emb = shards.load_split(split_dir, tokens=tokens, allow_test=allow_test, dtype=np.float16)
    ocr_cfg, texts = shards.load_ocr(split_dir, allow_test=allow_test)
    row_of = {i: n for n, i in enumerate(ids)}
    wanted = [r for r in records if r["id"] in row_of]
    unscorable = [r["id"] for r in records if r["id"] not in row_of]
    sel = [row_of[r["id"]] for r in wanted]
    ocr_ids, lengths = vision.tokenize_ocr(builder, [texts.get(r["id"], "") for r in wanted], cap=builder.max_len)
    data = vision.VisionData([r["id"] for r in wanted], emb[sel], ocr_ids, lengths,
                             [rvlcdip.CLASSES.index(r["label_name"]) for r in wanted])
    prov = {"split": split, "n_records": len(records), "n_scored": len(wanted), "unscorable_ids": unscorable,
            "shards_identity": emb_cfg["identity"], "embed_pass": emb_cfg["passes"]["embed"],
            "ocr_pass": ocr_cfg["passes"]["ocr"], "tokens": tokens}
    return data, prov


def score_view(logits, labels, temperature):
    raw = vision.softmax_t(logits)
    m = vision.metrics(raw, labels)
    m["reliability"] = vision.reliability_bins(raw, labels)
    m["confusion_matrix"] = vision.confusion_matrix(raw, labels).tolist()
    cal = vision.softmax_t(logits, temperature)
    m["calibrated"] = {"temperature": temperature, **vision.metrics(cal, labels),
                       "reliability": vision.reliability_bins(cal, labels)}
    return m


def percentiles_ms(seconds):
    s = np.asarray(seconds, dtype=float) * 1000
    return {"n": int(len(s)), "p50_ms": float(np.percentile(s, 50)), "p95_ms": float(np.percentile(s, 95))}


def read_route_a(path, sub, check_manifest=True):
    """Route A outputs -> {id: row}, checked before the touch (no labels needed): the run's
    route_a.manifest.json names this subsample, and the outputs cover every subsample id."""
    path = Path(path)
    if check_manifest:
        manifest_path = path.parent / "route_a.manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"{manifest_path} missing: cannot tell which comparison set {path} was run on.")
        with open(manifest_path, encoding="utf-8") as f:
            cs = json.load(f)["comparison_set"]
        if (cs.get("split"), cs["seed"], cs["n"], cs["source_sha256"]) != \
                (sub["split"], sub["seed"], sub["n"], sub["source_sha256"]):
            raise SystemExit(f"{manifest_path}: Route A ran on {cs}, not on this test subsample.")
    sub_ids = sub["ids"]
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by_id = {r["id"]: r for r in rows}
    missing = sorted(set(sub_ids) - set(by_id))
    if missing:
        raise SystemExit(f"{path}: {len(missing)} subsample ids have no Route A output (e.g. {missing[:3]}).")
    return by_id


def score_route_a(by_id, sub_ids, label_of):
    """Route A on the subsample; decode errors count as wrong (Route A's own rule)."""
    probs, labels, correct, vlm_s, laya_s, errors = [], [], [], [], [], 0
    for i in sub_ids:
        r = by_id[i]
        y = label_of[i]
        if "label_name" in r and r["label_name"] != rvlcdip.CLASSES[y]:
            raise SystemExit(f"Route A output {i} has label {r['label_name']!r}, test.jsonl says "
                             f"{rvlcdip.CLASSES[y]!r}; the outputs are not from this test split.")
        labels.append(y)
        if r.get("error") is not None:
            errors += 1
            correct.append(False)
            probs.append(None)
            continue
        p = np.array([r["laya"]["probabilities"].get(c, 1e-6) for c in rvlcdip.CLASSES], dtype=np.float64)
        probs.append(p / p.sum())
        correct.append(r["laya"]["choice"] == rvlcdip.CLASSES[y])
        vlm_s.append(r["vlm_seconds"])
        laya_s.append(r["laya_seconds"])
    ok = [n for n, p in enumerate(probs) if p is not None]
    P, Y = np.stack([probs[n] for n in ok]), np.asarray(labels)[ok]
    m = {k: v for k, v in vision.metrics(P.astype(np.float32), Y).items()}
    m["n_errors"] = errors
    m["accuracy_errors_count_as_wrong"] = float(np.mean(correct))
    total = [a + b for a, b in zip(vlm_s, laya_s)]
    m["cost"] = {"gpu_seconds_per_image": float(np.mean(total)), "latency_total": percentiles_ms(total),
                 "latency_vlm": percentiles_ms(vlm_s), "latency_laya": percentiles_ms(laya_s)}
    m["note"] = ("Route A probabilities are laya.Agent's 4-decimal output with the Gate-0 checkpoint's own "
                 "temperatures; no refit. Agent choice used for accuracy. route_a.py also writes its own "
                 "route_a_results.json; the numbers here use the same metric code as Laya-Vision.")
    return m


def main():
    args = parse_args()
    for name in ("base_ckpt", "run_dir", "temperatures", "shards_root", "out_dir"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required.")
    if args.limit is not None and not args.dry_run:
        raise SystemExit("--limit is for --dry-run only: the test run scores the whole split once.")
    vision.check_laya_version()
    views = args.views.split(",")
    if any(v not in vision.MODALITIES for v in views):
        raise SystemExit(f"--views must be from {vision.MODALITIES}")
    device = torch.device(args.device)
    amp = None if args.amp == "none" else args.amp
    records_path = Path(args.records)
    out_dir = Path(args.out_dir) / ("dry_run" if args.dry_run else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} missing: the protocol is written before any test run.")

    # Everything that can fail without labels is checked before the touch.
    model, tok, cfg, base_sha, run_info, max_len, tokens = vision.load_run(args.base_ckpt, args.run_dir,
                                                                            args.checkpoint)
    temperatures, cal = vision.load_temperatures(args.temperatures, run_info["trainable_sha256"], tokens)
    if [v for v in views if v not in temperatures]:
        raise SystemExit(f"{args.temperatures} lacks temperatures for some of {views}.")
    if args.dry_run:
        split, forced = "val", None
    else:
        split = "test"
        if not Path(args.route_a_subsample).exists():
            raise SystemExit(f"{args.route_a_subsample} missing: draw it with route_a.py --make-comparison-set "
                             "(--records test.jsonl) and commit it first.")
        sub = load_subsample(Path(args.route_a_subsample), records_path)
        route_a = read_route_a(args.route_a_outputs, sub) if args.route_a_outputs else None
        forced = one_touch_guard(out_dir, args.force, args.reason)  # last check: renames superseded files
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not args.dry_run:
        # The touch begins here: mark it before any label is read, so a crash still counts.
        vision.write_json_atomic(out_dir / "final_eval.started.json",
                                 {"started": started, "script_git_hash": vision.script_git_hash(),
                                  "forced": forced})
        print("NOTICE: reading TEST labels (final_eval.py, one touch).", flush=True)

    records = vision.read_records(records_path, split)
    if args.dry_run:
        comparison, selected = load_comparison_set(Path(args.comparison_set), records_path, records)
        records = selected[:args.limit] if args.limit else selected
        picked = rvlcdip.stratified_sample(records, key=lambda r: r["label_int"],
                                           n=min(SUBSAMPLE_N, len(records)), seed=SUBSAMPLE_SEED)
        sub = {"ids": sorted(r["id"] for r in picked), "seed": SUBSAMPLE_SEED, "source_sha256": None,
               "note": "dry run: drawn from the val comparison set"}
        route_a = (read_route_a(args.route_a_outputs, sub, check_manifest=False)
                   if args.route_a_outputs else None)
    elif not set(sub["ids"]) <= {r["id"] for r in records}:
        raise SystemExit(f"{args.route_a_subsample}: ids are not all in the test split.")
    label_of = {r["id"]: rvlcdip.CLASSES.index(r["label_name"]) for r in records}

    model.to(device).eval()
    builder = vision.SequenceBuilder(tok, vision.doc_type_question(records), max_len, cfg.get("head_max_len", 256),
                                     tokens)
    data, prov = load_split_for_eval(args.shards_root, split, records, builder, args.cache_dir, tokens,
                                     allow_test=not args.dry_run)
    diff = vision.equivalence_check(model, data, builder, device, amp)
    print(f"equivalence check passed (max |diff| {diff:.2e}); {split}: {len(data)} scored of "
          f"{len(records)} records", flush=True)

    counts = Counter(rvlcdip.CLASSES[y] for y in data.labels)
    sub_rows = [data.index[i] for i in sub["ids"] if i in data.index]
    results = {
        "split": split, "dry_run": args.dry_run, "started": started, "forced": forced,
        "protocol": {"path": "docs/phase4_protocol.md", "sha256": vision.sha256_file(PROTOCOL)},
        "records_sha256": vision.sha256_file(records_path), "data": prov,
        "checkpoint": {**{k: v for k, v in run_info.items() if k != "train_identity"}, "visual_tokens": tokens,
                       "max_len": max_len, "base_model_sha256": base_sha},
        "calibration": {"path": str(args.temperatures), "sha256": vision.sha256_file(args.temperatures),
                        "temperatures": temperatures, "fitted_temperatures": cal["temperatures"],
                        "policy": f"views {list(vision.UNCALIBRATED_VIEWS)} at T = 1.0 (DECISIONS.md 2026-10-10)",
                        "fitted_on": cal["fitted_on"]["pool"],
                        "fitted_n": cal["fitted_on"]["n"]},
        "classes": rvlcdip.CLASSES, "counts_per_class": dict(counts),
        "majority_class_floor": max(counts.values()) / len(data),
        "amp": args.amp, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "script_git_hash": vision.script_git_hash(),
        "cost_note": "Laya-Vision cost excludes precompute (SigLIP + Tesseract, Phase 2).",
        "views": {},
        "route_a_subsample": {"seed": sub["seed"], "n": len(sub["ids"]), "n_scored_laya_vision": len(sub_rows),
                              "source_sha256": sub["source_sha256"], "laya_vision": {}},
    }
    pred_path = out_dir / "final_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8", newline="\n") as pf:
        for view in views:
            batch_times = []
            logits, rows, seconds, truncated = vision.predict_logits(model, data, builder, view, device,
                                                                     args.batch_size, amp, batch_times=batch_times)
            labels = data.labels[rows]
            m = score_view(logits, labels, temperatures[view])
            m["n_unscorable"] = len(prov["unscorable_ids"])
            m["accuracy_unscorable_count_as_wrong"] = (m["accuracy"] * len(rows)) / (len(rows) + m["n_unscorable"])
            m["text_truncated"] = truncated
            per_batch = percentiles_ms(batch_times[1:] or batch_times)
            m["cost"] = {
                "gpu_seconds_per_image_batched": seconds / len(rows),
                "batch_size": args.batch_size,
                "latency_batched_per_batch": per_batch,
                "latency_batched_per_image": {"p50_ms": per_batch["p50_ms"] / args.batch_size,
                                              "p95_ms": per_batch["p95_ms"] / args.batch_size},
                "latency_batch1": vision.batch1_latency(model, data, builder, view, device, amp, args.latency_n),
            }
            results["views"][view] = m
            pos = {r: n for n, r in enumerate(rows)}
            sel = [pos[r] for r in sub_rows]
            results["route_a_subsample"]["laya_vision"][view] = score_view(logits[sel], labels[sel],
                                                                           temperatures[view])
            plots.confusion_png(out_dir / f"confusion_{view}.png", np.array(m["confusion_matrix"]), rvlcdip.CLASSES,
                                f"Laya-Vision {view} view, {split} (n={len(rows)}, acc {m['accuracy']:.3f})")
            plots.reliability_diagram(out_dir / f"reliability_{view}.png",
                                      [("raw", m["reliability"], m["ece_15bin"]),
                                       (f"calibrated (T = {temperatures[view]:.3f})", m["calibrated"]["reliability"],
                                        m["calibrated"]["ece_15bin"])],
                                      f"Laya-Vision {view} view, {split} (n={len(rows)})")
            for r, z in zip(rows, logits):
                pf.write(json.dumps({"id": data.ids[r], "view": view, "label": rvlcdip.CLASSES[data.labels[r]],
                                     "pred": rvlcdip.CLASSES[int(z.argmax())],
                                     "logits": [round(float(x), 4) for x in z]}) + "\n")
            c = m["calibrated"]
            print(f"{view:9s} acc {m['accuracy']:.4f} | raw nll {m['nll']:.3f} ece {m['ece_15bin']:.4f} | "
                  f"calibrated (T {c['temperature']:.3f}) nll {c['nll']:.3f} ece {c['ece_15bin']:.4f} | "
                  f"{1000 * m['cost']['gpu_seconds_per_image_batched']:.1f} ms/img batched, batch-1 p50 "
                  f"{m['cost']['latency_batch1']['p50_ms']:.0f} ms", flush=True)
    if route_a is not None:
        results["route_a_subsample"]["route_a"] = score_route_a(route_a, sub["ids"], label_of)
        ra = results["route_a_subsample"]["route_a"]
        print(f"Route A on the subsample: acc {ra['accuracy_errors_count_as_wrong']:.4f} "
              f"({ra['n_errors']} errors counted wrong), {ra['cost']['gpu_seconds_per_image']:.1f} GPU-s/img", flush=True)
    else:
        results["route_a_subsample"]["route_a"] = None
        print("No --route-a-outputs: Route A not scored on test in this run.", flush=True)
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    vision.write_json_atomic(out_dir / "final_results.json", results)
    print(f"Wrote {out_dir / 'final_results.json'}, {pred_path.name}, confusion_<view>.png, reliability_<view>.png. "
          "Results are reported as they are (docs/phase4_protocol.md).")


if __name__ == "__main__":
    main()

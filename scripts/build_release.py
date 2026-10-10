"""Assemble the Laya-Vision release folder (DECISIONS.md 2026-10-10, "P5 release decisions").

Copies, never modifies:
  weights/base/                  the Gate-0 base, unchanged (rl_agent_config.json, model.safetensors,
                                 tokenizer/, encoder/ config). Weight files in encoder/ are never read
                                 (laya.common.build_model takes only the config from encoder/ and
                                 load_base takes every weight from model.safetensors); they are left
                                 out only after checking they equal model.safetensors' encoder
                                 tensors, otherwise the build stops. A folder named with
                                 --exclude-base is left out only if it holds a byte-identical copy
                                 of model.safetensors (e.g. a resume snapshot); otherwise the build
                                 stops.
  weights/trainable.safetensors  --trainable (the run's final/trainable.safetensors)
  calibration.json               calibrate.py output
  eval/final_results.json, eval/figures/{confusion,reliability}_<view>.png   (from --final-dir)
Writes laya_vision_config.json (architecture, sequence contract, temperatures, provenance, sha256
of every file; SigLIP + preprocessing + OCR spec of the train shards, as recorded in the run's
run_config.json),
then loads the result with laya_vision.load_release as a check.

README.md (model card), LICENSE and NOTICE are written separately.

    python scripts/build_release.py --base-ckpt release_inputs/base --run-config release_inputs/run_config.json \\
        --trainable release_inputs/trainable.safetensors --calibration release_inputs/calibration.json \\
        --final-dir docs/final --out release
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import vision
import rvlcdip
from laya_vision.release import BASE_DIR, CALIBRATION_FILE, CONFIG_NAME, FORMAT_VERSION, TRAINABLE_FILE, load_release

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
EVAL_FILES = ("final_results.json",)
FIGURE_GLOBS = ("confusion_*.png", "reliability_*.png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-ckpt", required=True, help="the Gate-0 base (step5/laya_finetuned)")
    p.add_argument("--run-config", required=True, help="the train_vision.py run's run_config.json (run02)")
    p.add_argument("--trainable", required=True, help="the run's final/trainable.safetensors")
    p.add_argument("--checkpoint-name", default="run02_final", help="name recorded in the config")
    p.add_argument("--calibration", required=True, help="calibration.json fitted on this checkpoint")
    p.add_argument("--exclude-base", action="append", default=[],
                   help="sub-folder of the base to leave out; allowed only for a byte-identical copy of "
                        "model.safetensors (repeatable)")
    p.add_argument("--final-dir", required=True, help="final_eval.py output (final_results.json, PNGs)")
    p.add_argument("--out", default="release")
    return p.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def encoder_weights_duplicate(encoder_files, model_path):
    """True when every tensor in the encoder/ weight files equals model.safetensors' encoder.<name>
    exactly (same dtype, shape and values)."""
    import torch
    from safetensors import safe_open

    with safe_open(str(model_path), "pt") as full:
        keys = set(full.keys())
        for path in encoder_files:
            if path.suffix != ".safetensors":
                print(f"  {path.name}: not safetensors; cannot compare.")
                return False
            with safe_open(str(path), "pt") as enc:
                for name in enc.keys():
                    full_name = name if name.startswith("encoder.") else f"encoder.{name}"
                    if full_name not in keys:
                        print(f"  {path.name}: {name} has no counterpart {full_name} in model.safetensors.")
                        return False
                    a, b = enc.get_tensor(name), full.get_tensor(full_name)
                    if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                        print(f"  {path.name}: {name} differs from {full_name}.")
                        return False
    return True


def check_excluded(base, rel, base_sha):
    """An excluded sub-folder must be a duplicate snapshot: its model.safetensors equals the base's."""
    sub = base / rel
    if not sub.is_dir() or sub.resolve().parent != base.resolve():
        raise SystemExit(f"--exclude-base {rel}: not a top-level folder of {base}.")
    weights = sub / "model.safetensors"
    if not weights.exists() or vision.sha256_file(weights) != base_sha:
        raise SystemExit(f"--exclude-base {rel}: its model.safetensors is not byte-identical to the base's; "
                         "leaving it out is a decision, not a deduplication. Stopping.")
    print(f"{rel}/ holds a byte-identical copy of model.safetensors: left out.")
    return sub


def copy_base(base, dest, excluded=()):
    """Copy the base unchanged, minus encoder/ weight files (checked to duplicate model.safetensors)
    and excluded duplicate folders (checked by check_excluded)."""
    encoder_weights = [p for p in (base / "encoder").iterdir() if p.is_file() and p.suffix in WEIGHT_SUFFIXES]
    if encoder_weights:
        print(f"encoder/ holds weight files {[p.name for p in encoder_weights]}; comparing with model.safetensors")
        if not encoder_weights_duplicate(encoder_weights, base / "model.safetensors"):
            raise SystemExit("encoder/ weights are not a copy of model.safetensors' encoder tensors. They are never "
                             "loaded, but shipping or dropping them is a decision (DECISIONS.md P5): stopping.")
        print("  identical to model.safetensors' encoder tensors: left out (config only).")
    skipped = set(encoder_weights)
    copied = []
    for src in sorted(p for p in base.rglob("*") if p.is_file()):
        if src in skipped or src.name.startswith(".") or "__pycache__" in src.parts:
            continue
        if any(ex in src.parents for ex in excluded):
            continue
        rel = src.relative_to(base)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / rel)
        copied.append(rel.as_posix())
    return copied, [p.name for p in encoder_weights]


def main():
    args = parse_args()
    vision.check_laya_version()
    base, out = Path(args.base_ckpt), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; the release is built once into an empty folder.")

    # Consistency before copying anything.
    run_cfg = read_json(args.run_config)
    ident = run_cfg["identity"]
    base_sha = vision.sha256_file(base / "model.safetensors")
    if base_sha != ident["base_model_sha256"]:
        raise SystemExit(f"{base}/model.safetensors is not the base {args.run_config} was trained from.")
    trainable = Path(args.trainable)
    trainable_sha = vision.sha256_file(trainable)
    cal = read_json(args.calibration)
    if cal["checkpoint"]["trainable_sha256"] != trainable_sha or cal["checkpoint"]["visual_tokens"] != ident["tokens"]:
        raise SystemExit(f"{args.calibration} was not fitted on {trainable}.")
    if not cal.get("complete", False):
        raise SystemExit(f"{args.calibration} is not complete.")
    if cal["checkpoint"]["base_model_sha256"] != base_sha:
        raise SystemExit(f"{args.calibration} names another base.")
    train = ident.get("train", {})
    embed, ocr = train.get("embed_pass", {}), train.get("ocr_pass", {})
    if train.get("split") != "train" or not {"siglip", "preprocessing"} <= set(embed) or "engine" not in ocr:
        raise SystemExit(f"{args.run_config}: identity.train needs split train, embed_pass.siglip, "
                         "embed_pass.preprocessing and ocr_pass.")
    excluded = [check_excluded(base, rel, base_sha) for rel in args.exclude_base]
    final_dir = Path(args.final_dir)
    final_results = final_dir / "final_results.json"
    if not final_results.exists():
        raise SystemExit(f"{final_results} missing.")
    figures = sorted(p for g in FIGURE_GLOBS for p in final_dir.glob(g))
    encoder_cfg = read_json(base / "encoder" / "config.json")

    out.mkdir(parents=True, exist_ok=True)
    base_files, dropped = copy_base(base, out / BASE_DIR, excluded)
    (out / TRAINABLE_FILE).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trainable, out / TRAINABLE_FILE)
    shutil.copy2(args.calibration, out / CALIBRATION_FILE)
    (out / "eval" / "figures").mkdir(parents=True, exist_ok=True)
    for name in EVAL_FILES:
        shutil.copy2(final_dir / name, out / "eval" / name)
    for fig in figures:
        shutil.copy2(fig, out / "eval" / "figures" / fig.name)

    files = {p.relative_to(out).as_posix(): vision.sha256_file(p)
             for p in sorted(out.rglob("*")) if p.is_file()}
    fitted = cal["temperatures"]
    config = {
        "format_version": FORMAT_VERSION,
        "model": "Laya-Vision",
        "checkpoint": args.checkpoint_name,
        "laya_version": vision.LAYA_VERSION,
        "visual_tokens": ident["tokens"],
        "visual_dim": vision.VISUAL_DIM,
        "d_model": encoder_cfg["hidden_size"],
        "max_len": ident["max_len"],
        "head_max_len": ident["head_max_len"],
        "head_len_doc_type": ident["head_len"],
        "text_budget": ident["text_budget"],
        "trainable_params": ident["trainable_params"],
        "sequence": {
            "layout": "[CLS] question head with [MASK] option markers [SEP] state [SEP] (laya.common.build_sequence)",
            "state_order": ["visual", "ocr_text", "derived_features", "user_text"],
            "visual_block": f"{ident['tokens']} placeholder ids (pad id) then one [SEP]; at the placeholders "
                            "the embedding output is replaced by norm(projector(SigLIP token))",
            "truncation": "text yields first, from the end; options and visual tokens are never truncated",
            "no_image": "exactly upstream laya's sequence and forward",
        },
        "doc_type": {"question_type": "choice", "classes": rvlcdip.CLASSES},
        "views": {"combined": "visual + OCR text", "image": "visual only", "text": "OCR text only (upstream path)"},
        "siglip": embed["siglip"],
        "preprocess": embed["preprocessing"],
        "ocr": ocr,
        "temperatures_fitted": fitted,
        "uncalibrated_views": list(vision.UNCALIBRATED_VIEWS),
        "temperatures_applied": {v: 1.0 if v in vision.UNCALIBRATED_VIEWS else float(t) for v, t in fitted.items()},
        "temperature_note": "Applied temperatures are those of the evaluation protocol (docs/phase4_protocol.md "
                            "§2). The SDK's image=None path is upstream laya's, including the base's "
                            "rl_agent_config.json temperatures (Release.upstream_temperature).",
        "base_model_sha256": base_sha,
        "trainable_sha256": trainable_sha,
        "encoder_weight_files_left_out": dropped,
        "base_folders_left_out": {p.name: "byte-identical copy of model.safetensors" for p in excluded},
        "provenance": {
            "base": "Gate-0 fine-tune of convaiinnovations/laya (typed-decisions notebook, reproduced in "
                    "Phase 0), shipped unchanged",
            "train_git_hash": run_cfg.get("script_git_hash"),
            "train": {k: ident[k] for k in ("epochs", "micro_batch", "grad_accum", "lr_head", "lr_projector",
                                            "warmup_frac", "weight_decay", "modality_probs", "seed", "amp",
                                            "loss") if k in ident},
            "calibration_fitted_on": cal.get("fitted_on", {}).get("pool"),
            "build_git_hash": vision.script_git_hash(),
        },
        "files": files,
    }
    vision.write_json_atomic(out / CONFIG_NAME, config)
    print(f"Wrote {out}: {len(base_files)} base files, trainable, calibration, {len(figures)} figures, {CONFIG_NAME}.")

    release = load_release(out)
    print(f"load_release check passed: temperatures {release.temperatures}, "
          f"upstream choice/16 temperature {release.upstream_temperature('choice', 16):.4f}")


if __name__ == "__main__":
    sys.exit(main())

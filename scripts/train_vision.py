"""Phase 3: train the Laya-Vision projector + decision head on precomputed shards.

Frozen: SigLIP (offline, Phase 2), the Laya encoder (Gate-0 fine-tuned
convaiinnovations/laya, ModernBERT-large) and the act head. Trained: the
projector (768 -> 1024 -> 1024) and the decision head. Design:
docs/phase3_design.md; decisions: DECISIONS.md 2026-10-08.

Per example and epoch the modality is drawn from a hash of (seed, epoch, id):
40% combined (visual + OCR), 30% image-only, 30% text-only. Loss, optimiser and
noise schedule follow upstream's fine-tune script (CE + RLCD, AdamW wd 0.01,
grad clip 1.0, sigma 0.4 -> 0.1), with two recorded divergences: the encoder
group is gone (frozen) and a projector group (lr 1e-3) is added; a linear warmup
precedes upstream's cosine schedule; bf16 autocast + encoder gradient
checkpointing replace upstream's fp32.

Validation while training uses val images OUTSIDE the fixed 2,000-image
comparison set (first --monitor-limit of them), so checkpoint choice never
touches the ids eval_vision.py reports on. Test is never loaded.

Outputs, in --out-dir (put it on Drive):
  run_config.json      identity of the run; a resume must match it
  train_log.jsonl      one line per --log-every updates, plus one per epoch eval
  latest/              trainable weights, optimizer, scheduler, RNG, position
  epoch_N/             trainable weights after epoch N (+ monitor metrics)
  final/               trainable weights after the last epoch
Every --save-every updates `latest/` is replaced atomically; a disconnected run
resumes from it when started again with the same arguments.

    python scripts/train_vision.py --base-ckpt <drive>/step5/laya_finetuned \\
        --shards-root <drive>/phase2/shards --records-dir <drive>/data/rvl_v1 \\
        --out-dir <drive>/phase3/runs/tiny --limit 2000 --epochs 1          # tiny run
"""

import argparse
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import rvlcdip
import vision
from route_a import DEFAULT_COMPARISON_SET


def parse_args():
    p = argparse.ArgumentParser(description="Train the Laya-Vision projector + decision head")
    p.add_argument("--base-ckpt", required=True, help="Gate-0 fine-tuned checkpoint dir (step5/laya_finetuned)")
    p.add_argument("--shards-root", required=True, help="Phase 2 shards dir holding train/ and val/")
    p.add_argument("--records-dir", required=True, help="dir with train.jsonl and val.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default=None, help="OCR token cache (default: <out-dir>/cache)")
    p.add_argument("--comparison-set", default=str(DEFAULT_COMPARISON_SET))
    p.add_argument("--limit", type=int, default=None, help="dry run: first N train examples")
    p.add_argument("--monitor-limit", type=int, default=1000, help="val (non-comparison) examples per epoch eval")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lr-projector", type=float, default=1e-3)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-len", type=int, default=vision.MAX_LEN)
    p.add_argument("--tokens", type=int, default=vision.N_VISUAL, choices=[8, 16, 32, 64])
    p.add_argument("--modality-probs", default="0.4,0.3,0.3", help="combined,image,text")
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--no-checkpointing", action="store_true", help="disable encoder/head gradient checkpointing")
    p.add_argument("--save-every", type=int, default=200, help="optimizer updates between latest/ saves")
    p.add_argument("--log-every", type=int, default=20, help="optimizer updates between log lines")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def identity(args, base_sha, train_prov, monitor_prov, param_report, builder):
    return {
        "base_ckpt": str(Path(args.base_ckpt).resolve()),
        "base_model_sha256": base_sha,
        "laya_version": vision.LAYA_VERSION,
        "max_len": args.max_len,
        "head_max_len": builder.head_max_len,
        "head_len": builder.head_len,
        "text_budget": {"combined": builder.text_budget(True), "text": builder.text_budget(False)},
        "tokens": args.tokens,
        "limit": args.limit,
        "monitor_limit": args.monitor_limit,
        "epochs": args.epochs,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "lr_head": args.lr_head,
        "lr_projector": args.lr_projector,
        "warmup_frac": args.warmup_frac,
        "weight_decay": args.weight_decay,
        "modality_probs": args.modality_probs,
        "seed": args.seed,
        "amp": args.amp,
        "gradient_checkpointing": not args.no_checkpointing,
        "loss": "upstream CE + RLCD (4 samples, sigma 0.4->0.1, w_sph 0.75), grad clip 1.0",
        "train": train_prov,
        "monitor": monitor_prov,
        "trainable_params": param_report["trainable_total"],
    }


def save_dir(out_dir, name, writer):
    """Write into <name>.tmp, then swap it in; a crash leaves either the old or the new copy."""
    final, tmp, old = out_dir / name, out_dir / f"{name}.tmp", out_dir / f"{name}.old"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    writer(tmp)
    if final.exists():
        shutil.rmtree(old, ignore_errors=True)
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


def find_resume(out_dir):
    for name in ("latest", "latest.old"):
        if (out_dir / name / "state.json").exists():
            return out_dir / name
    return None


def main():
    args = parse_args()
    vision.check_laya_version()
    probs = tuple(float(x) for x in args.modality_probs.split(","))
    if len(probs) != 3 or abs(sum(probs) - 1) > 1e-6:
        raise SystemExit("--modality-probs needs three values summing to 1 (combined,image,text).")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    vision.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False

    print("Loading Gate-0 base checkpoint ...", flush=True)
    model, tok, cfg, base_sha = vision.load_base(args.base_ckpt, args.max_len, projector_seed=args.seed)
    records_dir = Path(args.records_dir)
    train_records = vision.read_records(records_dir / "train.jsonl", "train")
    question = vision.doc_type_question(train_records)
    del train_records
    builder = vision.SequenceBuilder(tok, question, args.max_len, cfg.get("head_max_len", 256), args.tokens)
    print(f"head {builder.head_len} tokens; text budget combined {builder.text_budget(True)}, "
          f"text-only {builder.text_budget(False)}", flush=True)

    cache_dir = args.cache_dir or out_dir / "cache"
    train, train_prov = vision.load_split_data(args.shards_root, records_dir / "train.jsonl", "train", builder,
                                               cache_dir, args.tokens, args.limit)
    val, val_prov = vision.load_split_data(args.shards_root, records_dir / "val.jsonl", "val", builder,
                                           cache_dir, args.tokens)
    with open(args.comparison_set, encoding="utf-8") as f:
        comparison_ids = set(json.load(f)["ids"])
    monitor_rows = [r for r, i in enumerate(val.ids) if i not in comparison_ids][:args.monitor_limit]
    monitor = val.subset(monitor_rows)
    del val
    monitor_prov = {**val_prov, "n": len(monitor), "excludes": "comparison set ids"}

    vision.freeze_for_training(model)
    report = vision.parameter_report(model)
    print("trainable parameters: " + ", ".join(f"{k} {v['trainable']:,}" for k, v in report["modules"].items()
                                               if v["trainable"]) + f"; total {report['trainable_total']:,} "
          f"of {report['total']:,}", flush=True)

    ident = identity(args, base_sha, train_prov, monitor_prov, report, builder)
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing["identity"] != ident:
            raise SystemExit(f"{config_path} was written by a different configuration; resume refused. "
                             "Use the original arguments or a new --out-dir.")
    else:
        vision.write_json_atomic(config_path, {"identity": ident, "script_git_hash": vision.script_git_hash(),
                                               "started": time.strftime("%Y-%m-%dT%H:%M:%S")})

    model.to(device)
    if not args.no_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    diff = vision.equivalence_check(model, monitor, builder, device, None if args.amp == "none" else args.amp)
    print(f"equivalence check passed (no-image path == DecisionModel.forward; inputs_embeds path max |diff| "
          f"{diff:.2e})", flush=True)

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("projector.")]
    proj_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("projector.")]
    trainable = head_params + proj_params
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": args.lr_head},
                                   {"params": proj_params, "lr": args.lr_projector}],
                                  weight_decay=args.weight_decay)
    micro_per_epoch = -(-len(train) // args.micro_batch)
    updates_per_epoch = -(-micro_per_epoch // args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup = int(args.warmup_frac * total_updates)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [vision.lr_lambda(total_updates, warmup, args.lr_head),
                    vision.lr_lambda(total_updates, warmup, args.lr_projector)])

    start_epoch, start_micro, update = 0, 0, 0
    resume = find_resume(out_dir)
    if resume is not None:
        with open(resume / "state.json", encoding="utf-8") as f:
            state = json.load(f)
        vision.load_trainable(model, resume / "trainable.safetensors")
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location=device, weights_only=False))
        scheduler.load_state_dict(torch.load(resume / "scheduler.pt", weights_only=False))
        rng = torch.load(resume / "rng.pt", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        start_epoch, start_micro, update = state["epoch"], state["micro_step"], state["update"]
        print(f"Resumed from {resume}: epoch {start_epoch + 1}, micro-step {start_micro}, update {update}",
              flush=True)

    print(f"{len(train)} train examples; {micro_per_epoch} micro-batches x {args.epochs} epochs; "
          f"{total_updates} updates (warmup {warmup}); effective batch {args.micro_batch * args.grad_accum}",
          flush=True)

    def checkpoint_state(epoch, micro_step):
        def write(path):
            from safetensors.torch import save_file

            save_file(vision.trainable_state(model), str(path / "trainable.safetensors"))
            torch.save(optimizer.state_dict(), path / "optimizer.pt")
            torch.save(scheduler.state_dict(), path / "scheduler.pt")
            torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       path / "rng.pt")
            vision.write_json_atomic(path / "state.json", {"epoch": epoch, "micro_step": micro_step,
                                                           "update": update, "saved": time.strftime("%Y-%m-%dT%H:%M:%S")})
        save_dir(out_dir, "latest", write)

    log_path = out_dir / "train_log.jsonl"

    def log(row):
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    amp = None if args.amp == "none" else args.amp
    lengths = [train.ocr_lengths[r] for r in range(len(train))]
    for epoch in range(start_epoch, args.epochs):
        sigma = vision.sigma_for_epoch(epoch, args.epochs)
        rng = random.Random(args.seed + epoch)
        order = list(range(len(train)))
        rng.shuffle(order)
        batches = vision.length_batches(lengths, order, args.micro_batch, rng)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        window = {"loss": 0.0, "ce": 0.0, "rl": 0.0, "n": 0, "examples": 0, "tokens": 0, "correct": 0,
                  "modality": Counter(), "truncated": 0, "t0": time.perf_counter()}
        first = start_micro if epoch == start_epoch else 0
        for micro, rows in enumerate(batches):
            if micro < first:
                continue
            items = [train.item(builder, r, vision.modality_for(args.seed, epoch, train.ids[r], probs)) for r in rows]
            batch = vision.collate(items, train, tok.pad_token_id)
            with vision.autocast(device, amp):
                logits, act = vision.forward(model, batch, device)
            mask = batch["marker_mask"].to(device)
            loss, ce, rl = vision.rlcd_loss(logits, mask, batch["target"].to(device), batch["qtype"].to(device), sigma)
            ((loss + 0.0 * act.sum()) / args.grad_accum).backward()

            window["loss"] += loss.item()
            window["ce"] += ce.item()
            window["rl"] += rl.item()
            window["n"] += 1
            window["examples"] += len(items)
            window["tokens"] += int(batch["attention_mask"].sum())
            window["correct"] += int((logits.argmax(-1).cpu() == batch["label"]).sum())
            window["modality"].update(it["modality"] for it in items)
            window["truncated"] += sum(it["truncated"] for it in items)

            last = micro + 1 == len(batches)
            if (micro + 1) % args.grad_accum == 0 or last:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % args.log_every == 0 or last:
                    dt = time.perf_counter() - window["t0"]
                    row = {"update": update, "epoch": epoch + 1, "micro_step": micro + 1,
                           "loss": window["loss"] / window["n"], "ce": window["ce"] / window["n"],
                           "rl": window["rl"] / window["n"], "train_acc": window["correct"] / window["examples"],
                           "grad_norm": float(grad_norm), "lr_head": scheduler.get_last_lr()[0],
                           "lr_projector": scheduler.get_last_lr()[1], "sigma": sigma,
                           "examples_per_s": window["examples"] / dt, "tokens_per_example":
                           window["tokens"] / window["examples"], "modality": dict(window["modality"]),
                           "text_truncated": window["truncated"],
                           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None}
                    log(row)
                    print(f"  ep {epoch + 1} upd {update}/{total_updates} loss {row['loss']:.4f} "
                          f"(ce {row['ce']:.4f}) acc {row['train_acc']:.3f} gnorm {row['grad_norm']:.2f} "
                          f"{row['examples_per_s']:.1f} ex/s", flush=True)
                    window = {"loss": 0.0, "ce": 0.0, "rl": 0.0, "n": 0, "examples": 0, "tokens": 0, "correct": 0,
                              "modality": Counter(), "truncated": 0, "t0": time.perf_counter()}
                if update % args.save_every == 0 and not last:
                    checkpoint_state(epoch, micro + 1)

        model.eval()
        evals = {}
        for modality in vision.MODALITIES:
            p, rows, seconds, truncated = vision.predict(model, monitor, builder, modality, device,
                                                         args.eval_batch, amp)
            evals[modality] = {**vision.metrics(p, monitor.labels[rows]), "text_truncated": truncated}
        log({"epoch_eval": epoch + 1, "update": update, "monitor": {m: {k: v for k, v in e.items()
                                                                       if k in ("n", "accuracy", "nll", "ece_15bin")}
                                                                   for m, e in evals.items()}})
        print(f"epoch {epoch + 1} monitor (val outside comparison set, n={len(monitor)}): " + ", ".join(
            f"{m} acc {e['accuracy']:.4f} nll {e['nll']:.3f}" for m, e in evals.items()), flush=True)

        def write_epoch(path, evals=evals, epoch=epoch):
            from safetensors.torch import save_file

            save_file(vision.trainable_state(model), str(path / "trainable.safetensors"))
            vision.write_json_atomic(path / "monitor_metrics.json", {"epoch": epoch + 1, "update": update,
                                                                      "metrics": evals})
        save_dir(out_dir, f"epoch_{epoch + 1}", write_epoch)
        checkpoint_state(epoch + 1, 0)

    save_dir(out_dir, "final", lambda path: shutil.copytree(out_dir / f"epoch_{args.epochs}", path,
                                                            dirs_exist_ok=True))
    with open(out_dir / f"epoch_{args.epochs}" / "monitor_metrics.json", encoding="utf-8") as f:
        final = json.load(f)["metrics"]
    print("\nDone. experiments.md line (fill in the commit hash of the code that ran):")
    print(f"{time.strftime('%Y-%m-%d')} | <commit> | P3 train_vision: Gate-0 laya (ModernBERT-large) frozen, "
          f"projector+head {report['trainable_total'] / 1e6:.1f}M, {len(train)} train, {args.epochs} ep, "
          f"mb{args.micro_batch}x{args.grad_accum}, lr head {args.lr_head} / proj {args.lr_projector}, "
          f"{args.amp}, modality {args.modality_probs}, {args.tokens} visual tokens, max_len {args.max_len} | "
          f"monitor val (n={len(monitor)}, non-comparison): " + ", ".join(
              f"{m} {final[m]['accuracy']:.4f}" for m in vision.MODALITIES) + f" | out {out_dir}")


if __name__ == "__main__":
    main()

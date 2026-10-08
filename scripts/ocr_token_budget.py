"""How much OCR text fits? Token counts of train OCR against the Laya-Vision budget.

Tokenises every train OCR text with the Gate-0 checkpoint's tokenizer, exactly
as vision.SequenceBuilder does (upstream state tokenisation), and compares the
length with the text budget left at max_len 512:
  combined  max_len - question head - closing [SEP] - 32 visual - 1 [SEP]
  text      max_len - question head - closing [SEP]
Reports the share of pages over budget, the share of tokens they lose, and
flags (DECISIONS.md 2026-10-08) when more than --flag-share of pages lose at
least --large-share of their text in the combined view. Per-class figures use
train labels. Train only: no val or test text is read.

    python scripts/ocr_token_budget.py --base-ckpt <drive>/step5/laya_finetuned \\
        --shards-root <drive>/phase2/shards --records <drive>/data/rvl_v1/train.jsonl --out ocr_budget_train.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import rvlcdip
import shards
import vision


def describe(lengths, budget, large_share):
    lengths = np.asarray(lengths)
    over = lengths > budget
    lost = np.where(over, (lengths - budget) / np.maximum(lengths, 1), 0.0)
    res = {
        "budget": int(budget),
        "pages": int(len(lengths)),
        "share_over_budget": float(over.mean()),
        f"share_losing_ge_{large_share:g}": float((lost >= large_share).mean()),
        "share_of_all_tokens_lost": float((lengths - np.minimum(lengths, budget)).sum() / max(1, lengths.sum())),
    }
    if over.any():
        res["fraction_lost_among_over"] = {f"p{q}": float(np.percentile(lost[over], q)) for q in (25, 50, 75, 90)}
    return res


def main():
    p = argparse.ArgumentParser(description="OCR token counts vs the Laya-Vision text budget (train)")
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--shards-root", required=True)
    p.add_argument("--records", required=True, help="train.jsonl (labels for the per-class view)")
    p.add_argument("--max-len", type=int, default=vision.MAX_LEN)
    p.add_argument("--n-visual", type=int, default=vision.N_VISUAL)
    p.add_argument("--large-share", type=float, default=0.5, help="'large share' of a page's tokens lost")
    p.add_argument("--flag-share", type=float, default=0.25, help="flag when more pages than this lose a large share")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    records = vision.read_records(args.records, "train")
    label_of = {r["id"]: r["label_name"] for r in records}
    question = vision.doc_type_question(records)
    with open(Path(args.base_ckpt) / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    _fix_tokenizer_config(str(args.base_ckpt))
    tok = AutoTokenizer.from_pretrained(Path(args.base_ckpt) / "tokenizer")
    builder = vision.SequenceBuilder(tok, question, args.max_len, cfg.get("head_max_len", 256), args.n_visual)

    _, texts = shards.load_ocr(Path(args.shards_root) / "train")
    ids = sorted(texts)[:args.limit] if args.limit else sorted(texts)
    t0 = time.perf_counter()
    lengths = []
    for start in range(0, len(ids), 512):
        lengths.extend(len(x) for x in builder.encode_text([texts[i] for i in ids[start:start + 512]]))
    lengths = np.asarray(lengths)
    print(f"tokenised {len(ids)} train OCR texts in {time.perf_counter() - t0:.0f}s", flush=True)

    res = {
        "tokenizer": str(Path(args.base_ckpt) / "tokenizer"),
        "max_len": args.max_len, "head_len": builder.head_len, "n_visual": args.n_visual,
        "pages": len(ids),
        "empty_pages": int((lengths == 0).sum()),
        "tokens": {"mean": float(lengths.mean()), **{f"p{q}": float(np.percentile(lengths, q))
                                                     for q in (10, 25, 50, 75, 90, 95, 99)},
                   "max": int(lengths.max())},
        "combined": describe(lengths, builder.text_budget(True), args.large_share),
        "text_only": describe(lengths, builder.text_budget(False), args.large_share),
        "per_class_combined": {},
    }
    labels = np.array([label_of[i] for i in ids])
    for name in rvlcdip.CLASSES:
        m = labels == name
        if m.any():
            d = describe(lengths[m], builder.text_budget(True), args.large_share)
            res["per_class_combined"][name] = {"pages": d["pages"], "median_tokens": float(np.median(lengths[m])),
                                               "share_over_budget": d["share_over_budget"],
                                               f"share_losing_ge_{args.large_share:g}":
                                                   d[f"share_losing_ge_{args.large_share:g}"]}
    key = f"share_losing_ge_{args.large_share:g}"
    res["flag"] = res["combined"][key] > args.flag_share

    t = res["tokens"]
    print(f"head {builder.head_len} tokens; budget combined {res['combined']['budget']}, "
          f"text-only {res['text_only']['budget']}")
    print(f"OCR tokens/page: mean {t['mean']:.0f}, median {t['p50']:.0f}, p75 {t['p75']:.0f}, p90 {t['p90']:.0f}, "
          f"p99 {t['p99']:.0f}, max {t['max']}; empty pages {res['empty_pages']}")
    for view in ("combined", "text_only"):
        d = res[view]
        line = (f"{view:9s}: {100 * d['share_over_budget']:.1f}% of pages over budget; "
                f"{100 * d[key]:.1f}% lose >= {100 * args.large_share:.0f}% of their tokens; "
                f"{100 * d['share_of_all_tokens_lost']:.1f}% of all OCR tokens dropped")
        if "fraction_lost_among_over" in d:
            f_ = d["fraction_lost_among_over"]
            line += f"; among pages over: median {100 * f_['p50']:.0f}% lost (p90 {100 * f_['p90']:.0f}%)"
        print(line)
    print("per class (combined): " + ", ".join(
        f"{n} {100 * v['share_over_budget']:.0f}%" for n, v in sorted(
            res["per_class_combined"].items(), key=lambda kv: -kv[1]["share_over_budget"])))
    if res["flag"]:
        print(f"FLAG: {100 * res['combined'][key]:.1f}% of pages lose >= {100 * args.large_share:.0f}% of their "
              f"OCR text in the combined view (> {100 * args.flag_share:.0f}%). Review the budget before training.")
    else:
        print(f"No flag: pages losing >= {100 * args.large_share:.0f}% of their text are "
              f"{100 * res['combined'][key]:.1f}% (threshold {100 * args.flag_share:.0f}%).")
    if args.out:
        vision.write_json_atomic(args.out, res)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

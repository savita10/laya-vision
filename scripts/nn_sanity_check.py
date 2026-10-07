"""P2 gate check: do same-class documents cluster in the pooled SigLIP embeddings?

Loads the val embedding shards written by precompute.py and the val labels,
and for every image finds its nearest neighbours (cosine, excluding itself)
among the other val images. Reports, overall and per class:
  1-NN agreement   the nearest neighbour has the same label
  k-NN accuracy    majority vote of the k nearest (ties -> nearest wins)
against the chance level of a random neighbour (sum of squared class
shares). Two representations are scored:
  mean   mean of the pooled tokens (one 768-d vector per image)
  flat   all pooled tokens concatenated (keeps the layout grid)
It also counts near-duplicate pairs (cosine >= --dup-threshold) and how many
cross classes; RVL-CDIP is known to contain duplicates.

Scores the --tokens view (default 32) derived from the stored 8x8 grid
(shards.py). Val only: refuses any other split.

    python scripts/nn_sanity_check.py --shards <out-dir>/val --records data/rvl_v1/val.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import rvlcdip
import shards


def neighbours(x, k, chunk=1024):
    """Top-k cosine neighbours of each row (excluding itself) -> (indices, sims)."""
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    idx = np.empty((len(x), k), dtype=np.int64)
    sims = np.empty((len(x), k), dtype=np.float32)
    for start in range(0, len(x), chunk):
        s = x[start:start + chunk] @ x.T
        rows = np.arange(s.shape[0])
        s[rows, start + rows] = -np.inf
        part = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
        idx[start:start + chunk] = np.take_along_axis(part, order, axis=1)
        sims[start:start + chunk] = np.take_along_axis(s, idx[start:start + chunk], axis=1)
    return idx, sims


def score(labels, idx, k):
    nn1 = labels[idx[:, 0]] == labels
    votes = []
    for i in range(len(labels)):
        counts = Counter(labels[idx[i, :k]].tolist())
        best = max(counts.values())
        # tie -> the nearest neighbour among the tied labels
        winner = next(l for l in labels[idx[i, :k]].tolist() if counts[l] == best)
        votes.append(winner == labels[i])
    votes = np.array(votes)
    per_class = {}
    for c, name in enumerate(rvlcdip.CLASSES):
        m = labels == c
        if m.any():
            per_class[name] = {"n": int(m.sum()), "nn1_agreement": float(nn1[m].mean()),
                               f"knn{k}_accuracy": float(votes[m].mean())}
    return {"nn1_agreement": float(nn1.mean()), f"knn{k}_accuracy": float(votes.mean()), "per_class": per_class}


def duplicates(labels, idx, sims, threshold):
    pairs = {tuple(sorted((i, int(j)))) for i in range(len(labels)) for j, s in zip(idx[i], sims[i]) if s >= threshold}
    cross = sum(labels[a] != labels[b] for a, b in pairs)
    return {"threshold": threshold, "pairs": len(pairs), "cross_class_pairs": int(cross)}


def main():
    parser = argparse.ArgumentParser(description="Nearest-neighbour label agreement on val embeddings")
    parser.add_argument("--shards", required=True, help="<out-dir>/val from precompute.py")
    parser.add_argument("--records", required=True, help="val.jsonl (labels)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dup-threshold", type=float, default=0.995)
    parser.add_argument("--tokens", type=int, choices=sorted(shards.TOKEN_VIEWS), default=32,
                        help="token view derived from the stored 8x8 grid")
    parser.add_argument("--allow-partial", action="store_true", help="score only the shards that are done")
    parser.add_argument("--out", default=None, help="write results JSON here")
    args = parser.parse_args()

    labels_by_id, splits = {}, set()
    with open(args.records, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                splits.add(r["split"])
                labels_by_id[r["id"]] = r["label_int"]
    if splits != {"val"}:
        raise SystemExit("--records must be val.jsonl; this check runs on val only.")

    config, ids, emb = shards.load_split(Path(args.shards), args.tokens, args.allow_partial)
    if config["identity"]["split"] != "val":
        raise SystemExit(f"{args.shards} holds split {config['identity']['split']!r}; this check runs on val only.")
    labels = np.array([labels_by_id[i] for i in ids])
    shares = np.bincount(labels, minlength=len(rvlcdip.CLASSES)) / len(labels)
    chance = float((shares ** 2).sum())
    print(f"{len(ids)} val images, embeddings {emb.shape[1:]} "
          f"({args.tokens}-token view {shards.TOKEN_VIEWS[args.tokens]} of stored "
          f"{config['passes']['embed']['siglip']['pooling']['grid_rows_cols']}), "
          f"chance agreement {chance:.4f}\n")

    reps = {"mean": emb.mean(axis=1), "flat": emb.reshape(len(emb), -1)}
    results = {"n": len(ids), "k": args.k, "tokens": args.tokens, "chance_agreement": chance,
               "embed_config": config["passes"]["embed"], "representations": {}}
    for name, x in reps.items():
        idx, sims = neighbours(x, args.k)
        r = score(labels, idx, args.k)
        r["duplicates"] = duplicates(labels, idx, sims, args.dup_threshold)
        results["representations"][name] = r

    knn = f"knn{args.k}_accuracy"
    header = f"{'class':24s} {'n':>5s}" + "".join(f" {n + ' 1-NN':>10s} {n + f' {args.k}-NN':>10s}" for n in reps)
    print(header)
    print("-" * len(header))
    for cls in rvlcdip.CLASSES:
        row = results["representations"]["mean"]["per_class"].get(cls)
        if row is None:
            continue
        line = f"{cls:24s} {row['n']:5d}"
        for n in reps:
            pc = results["representations"][n]["per_class"][cls]
            line += f" {pc['nn1_agreement']:10.3f} {pc[knn]:10.3f}"
        print(line)
    print("-" * len(header))
    line = f"{'overall':24s} {len(ids):5d}"
    for n in reps:
        r = results["representations"][n]
        line += f" {r['nn1_agreement']:10.3f} {r[knn]:10.3f}"
    print(line)
    print(f"\nchance (random neighbour) {chance:.4f}; 1/16 = {1 / 16:.4f}")
    for n in reps:
        r = results["representations"][n]
        d = r["duplicates"]
        print(f"{n}: 1-NN agreement {r['nn1_agreement']:.3f} = {r['nn1_agreement'] / chance:.1f}x chance; "
              f"{d['pairs']} near-duplicate pairs (cos >= {d['threshold']}), {d['cross_class_pairs']} cross-class")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

"""Load precompute.py embedding shards and derive the token views.

Embeddings are stored at 64 tokens (8x8 grid over SigLIP's 24x24 patches,
row-major, fp16; DECISIONS.md 2026-10-07). The 32 / 16 / 8 token views are
derived at load time by averaging equal-size blocks of the stored grid. Each
stored token is the mean of an equal number of patches (3x3), so a block mean
of stored tokens equals the direct average pool of the patch grid: the
derivation is exact, up to the fp16 rounding of the stored values (computed in
fp32 here).

  tokens  grid (rows x cols)  patches per token
  64      8x8                 3x3   (as stored)
  32      8x4                 3x6
  16      4x4                 6x6
  8       4x2                 6x12

load_split refuses the test split unless allow_test=True; only final_eval.py
passes it (DECISIONS.md 2026-10-07).
"""

import json
from pathlib import Path

import numpy as np

TOKEN_VIEWS = {64: (8, 8), 32: (8, 4), 16: (4, 4), 8: (4, 2)}


def derive_view(emb, stored_grid, tokens):
    """[n, R*C, d] stored row-major on stored_grid -> [n, r*c, d] float32 on TOKEN_VIEWS[tokens]."""
    R, C = stored_grid
    r, c = TOKEN_VIEWS[tokens]
    n, t, d = emb.shape
    if t != R * C:
        raise ValueError(f"embeddings have {t} tokens, stored grid {R}x{C} needs {R * C}")
    if R % r or C % c:
        raise ValueError(f"{tokens}-token view {r}x{c} does not evenly divide the stored {R}x{C} grid")
    x = emb.astype(np.float32).reshape(n, r, R // r, c, C // c, d)
    return x.mean(axis=(2, 4)).reshape(n, r * c, d)


def read_done(split_dir):
    path = Path(split_dir) / "done.log"
    if not path.exists():
        return set()
    return {tuple(line.split()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_split(split_dir, tokens=32, allow_partial=False, allow_test=False):
    """Load every embed-done shard of one split.

    Returns (config, ids, emb) with emb float32 [n, tokens, d]. Images that
    failed to decode (manifest "valid" false) are dropped.
    """
    split_dir = Path(split_dir)
    with open(split_dir / "run_config.json", encoding="utf-8") as f:
        config = json.load(f)
    split = config["identity"]["split"]
    if split == "test" and not allow_test:
        raise SystemExit(f"{split_dir} holds the test split; only final_eval.py may load it.")
    if "embed" not in config["passes"]:
        raise SystemExit(f"{split_dir}: no embed pass recorded.")
    stored_grid = tuple(config["passes"]["embed"]["siglip"]["pooling"]["grid_rows_cols"])
    done = read_done(split_dir)
    shard_dirs = sorted(p for p in split_dir.glob("shard_*") if p.is_dir())
    complete = [p for p in shard_dirs if ("embed", p.name) in done]
    if not shard_dirs or len(complete) != len(shard_dirs):
        msg = f"{split_dir}: embed done for {len(complete)}/{len(shard_dirs)} shards"
        if not allow_partial or not complete:
            raise SystemExit(msg + "; finish the run or allow partial loading.")
        print("WARNING: " + msg, flush=True)
    ids, embs = [], []
    for p in complete:
        with open(p / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        emb = np.load(p / "embeddings.npy")
        valid = manifest["passes"]["embed"]["valid"]
        if emb.shape[0] != len(manifest["ids"]) or len(valid) != len(manifest["ids"]):
            raise SystemExit(f"{p}: embeddings / manifest length mismatch")
        keep = [i for i, v in enumerate(valid) if v]
        ids.extend(manifest["ids"][i] for i in keep)
        embs.append(derive_view(emb[keep], stored_grid, tokens))
    return config, ids, np.concatenate(embs)

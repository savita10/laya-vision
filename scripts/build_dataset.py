"""Phase 1 RVL-CDIP converter: stratified split selection -> JSONL records.

Reads only the per-split label files from aharley/rvl_cdip (pinned revision);
no images are downloaded or loaded, for any split. Each source split is
sampled independently (train -> train, val -> val, test -> test), stratified
by the 16 classes, and written as <out-dir>/<split>.jsonl plus
splits_manifest.json. Output is deterministic for a given seed and revision;
re-running overwrites it.

    python scripts/build_dataset.py --out-dir data/rvl_v1
    python scripts/build_dataset.py --out-dir /tmp/rvl_dry --limit 10   # dry run, CPU only
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import rvlcdip

SPLIT_SIZES = {"train": 60_000, "val": 5_000, "test": 5_000}
QUESTION_ID = "doc_type"
QUESTION_INSTRUCTIONS = "What type of document is this?"


def build_questions(label_name):
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": QUESTION_INSTRUCTIONS,
            # Laya choice criteria: label -> description; "" renders the label alone.
            "criteria": {name: "" for name in rvlcdip.CLASSES},
            "answer": label_name,
        }
    }


def split_seed(seed, split):
    # Independent, reproducible stream per split.
    return f"{seed}:{split}"


def select_split(rows, split, n, seed):
    indexed = [(i, path, label) for i, (path, label) in enumerate(rows)]
    picked = rvlcdip.stratified_sample(indexed, key=lambda r: r[2], n=n, seed=split_seed(seed, split))
    picked.sort(key=lambda r: r[0])
    return [
        {
            "id": f"rvl-{split}-{i:06d}",
            "split": split,
            "image_index": i,
            "image_path": path,
            "label_int": label,
            "label_name": rvlcdip.CLASSES[label],
            "questions": build_questions(rvlcdip.CLASSES[label]),
        }
        for i, path, label in picked
    ]


def write_jsonl(records, path):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_git_hash():
    repo = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


def class_counts(labels):
    counts = Counter(labels)
    return {name: counts.get(i, 0) for i, name in enumerate(rvlcdip.CLASSES)}


def main():
    parser = argparse.ArgumentParser(description="Build stratified RVL-CDIP split records")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="dry run: N records per split")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--revision", default=rvlcdip.DATASET_REVISION)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {s: (min(args.limit, n) if args.limit is not None else n) for s, n in SPLIT_SIZES.items()}

    manifest = {
        "dataset": rvlcdip.DATASET_REPO,
        "dataset_revision": args.revision,
        "seed": args.seed,
        "limit": args.limit,
        "requested_sizes": sizes,
        "classes": rvlcdip.CLASSES,
        "question": {"id": QUESTION_ID, "type": "choice", "instructions": QUESTION_INSTRUCTIONS},
        "script_git_hash": script_git_hash(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "splits": {},
    }

    for split, n in sizes.items():
        label_file = rvlcdip.download_label_file(split, args.revision)
        rows = rvlcdip.read_label_file(label_file)
        records = select_split(rows, split, n, args.seed)
        write_jsonl(records, out_dir / f"{split}.jsonl")
        manifest["splits"][split] = {
            "source_file": rvlcdip.SPLIT_FILES[split],
            "source_sha256": sha256(label_file),
            "source_size": len(rows),
            "source_counts_per_class": class_counts(label for _, label in rows),
            "selected": len(records),
            "counts_per_class": class_counts(r["label_int"] for r in records),
        }
        print(f"{split}: {len(records)} of {len(rows)} -> {out_dir / f'{split}.jsonl'}")

    with open(out_dir / "splits_manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Manifest: {out_dir / 'splits_manifest.json'}")


if __name__ == "__main__":
    main()

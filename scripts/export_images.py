"""Export a comparison set's images from RVL-CDIP as PNGs, for local testing (scripts/demo.py).

Streams the RVL-CDIP tar with route_a.py's loader (no full download) and writes, in --out-dir:
  <id>.png      the original scan, decoded and saved losslessly (no resize, no mode change)
  labels.csv    filename,label for every exported image. Labels appear only here, never in
                file names, so the folder can be fed to demo.py without showing the answer.

The comparison set is checked exactly as route_a.py checks it (sha256 of the records file, split).

Test (recorded exception, DECISIONS.md 2026-10-11 "Image export of the test subsample"): test.jsonl
is accepted ONLY with --comparison-set splits/test_comparison_1000_seed17.json, the pre-registered
Route A subsample; the script prints a notice. Test was closed at Gate 4 and no result is computed
here; nothing exported may feed back into training, calibration or any reported number.

Resumable: an image whose PNG exists is skipped; PNGs are written atomically. --limit N exports at
most N new images.

    python scripts/export_images.py --records data/rvl_v1/val.jsonl \\
        --comparison-set splits/val_comparison_2000_seed11.json --out-dir scans/val --limit 32
    python scripts/export_images.py --records data/rvl_v1/test.jsonl \\
        --comparison-set splits/test_comparison_1000_seed17.json --out-dir scans/test
"""

import argparse
import csv
import io
import os
import queue
import threading
from pathlib import Path

from distill_secondary_labels import read_jsonl
from route_a import check_split_and_set, load_comparison_set, read_stream, records_split

TEST_NOTICE = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  TEST SPLIT: exporting the pre-registered 1,000-image test subsample with   !!
!!  its labels (recorded exception, DECISIONS.md 2026-10-11). Local testing    !!
!!  only; test is closed (Gate 4). Labels go to labels.csv only.               !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records", required=True,
                   help="val.jsonl; or test.jsonl with --comparison-set splits/test_comparison_1000_seed17.json")
    p.add_argument("--comparison-set", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="export at most N new images (dry run)")
    p.add_argument("--max-retries", type=int, default=5,
                   help="consecutive tar stream restarts without progress before giving up")
    return p.parse_args()


def save_png(data, path):
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def write_labels(out_dir, selected):
    rows = [(f"{r['id']}.png", r["label_name"]) for r in selected if (out_dir / f"{r['id']}.png").exists()]
    tmp = out_dir / "labels.csv.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filename", "label"])
        w.writerows(rows)
    os.replace(tmp, out_dir / "labels.csv")
    return len(rows)


def main():
    import http.client
    import tarfile
    import zlib

    import requests
    import urllib3

    args = parse_args()
    records_path, comparison_path = Path(args.records), Path(args.comparison_set)
    records = read_jsonl(records_path)
    if not records:
        raise SystemExit(f"{records_path} is empty.")
    split = records_split(records)
    check_split_and_set(split, comparison_path)
    if split == "test":
        print(TEST_NOTICE, flush=True)
    comparison, selected = load_comparison_set(comparison_path, records_path, records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {r["image_path"]: r for r in selected if not (out_dir / f"{r['id']}.png").exists()}
    budget = len(wanted) if args.limit is None else min(args.limit, len(wanted))
    print(f"{len(selected)} in {comparison_path.name}, {len(wanted)} pending, exporting {budget} this run",
          flush=True)

    if budget > 0:
        stream_errors = (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException,
                         tarfile.TarError, zlib.error, EOFError, ConnectionError, TimeoutError)
        images, stop = queue.Queue(maxsize=64), threading.Event()
        reader = threading.Thread(target=read_stream, daemon=True,
                                  args=(wanted, budget, images, stop, stream_errors, args.max_retries))
        reader.start()
        done = 0
        try:
            while True:
                item = images.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                record, data = item
                save_png(data, out_dir / f"{record['id']}.png")
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{budget}", flush=True)
        finally:
            stop.set()
        print(f"Exported {done} images.")

    n = write_labels(out_dir, selected)
    print(f"Wrote {out_dir / 'labels.csv'}: {n} of {comparison['n']} images.")
    if args.limit is None and n < comparison["n"]:
        raise SystemExit(f"{comparison['n'] - n} images were not exported.")


if __name__ == "__main__":
    main()

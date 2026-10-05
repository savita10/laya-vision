"""Save RVL-CDIP images for manual review: a stratified train sample, or chosen ids.

Default: draws a stratified random sample of --n train records and writes
<out>/<id>_<label_name>.png for each, plus <out>/audit_sheet.csv
(id,label_name,looks_correct,notes) for manual review.

--ids / --ids-file: saves exactly the listed records (train or val; test is
refused) instead of a sample, with the sheet named audit_sheet_ids.csv.

Images come from streaming the RVL-CDIP tar (gzip, not seekable), so a full
pass reads up to 38.7 GB over the network but stores nothing else; the
stream stops as soon as every requested image is found. load_dataset is not
used: it rejects script-based datasets such as aharley/rvl_cdip.

With --secondary <split>_secondary.jsonl, the sheet gains the distilled
handwriting answer and a blank column to judge it (written as
audit_sheet_secondary.csv / audit_sheet_ids_secondary.csv), and image names
gain it too: <id>_<label_name>_hw<p_true>.png.

Re-runnable: images already in <out> are not fetched again (a plain-named
copy from an earlier run is reused for the --secondary name), and an existing
sheet is never overwritten (it holds the human's review).

    python scripts/audit_sample.py --records data/rvl_v1/train.jsonl --n 200 --seed 7 --out audit/
    python scripts/audit_sample.py --records data/rvl_v1/val.jsonl --ids rvl-val-000123,rvl-val-004567 \\
        --secondary data/rvl_v1/val_secondary.jsonl --out audit_ids/
    python scripts/audit_sample.py ... --limit 2000   # dry run: stop after 2,000 tar members
"""

import argparse
import csv
import io
import json
import shutil
from pathlib import Path

import rvlcdip


def image_filename(record, secondary_row=None):
    name = f"{record['id']}_{record['label_name'].replace(' ', '_')}"
    if secondary_row is not None:
        if secondary_row.get("questions") is None:
            name += "_ERROR"
        else:
            hw = secondary_row["questions"]["handwriting"]["probabilities"]["true"]
            name += f"_hw{hw:.2f}"
    return name + ".png"


def load_records(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def load_secondary(paths):
    """{id: secondary record} from distill_secondary_labels.py outputs."""
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
    return out


def secondary_cells(row):
    """[handwriting, p_handwriting, handwriting_correct]"""
    if row is None:
        return ["", "", ""]
    if row.get("questions") is None:
        return [f"ERROR: {row.get('error', '')}", "", ""]
    hw = row["questions"]["handwriting"]
    return [str(hw["answer"]).lower(), f"{hw['probabilities']['true']:.3f}", ""]


def parse_ids(args):
    ids = []
    if args.ids:
        ids.extend(i.strip() for i in args.ids.split(","))
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            ids.extend(line.strip() for line in f)
    # Keep the given order, drop blanks and repeats.
    return list(dict.fromkeys(i for i in ids if i))


def main():
    parser = argparse.ArgumentParser(description="Save RVL-CDIP images for manual review")
    parser.add_argument("--records", nargs="+", required=True, help="JSONL file(s) from build_dataset.py")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ids", default=None, help="comma-separated record ids to save instead of a sample")
    parser.add_argument("--ids-file", default=None, help="file with one record id per line")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="dry run: stop after N tar members")
    parser.add_argument("--secondary", nargs="+", default=None,
                        help="<split>_secondary.jsonl file(s); adds their answers to the sheet and image names")
    args = parser.parse_args()

    records = load_records(args.records)
    if any(r["split"] == "test" for r in records):
        raise SystemExit("--records contains test records; the test split is frozen (DECISIONS.md 2026-10-03).")

    ids = parse_ids(args)
    if args.ids is not None or args.ids_file is not None:
        if not ids:
            raise SystemExit("--ids/--ids-file given but no ids found.")
        by_id = {r["id"]: r for r in records}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise SystemExit(f"{len(unknown)} ids not in --records (test ids are never accepted): {unknown[:10]}")
        sample = [by_id[i] for i in ids]
        sheet_stem = "audit_sheet_ids"
    else:
        splits = {r["split"] for r in records}
        if splits != {"train"}:
            raise SystemExit(f"Random sampling needs train records only; found splits {sorted(splits)}.")
        sample = rvlcdip.stratified_sample(records, key=lambda r: r["label_int"], n=args.n, seed=args.seed)
        sample.sort(key=lambda r: r["image_index"])
        sheet_stem = "audit_sheet"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    secondary = load_secondary(args.secondary) if args.secondary else None
    if secondary is not None:
        # Separate sheet, so a completed doc-type sheet is never touched.
        sheet = out / f"{sheet_stem}_secondary.csv"
        header = ["id", "label_name", "looks_correct",
                  "handwriting", "p_handwriting", "handwriting_correct", "notes"]
        missing = [r["id"] for r in sample if r["id"] not in secondary]
        if missing:
            print(f"Warning: {len(missing)} selected records have no secondary labels (blank in sheet), "
                  f"e.g. {missing[:3]}")
        rows = [[r["id"], r["label_name"], ""] + secondary_cells(secondary.get(r["id"])) + [""]
                for r in sample]
    else:
        sheet = out / f"{sheet_stem}.csv"
        header = ["id", "label_name", "looks_correct", "notes"]
        rows = [[r["id"], r["label_name"], "", ""] for r in sample]

    if sheet.exists():
        print(f"{sheet} exists; leaving it untouched.")
    else:
        with open(sheet, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Wrote {sheet} ({len(rows)} rows)")

    targets = {}
    for r in sample:
        target = out / image_filename(r, secondary.get(r["id"]) if secondary is not None else None)
        plain = out / image_filename(r)
        if not target.exists() and plain.exists() and plain != target:
            shutil.copyfile(plain, target)
        if not target.exists():
            targets[r["image_path"]] = target
    if not targets:
        print("All selected images already present.")
        return
    print(f"Streaming {rvlcdip.DATASET_REPO}@{rvlcdip.DATASET_REVISION[:7]} for {len(targets)} images ...")

    from PIL import Image

    found = 0
    for image_path, data in rvlcdip.iter_tar_images(max_members=args.limit):
        target = targets.pop(image_path, None)
        if target is None:
            continue
        tmp = target.with_name(target.name + ".tmp")
        Image.open(io.BytesIO(data)).save(tmp, format="PNG")
        tmp.replace(target)
        found += 1
        if found % 20 == 0:
            print(f"  {found} saved, {len(targets)} to go")
        if not targets:
            break

    print(f"Saved {found} images to {out}")
    if targets:
        print(f"{len(targets)} selected images not found in the streamed members:")
        for path in sorted(targets)[:10]:
            print("  ", path)
        if args.limit is None:
            raise SystemExit("Some selected images are missing from the tar.")


if __name__ == "__main__":
    main()

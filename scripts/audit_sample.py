"""Draw a stratified audit sample from train records and fetch just those images.

Writes <out>/<id>_<label_name>.png for each sampled record and
<out>/audit_sheet.csv (id,label_name,looks_correct,notes) for manual review.
Images come from streaming the RVL-CDIP tar (gzip, not seekable), so a full
pass reads up to 38.7 GB over the network but stores nothing else; the
stream stops as soon as every sampled image is found.

With --secondary <split>_secondary.jsonl, writes audit_sheet_secondary.csv
instead, adding the distilled handwriting/legibility answers and blank
columns to judge each.

Re-runnable: images already in <out> are not fetched again, and an existing
sheet is never overwritten (it holds the human's review).

    python scripts/audit_sample.py --records data/rvl_v1/train.jsonl --n 200 --seed 7 --out audit/
    python scripts/audit_sample.py ... --limit 2000   # dry run: stop after 2,000 tar members
"""

import argparse
import csv
import io
import json
from pathlib import Path

import rvlcdip


def image_filename(record):
    return f"{record['id']}_{record['label_name'].replace(' ', '_')}.png"


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
    """[handwriting, p_handwriting, handwriting_correct, legibility, legibility_correct]"""
    if row is None:
        return ["", "", "", "", ""]
    if row.get("questions") is None:
        return [f"ERROR: {row.get('error', '')}", "", "", "", ""]
    hw = row["questions"]["handwriting"]
    lg = row["questions"]["legibility"]
    level = lg["criteria"][lg["answer"]].split(":")[0]
    return [str(hw["answer"]).lower(), f"{hw['probabilities']['true']:.3f}", "",
            f"{lg['answer']} ({level})", ""]


def main():
    parser = argparse.ArgumentParser(description="Stratified audit sample of RVL-CDIP train records")
    parser.add_argument("--records", required=True, help="train JSONL from build_dataset.py")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="dry run: stop after N tar members")
    parser.add_argument("--secondary", nargs="+", default=None,
                        help="<split>_secondary.jsonl file(s); writes audit_sheet_secondary.csv with their answers")
    args = parser.parse_args()

    with open(args.records, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    splits = {r["split"] for r in records}
    if splits != {"train"}:
        raise SystemExit(f"--records must contain only train records; found splits {sorted(splits)}.")

    sample = rvlcdip.stratified_sample(records, key=lambda r: r["label_int"], n=args.n, seed=args.seed)
    sample.sort(key=lambda r: r["image_index"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.secondary:
        # Separate sheet, so a completed doc-type audit_sheet.csv is never touched.
        sheet = out / "audit_sheet_secondary.csv"
        header = ["id", "label_name", "looks_correct",
                  "handwriting", "p_handwriting", "handwriting_correct",
                  "legibility", "legibility_correct", "notes"]
        secondary = load_secondary(args.secondary)
        missing = [r["id"] for r in sample if r["id"] not in secondary]
        if missing:
            print(f"Warning: {len(missing)} sampled records have no secondary labels (blank in sheet), "
                  f"e.g. {missing[:3]}")
        rows = [[r["id"], r["label_name"], ""] + secondary_cells(secondary.get(r["id"])) + [""]
                for r in sample]
    else:
        sheet = out / "audit_sheet.csv"
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

    wanted = {r["image_path"]: r for r in sample if not (out / image_filename(r)).exists()}
    if not wanted:
        print("All sampled images already present.")
        return
    print(f"Streaming {rvlcdip.DATASET_REPO}@{rvlcdip.DATASET_REVISION[:7]} for {len(wanted)} images ...")

    from PIL import Image

    found = 0
    for image_path, data in rvlcdip.iter_tar_images(max_members=args.limit):
        record = wanted.pop(image_path, None)
        if record is None:
            continue
        target = out / image_filename(record)
        tmp = target.with_name(target.name + ".tmp")
        Image.open(io.BytesIO(data)).save(tmp, format="PNG")
        tmp.replace(target)
        found += 1
        if found % 20 == 0:
            print(f"  {found} saved, {len(wanted)} to go")
        if not wanted:
            break

    print(f"Saved {found} images to {out}")
    if wanted:
        print(f"{len(wanted)} sampled images not found in the streamed members:")
        for path in sorted(wanted)[:10]:
            print("  ", path)
        if args.limit is None:
            raise SystemExit("Some sampled images are missing from the tar.")


if __name__ == "__main__":
    main()

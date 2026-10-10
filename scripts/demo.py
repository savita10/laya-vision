"""Classify a folder of document images with a Laya-Vision release.

    python scripts/demo.py --images ./my_scans --release release/ --out results.json
    python scripts/demo.py --images ./my_scans --release release/ --out results.json --with-ocr

Prints, per image, the predicted document type among the 16 RVL-CDIP classes and its calibrated
probability; writes every class probability to --out. Image-only by default; --with-ocr adds the
Tesseract text (the combined view; needs pytesseract and the tesseract binary). An image that
cannot be decoded is reported as an error for that file, never classified from text alone.

Resumable: images already in --out with a result are skipped (delete --out to start again).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import laya_vision  # noqa: F401  (pip install -e .)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from laya_vision import ImageError, VisionAgent

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", required=True, help="folder of document images (searched recursively)")
    p.add_argument("--release", default="release", help="release folder or Hub repo id")
    p.add_argument("--out", default="results.json")
    p.add_argument("--with-ocr", action="store_true", help="combined view: image + Tesseract OCR text")
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: upstream laya's choice)")
    p.add_argument("--limit", type=int, default=None, help="classify at most N images (dry run)")
    return p.parse_args()


def write_json_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def main():
    args = parse_args()
    folder = Path(args.images)
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder.")
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no images ({', '.join(sorted(EXTENSIONS))}) under {folder}.")

    out = Path(args.out)
    results = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            results = {r["file"]: r for r in json.load(f)["results"] if "doc_type" in r}
    todo = [p for p in files if p.relative_to(folder).as_posix() not in results]
    print(f"{len(files)} images, {len(files) - len(todo)} already in {out}; classifying {len(todo)}.")

    agent = VisionAgent(release_dir=args.release, device=args.device)
    questions = agent.doc_type_question()
    view = "combined" if args.with_ocr else "image"
    meta = {"release": str(args.release), "checkpoint": agent.config["checkpoint"], "view": view,
            "temperature": agent.release.temperatures[view], "device": str(agent.device)}
    errors = 0
    for i, path in enumerate(todo, 1):
        rel = path.relative_to(folder).as_posix()
        t0 = time.perf_counter()
        try:
            res = agent.predict(questions=questions, image=path, ocr=args.with_ocr)
        except ImageError as e:
            errors += 1
            results[rel] = {"file": rel, "error": str(e)}
            print(f"{rel}: ERROR {e}")
            continue
        ans = res["answers"]["doc_type"]
        results[rel] = {"file": rel, "doc_type": ans["choice"], "probability": ans["probabilities"][ans["choice"]],
                        "probabilities": ans["probabilities"], "view": res["usage"]["view"],
                        "truncated": res["usage"]["truncated"],
                        "ms": round((time.perf_counter() - t0) * 1000, 1)}
        print(f"{rel}: {ans['choice']} ({ans['probabilities'][ans['choice']]:.3f})")
        if i % 50 == 0:
            write_json_atomic(out, {"meta": meta, "results": list(results.values())})
    write_json_atomic(out, {"meta": meta, "results": [results[k] for k in sorted(results)]})
    print(f"Wrote {out}: {len(results)} images, {errors} errors.")


if __name__ == "__main__":
    sys.exit(main())

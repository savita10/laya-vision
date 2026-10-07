"""Benchmark the two precompute passes separately on a few val images.

Streams the first --limit (default 64) val images found in the RVL-CDIP tar,
then times, with the exact code precompute.py runs:
  embed  decode + letterbox (CPU) and SigLIP + pooling (GPU), in batches
  ocr    each requested engine on the original full-resolution images
For each pass: img/s, peak GPU memory, and projected hours for --project
images (default 70,000 = train + val + test). Projections cover compute
only; each split also needs one tar pass (download time).

Val only: no test images are read here.

    python scripts/bench_precompute.py --records data/rvl_v1/val.jsonl
    python scripts/bench_precompute.py --records data/rvl_v1/val.jsonl --ocr-engine tesseract --out bench_p2.json
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import precompute
import rvlcdip


def fetch_images(records, n, max_retries=3):
    """First n records found in tar order (bench only; no spool, no resume)."""
    wanted = {r["image_path"]: r for r in records}
    errors = precompute.stream_errors()
    for attempt in range(max_retries + 1):
        got = []
        try:
            for image_path, data in rvlcdip.iter_tar_images():
                if image_path in wanted:
                    got.append((wanted[image_path], data))
                    if len(got) == n:
                        return got
            return got
        except errors as e:
            if attempt == max_retries:
                raise
            print(f"Stream error ({type(e).__name__}: {e}); retrying", flush=True)
            time.sleep(10)


def gpu_peak(torch):
    if not torch.cuda.is_available():
        return None
    return {"allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}


def reset_gpu(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def project(rate, n):
    return round(n / rate / 3600, 2) if rate else None


def bench_embed(args, raw, torch):
    reset_gpu(torch)
    embedder = precompute.Embedder(args.device, args.dtype, precompute.parse_grid(args.pool_grid))
    load_gb = gpu_peak(torch)
    # Warm-up batch (cudnn autotune, allocator), not timed.
    embedder([precompute.letterbox(precompute.decode(raw[0][1]))] * min(args.batch_size, len(raw)))
    sync(torch)
    reset_gpu(torch)
    t_pre = t_gpu = 0.0
    for start in range(0, len(raw), args.batch_size):
        t0 = time.perf_counter()
        batch = [precompute.letterbox(precompute.decode(data)) for _, data in raw[start:start + args.batch_size]]
        t1 = time.perf_counter()
        out = embedder(batch)
        sync(torch)
        t_pre += t1 - t0
        t_gpu += time.perf_counter() - t1
    n = len(raw)
    total = t_pre + t_gpu
    return {
        "images": n,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "output_shape_per_image": list(out.shape[1:]),
        "preprocess_img_per_s": round(n / t_pre, 2),
        "siglip_img_per_s": round(n / t_gpu, 2),
        "end_to_end_img_per_s": round(n / total, 2),
        "peak_gpu_after_load": load_gb,
        "peak_gpu_during_embed": gpu_peak(torch),
        "projected_hours": project(n / total, args.project),
    }


def bench_ocr(args, engine, raw, torch):
    reset_gpu(torch)
    try:
        ocr = precompute.make_ocr(engine, args.ocr_workers)
    except (ImportError, SystemExit) as e:
        return {"skipped": f"{type(e).__name__}: {e}"}
    images = [precompute.decode(data) for _, data in raw]
    ocr(images[:1])  # warm-up (model load / first process spawn), not timed
    sync(torch)
    reset_gpu(torch)
    t0 = time.perf_counter()
    results = ocr(images)
    sync(torch)
    seconds = time.perf_counter() - t0
    rate = len(images) / seconds
    confs = [w[5] for r in results for w in r["words"] if w[5] >= 0]
    return {
        "spec": ocr.spec(),
        "images": len(images),
        "workers": getattr(ocr, "workers", 1),
        "cpu_count": os.cpu_count(),
        "img_per_s": round(rate, 3),
        "seconds_per_image_per_worker": round(seconds * getattr(ocr, "workers", 1) / len(images), 3),
        "peak_gpu": gpu_peak(torch) if engine == "easyocr" else None,
        "projected_hours": project(rate, args.project),
        "words_per_image_mean": round(statistics.mean(len(r["words"]) for r in results), 1),
        "word_confidence_mean": round(statistics.mean(confs), 1) if confs else None,
        "sample": {"id": raw[0][0]["id"], "text_head": results[0]["text"][:300]},
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark precompute passes (embed / OCR) separately")
    parser.add_argument("--records", required=True, help="val.jsonl")
    parser.add_argument("--limit", type=int, default=64, help="images to benchmark")
    parser.add_argument("--ocr-engine", choices=precompute.OCR_ENGINES + ("all", "none"), default="all")
    parser.add_argument("--ocr-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pool-grid", default=precompute.DEFAULT_POOL_GRID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--project", type=int, default=70000, help="images to project hours for")
    parser.add_argument("--out", default=None, help="write results JSON here")
    args = parser.parse_args()

    import torch

    split, records = precompute.load_records(Path(args.records))
    if split != "val":
        raise SystemExit("The benchmark runs on val only.")
    print(f"Streaming the first {args.limit} val images from the tar ...", flush=True)
    t0 = time.perf_counter()
    raw = fetch_images(records, args.limit)
    print(f"  got {len(raw)} images in {time.perf_counter() - t0:.0f}s", flush=True)
    sizes = [precompute.decode(d).size for _, d in raw]

    results = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "script_git_hash": precompute.script_git_hash(),
        "date": precompute.now(),
        "image_sizes": {"min": [min(s[0] for s in sizes), min(s[1] for s in sizes)],
                        "max": [max(s[0] for s in sizes), max(s[1] for s in sizes)]},
        "project_images": args.project,
    }
    if not args.skip_embed:
        print("Benchmarking embed pass ...", flush=True)
        results["embed"] = bench_embed(args, raw, torch)
    engines = {"all": precompute.OCR_ENGINES, "none": ()}.get(args.ocr_engine, (args.ocr_engine,))
    results["ocr"] = {}
    for engine in engines:
        print(f"Benchmarking OCR: {engine} ...", flush=True)
        results["ocr"][engine] = bench_ocr(args, engine, raw, torch)

    print()
    print(f"GPU {results['gpu']}, {results['cpu_count']} CPUs, {len(raw)} val images "
          f"(sizes {results['image_sizes']['min']}..{results['image_sizes']['max']})")
    if "embed" in results:
        e = results["embed"]
        print(f"embed: {e['end_to_end_img_per_s']} img/s end-to-end (preprocess {e['preprocess_img_per_s']}, "
              f"SigLIP {e['siglip_img_per_s']}), peak GPU {e['peak_gpu_during_embed']}, "
              f"projected {e['projected_hours']} h for {args.project}")
    for engine, r in results["ocr"].items():
        if "skipped" in r:
            print(f"ocr {engine}: skipped ({r['skipped']})")
            continue
        print(f"ocr {engine} {r['spec']['version']}: {r['img_per_s']} img/s with {r['workers']} worker(s) "
              f"({r['seconds_per_image_per_worker']} s/img/worker), peak GPU {r['peak_gpu']}, "
              f"projected {r['projected_hours']} h for {args.project}; "
              f"{r['words_per_image_mean']} words/img, mean conf {r['word_confidence_mean']}")
        print(f"  sample {r['sample']['id']}: {r['sample']['text_head'][:200]!r}")
    print("Projections are compute only; add one tar pass per split (download time).")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

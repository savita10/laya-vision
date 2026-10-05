"""Measure distillation throughput and batching headroom on a few val images.

Collects --n val images from the RVL-CDIP tar stream (the first ones the
stream reaches, or exactly --ids), holds them in memory, then times the real
labelling path of distill_secondary_labels.py (single encode, last-position
logits) with no streaming in the loop. Writes no labels.

Reports:
  - img/s for the batch-1 labelling path, with GPU peak memory
  - a batching probe: the prefix pass (vision encoder + LM over the image,
    where nearly all the compute is) run on B images at once, for each
    --probe-batches size, against the same pass at batch 1. OOM is caught
    and reported. The probe uses padded batches and does not check answers;
    it only measures memory and speed.
  - projected hours for --project-images images at the measured rate
    (inference only; one tar pass adds on top)

Pixel limits default to the distillation defaults (mid-res). Test records are
refused.

    python scripts/bench_distill.py --records data/rvl_v1/val.jsonl --n 32
"""

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import distill_secondary_labels as distill
import rvlcdip


def collect_images(records, n, ids):
    """[(record, PIL image)] for the first n records the tar stream reaches (or exactly ids)."""
    from PIL import Image

    if ids is not None:
        by_id = {r["id"]: r for r in records}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise SystemExit(f"{len(unknown)} ids not in --records: {unknown[:10]}")
        wanted = {by_id[i]["image_path"]: by_id[i] for i in ids}
        n = len(wanted)
    else:
        wanted = {r["image_path"]: r for r in records}
    out = []
    for image_path, data in rvlcdip.iter_tar_images():
        record = wanted.pop(image_path, None)
        if record is None:
            continue
        try:
            out.append((record, Image.open(io.BytesIO(data)).convert("RGB")))
        except Exception as e:  # corrupt TIFFs exist in RVL-CDIP
            print(f"  skipping {record['id']}: decode {type(e).__name__}: {e}")
            continue
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"Only {len(out)} of {n} images found in the tar stream.")
    return out


def time_labelling(distiller, images, torch):
    times, tokens = [], []
    for _, img in images:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = distiller(img)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        tokens.append(result["image"]["visual_tokens"])
    return times, tokens


def time_prefix(distiller, imgs, torch):
    """Seconds for one prefix pass (use_cache=True) over imgs as a single batch."""
    inputs = distiller.processor(text=[distiller.prefix_text] * len(imgs), images=imgs,
                                 padding=True, return_tensors="pt").to(distiller.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        distiller._forward(**inputs, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def probe_batch(distiller, images, size, torch):
    """img/s and peak GiB for prefix passes at this batch size over the collected images."""
    imgs = [img for _, img in images]
    groups = [imgs[i:i + size] for i in range(0, len(imgs) - size + 1, size)]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        time_prefix(distiller, groups[0], torch)  # warm-up for this shape
        seconds = sum(time_prefix(distiller, g, torch) for g in groups)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    return {"img_s": len(groups) * size / seconds, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}


def main():
    parser = argparse.ArgumentParser(description="Measure distillation throughput")
    parser.add_argument("--records", required=True, help="val.jsonl from build_dataset.py")
    parser.add_argument("--n", type=int, default=32, help="images to time")
    parser.add_argument("--ids", default=None, help="comma-separated record ids to time instead")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--min-pixels", type=int, default=distill.DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=distill.DEFAULT_MAX_PIXELS)
    parser.add_argument("--warmup", type=int, default=2, help="untimed images first")
    parser.add_argument("--probe-batches", default="4,8", help="batch sizes for the prefix-pass probe")
    parser.add_argument("--project-images", type=int, default=60000)
    parser.add_argument("--json-out", default=None, help="also write the results here")
    args = parser.parse_args()
    distill.check_transformers_version()

    records = distill.read_jsonl(Path(args.records))
    if any(r["split"] == "test" for r in records):
        raise SystemExit("--records contains test records; benchmark on val.")
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None

    import torch

    print(f"Streaming for {args.n if ids is None else len(ids)} images (+{args.warmup} warm-up) ...", flush=True)
    t0 = time.time()
    images = collect_images(records, (args.n + args.warmup) if ids is None else None, ids)
    print(f"  collected in {time.time() - t0:.0f}s", flush=True)
    warm = images[:args.warmup]
    # With --ids every listed image is timed; warm-up reuses the first few of them.
    timed = images[args.warmup:] if ids is None else images

    distiller = distill.Distiller(args.device, args.dtype, args.min_pixels, args.max_pixels)
    gpu = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    for _, img in warm:
        distiller(img)

    torch.cuda.reset_peak_memory_stats()
    times, tokens = time_labelling(distiller, timed, torch)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    rate = len(times) / sum(times)
    hours = args.project_images / rate / 3600

    print(f"\nGPU {gpu} ({total_gib:.1f} GiB), transformers {distill.TRANSFORMERS_VERSION}, "
          f"pixels {args.min_pixels}-{args.max_pixels}, {args.dtype}")
    print(f"Labelling path, batch 1, {len(times)} images:")
    print(f"  {rate:.3f} img/s  (median {statistics.median(times):.3f}s, "
          f"min {min(times):.3f}s, max {max(times):.3f}s per image)")
    print(f"  visual tokens per image: mean {statistics.mean(tokens):.0f}, min {min(tokens)}, max {max(tokens)}")
    print(f"  GPU peak memory: {peak:.2f} GiB of {total_gib:.1f}")
    print(f"  projected {args.project_images} images: {hours:.1f} h inference (+ one tar pass)")

    sizes = [int(s) for s in args.probe_batches.split(",") if s.strip()]
    probes = {1: probe_batch(distiller, timed, 1, torch)}
    for size in sizes:
        if size > len(timed):
            print(f"  batch {size}: skipped (only {len(timed)} images)")
            continue
        probes[size] = probe_batch(distiller, timed, size, torch)
    print("Prefix-pass batching probe:")
    for size, r in probes.items():
        if r is None:
            print(f"  batch {size}: OOM")
        else:
            speedup = r["img_s"] / probes[1]["img_s"] if probes[1] else float("nan")
            print(f"  batch {size}: {r['img_s']:.3f} img/s ({speedup:.2f}x batch 1), "
                  f"peak {r['peak_gib']:.2f} GiB")

    if args.json_out:
        result = {
            "gpu": gpu, "gpu_gib": total_gib, "transformers": distill.TRANSFORMERS_VERSION,
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels, "dtype": args.dtype,
            "images": [r["id"] for r, _ in timed], "seconds": times, "visual_tokens": tokens,
            "img_s": rate, "peak_gib": peak, "projected_hours": hours,
            "project_images": args.project_images,
            "prefix_batch_probe": {str(k): v for k, v in probes.items()},
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

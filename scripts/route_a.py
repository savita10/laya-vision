"""Route A baseline: VLM page description -> existing Laya, zero-shot, no training.

Qwen2-VL-2B-Instruct writes a neutral layout/content description of each page
(the prompt never mentions the 16 classes), the description becomes the Laya
state, and the existing Gate-0 fine-tuned Laya checkpoint (--laya-dir) answers
the 16-class document-type choice. No model is trained for this baseline
(DECISIONS.md 2026-10-05).

Evaluation runs on the FIXED comparison set: a stratified 2,000-image val
subset, seed 11, stored in splits/val_comparison_2000_seed11.json. Laya-Vision
is scored on the same ids later. The file records the sha256 of the val.jsonl
it was drawn from, and this script refuses a different val.jsonl.

Test (recorded exception, DECISIONS.md 2026-10-10 "Route A test subsample";
docs/phase4_protocol.md): Route A is the pre-registered external baseline on
the pre-registered test subsample, a stratified 1,000-image test subset, seed
17, in splits/test_comparison_1000_seed17.json (sha256-guarded like the val
set). --records test.jsonl is accepted ONLY with --comparison-set pointing at
that file; this script then reads test labels to draw the subsample and to
score it, prints a notice, and runs once: a complete test result in --out-dir
refuses further runs (--results-only still recomputes it). Laya-Vision's test
numbers come from final_eval.py only.

Outputs, in --out-dir:
  route_a_outputs.jsonl    one line per image: description, Laya answer, timings
  route_a.done.log         ids whose output line is durably written (resume state)
  route_a.manifest.json    VLM, revision, transformers, prompt, decoding, pixels,
                           Laya version and checkpoint hashes, comparison set hash
  route_a_results.json     accuracy vs labels, majority-class floor, latency
                           p50/p95 for VLM and Laya, GPU-seconds/image, projection
A resumed run must match the manifest, so outputs from two configurations
never mix. route_a_results.json is rewritten after every run; it is marked
partial until all 2,000 are done.

Timings are wall-clock seconds per stage on a dedicated GPU, synchronised
around the stage: VLM = preprocessing + generation, Laya = predict(). They
are reported as latency (p50/p95) and as GPU-seconds/image (mean).

    python scripts/route_a.py --records data/rvl_v1/val.jsonl --make-comparison-set   # once; commit it
    python scripts/route_a.py --records data/rvl_v1/val.jsonl --laya-dir <ckpt> --out-dir runs/route_a --limit 32
    python scripts/route_a.py --records data/rvl_v1/val.jsonl --laya-dir <ckpt> --out-dir runs/route_a
    python scripts/route_a.py --records data/rvl_v1/val.jsonl --out-dir runs/route_a --results-only
    python scripts/route_a.py --records data/rvl_v1/test.jsonl --comparison-set splits/test_comparison_1000_seed17.json \
        --make-comparison-set                                                          # once; commit it
    python scripts/route_a.py --records data/rvl_v1/test.jsonl --comparison-set splits/test_comparison_1000_seed17.json \
        --laya-dir <ckpt> --out-dir runs/route_a_test                                  # the one test run
"""

import argparse
import hashlib
import io
import json
import queue
import re
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

import rvlcdip
from distill_secondary_labels import (MODEL_ID, MODEL_REVISION, TRANSFORMERS_VERSION, check_transformers_version,
                                      fsync_append, read_jsonl, write_jsonl_atomic)

REPO = Path(__file__).resolve().parent.parent
COMPARISON_SEED = 11
COMPARISON_N = 2000
DEFAULT_COMPARISON_SET = REPO / "splits" / f"val_comparison_{COMPARISON_N}_seed{COMPARISON_SEED}.json"
# Pre-registered test subsample (docs/phase4_protocol.md; DECISIONS.md 2026-10-10).
TEST_COMPARISON_SEED = 17
TEST_COMPARISON_N = 1000
TEST_COMPARISON_SET = REPO / "splits" / f"test_comparison_{TEST_COMPARISON_N}_seed{TEST_COMPARISON_SEED}.json"
TEST_NOTICE = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  TEST SPLIT: Route A reads TEST LABELS for the pre-registered 1,000-image   !!
!!  test subsample (recorded exception, DECISIONS.md 2026-10-10). Runs once.   !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

# Neutral: asks for layout and content, never for a document type, and names no class.
# a1: first Route A prompt (never run).
# a2: "whether the text is printed, typed or handwritten" -> "how the text was
#     produced"; "handwritten" is a class label (DECISIONS.md 2026-10-05).
PROMPT_VERSION = "a2"
DESCRIBE_PROMPT = (
    "Describe this scanned page for someone who cannot see it. Cover its layout (for example "
    "columns, headings, tables, boxes, lines, logos, pictures or charts), how the text was "
    "produced, and what the text is about, quoting any title or heading. "
    "Describe only what is visible. Write plain prose, at most 120 words."
)
_leaked = [c for c in rvlcdip.CLASSES if re.search(rf"\b{re.escape(c)}\b", DESCRIBE_PROMPT, re.IGNORECASE)]
if _leaked:
    raise RuntimeError(f"DESCRIBE_PROMPT names class labels {_leaked}; Route A would become a VLM classifier.")
MAX_NEW_TOKENS = 192  # ~120 words plus headroom; hitting the cap is counted in the results
# Native resolution: the processor's own floor, and a cap above RVL-CDIP's
# ~1000x754 scans, so pages are not resized (only rounded to 28 px).
NATIVE_MIN_PIXELS = 56 * 56
NATIVE_MAX_PIXELS = 1024 * 28 * 28

QUESTION_ID = "doc_type"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- comparison set -------------------------------------------------------------

def records_split(records):
    splits = {r["split"] for r in records}
    if splits not in ({"val"}, {"test"}):
        raise SystemExit(f"--records must be val.jsonl or test.jsonl; found splits {sorted(splits)}.")
    return splits.pop()


def check_split_and_set(split, comparison_path):
    """Test records only with the pre-registered test subsample; val never with it."""
    is_test_set = comparison_path.resolve() == TEST_COMPARISON_SET.resolve()
    if split == "test" and not is_test_set:
        raise SystemExit(f"--records test.jsonl is allowed only with --comparison-set {TEST_COMPARISON_SET} "
                         "(the pre-registered test subsample; DECISIONS.md 2026-10-10).")
    if split == "val" and is_test_set:
        raise SystemExit(f"{comparison_path} is the test subsample; it needs --records test.jsonl.")


def make_comparison_set(records_path, records, out_path, split="val"):
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; the comparison set is fixed and is never regenerated.")
    n, seed = (TEST_COMPARISON_N, TEST_COMPARISON_SEED) if split == "test" else (COMPARISON_N, COMPARISON_SEED)
    picked = rvlcdip.stratified_sample(records, key=lambda r: r["label_int"], n=n, seed=seed)
    picked.sort(key=lambda r: r["image_index"])
    counts = Counter(r["label_name"] for r in picked)
    description = ("Pre-registered Route A / Laya-Vision test subsample (docs/phase4_protocol.md; DECISIONS.md "
                   "2026-10-10). Never regenerate." if split == "test" else
                   "Fixed Route A / Laya-Vision comparison set (DECISIONS.md 2026-10-05). Never regenerate.")
    data = {
        "description": description,
        "split": split,
        "source": f"{split}.jsonl from scripts/build_dataset.py",
        "source_sha256": sha256_file(records_path),
        "seed": seed,
        "n": len(picked),
        "counts_per_class": {name: counts.get(name, 0) for name in rvlcdip.CLASSES},
        "ids": [r["id"] for r in picked],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {out_path}: {len(picked)} ids, {dict(counts)}")


def load_comparison_set(path, records_path, records):
    if not path.exists():
        raise SystemExit(f"{path} not found; create it once with --make-comparison-set and commit it.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    actual = sha256_file(records_path)
    if actual != data["source_sha256"]:
        raise SystemExit(f"{records_path} has sha256 {actual}, but the comparison set was drawn from "
                         f"{data['source_sha256']}; refusing to score a different split file.")
    split = data.get("split", "val")  # the val set predates the field
    if any(r["split"] != split for r in records):
        raise SystemExit(f"{path} is a {split} comparison set; {records_path} is not {split}.jsonl.")
    by_id = {r["id"]: r for r in records}
    missing = [i for i in data["ids"] if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} comparison ids not in {records_path}: {missing[:5]}")
    return data, [by_id[i] for i in data["ids"]]


# --- models ---------------------------------------------------------------------

class Describer:
    """Greedy page description with Qwen2-VL-2B at native resolution."""

    def __init__(self, device, dtype):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=rvlcdip.hf_token(),
            min_pixels=NATIVE_MIN_PIXELS, max_pixels=NATIVE_MAX_PIXELS)
        ip = self.processor.image_processor
        ip.min_pixels, ip.max_pixels = NATIVE_MIN_PIXELS, NATIVE_MAX_PIXELS
        if isinstance(getattr(ip, "size", None), dict):
            ip.size = {**ip.size, "shortest_edge": NATIVE_MIN_PIXELS, "longest_edge": NATIVE_MAX_PIXELS}
        self.patch_size = getattr(ip, "patch_size", 14)
        self.merge_size = getattr(ip, "merge_size", 2)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch_dtype, token=rvlcdip.hf_token(),
        ).to(device).eval()
        # Generation reads only the last position's logits; 4.49 has no
        # logits_to_keep, so drop the rest before the LM head (as in distillation).
        self.model.get_output_embeddings().register_forward_pre_hook(
            lambda module, args: (args[0][:, -1:, :],) + tuple(args[1:]))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DESCRIBE_PROMPT}]}]
        self.text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def __call__(self, image):
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        inputs = self.processor(text=[self.text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()
        t, h, w = (int(x) for x in inputs["image_grid_thw"][0])
        return {
            "description": text,
            "description_tokens": int(new.shape[0]),
            "visual_tokens": t * h * w // (self.merge_size ** 2),
            "model_size": [w * self.patch_size, h * self.patch_size],
            "seconds": seconds,
        }


def laya_checkpoint_hashes(laya_dir):
    d = Path(laya_dir)
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"})
    if not files:
        raise SystemExit(f"No checkpoint files under {laya_dir}.")
    return {p.relative_to(d).as_posix(): sha256_file(p) for p in files}


def laya_questions(record):
    q = dict(record["questions"][QUESTION_ID])
    q.pop("answer", None)
    return {QUESTION_ID: q}


# --- outputs --------------------------------------------------------------------

class Outputs:
    """route_a_outputs.jsonl + done.log, reconciled on start (as in distillation)."""

    def __init__(self, out_dir, manifest):
        self.out = out_dir / "route_a_outputs.jsonl"
        self.done_log = out_dir / "route_a.done.log"
        manifest_path = out_dir / "route_a.manifest.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                if json.load(f) != manifest:
                    raise SystemExit(f"{manifest_path} was written with a different configuration; "
                                     "refusing to mix outputs. Use the original settings or a new --out-dir.")
        else:
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                f.write("\n")
        done = set()
        if self.done_log.exists():
            with open(self.done_log, encoding="utf-8") as f:
                done = {line.strip() for line in f if line.strip()}
        rows, seen = [], set()
        for row in read_jsonl(self.out):
            if row["id"] in done and row["id"] not in seen:
                rows.append(row)
                seen.add(row["id"])
        if seen != done:
            with open(self.done_log, "w", encoding="utf-8", newline="\n") as f:
                f.writelines(i + "\n" for i in sorted(seen))
        write_jsonl_atomic(self.out, rows)
        self.done = seen

    def commit(self, rows):
        fsync_append(self.out, [json.dumps(r, ensure_ascii=False) for r in rows])
        fsync_append(self.done_log, [r["id"] for r in rows])
        self.done.update(r["id"] for r in rows)

    def sort(self):
        write_jsonl_atomic(self.out, sorted(read_jsonl(self.out), key=lambda r: r["image_index"]))


# --- results --------------------------------------------------------------------

def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarise(values):
    if not values:
        return None
    return {"p50_ms": 1000 * percentile(values, 0.5), "p95_ms": 1000 * percentile(values, 0.95),
            "mean_ms": 1000 * statistics.mean(values)}


def compute_results(rows, comparison, manifest):
    n_set = comparison["n"]
    ok = [r for r in rows if r.get("error") is None]
    errors = [r for r in rows if r.get("error") is not None]
    correct = sum(r["correct"] for r in ok)
    # A property of the fixed set, so partial runs report the same floor.
    majority_class, majority_count = Counter(comparison["counts_per_class"]).most_common(1)[0]
    per_class = {}
    for name in rvlcdip.CLASSES:
        mine = [r for r in rows if r["label_name"] == name]
        if mine:
            per_class[name] = {"n": len(mine), "accuracy": sum(r.get("correct", False) for r in mine) / len(mine)}
    vlm = [r["vlm_seconds"] for r in ok]
    laya_s = [r["laya_seconds"] for r in ok]
    total = [a + b for a, b in zip(vlm, laya_s)]
    mean_total = statistics.mean(total) if total else None
    return {
        "complete": len(rows) == n_set,
        "n_comparison_set": n_set,
        "n_done": len(rows),
        "n_errors": len(errors),
        "errors": [{"id": r["id"], "error": r["error"]} for r in errors],
        # Errors count as wrong: accuracy is over every image attempted.
        "accuracy": correct / len(rows) if rows else None,
        "correct": correct,
        "majority_class_floor": {"class": majority_class, "accuracy": majority_count / n_set},
        "per_class_accuracy": per_class,
        "latency": {"vlm": summarise(vlm), "laya": summarise(laya_s), "total": summarise(total)},
        "gpu_seconds_per_image": {
            "vlm": statistics.mean(vlm) if vlm else None,
            "laya": statistics.mean(laya_s) if laya_s else None,
            "total": mean_total,
        },
        "img_per_s": 1 / mean_total if mean_total else None,
        "projected_hours_full_set": n_set * mean_total / 3600 if mean_total else None,
        "description_tokens": {
            "mean": statistics.mean(r["description_tokens"] for r in ok) if ok else None,
            "hit_max_new_tokens": sum(r["description_tokens"] >= MAX_NEW_TOKENS for r in ok),
        },
        "laya_state_truncated": sum(bool(r["laya"].get("truncated")) for r in ok),
        "gpus": sorted({r["gpu"] for r in ok}),
        "config": manifest,
    }


def write_results(out_dir, results):
    path = out_dir / "route_a_results.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)
    return path


def print_results(results):
    g = results["gpu_seconds_per_image"]
    lat = results["latency"]
    print(f"\n{results['n_done']}/{results['n_comparison_set']} done"
          + ("" if results["complete"] else " (partial)") + f", {results['n_errors']} errors, GPUs {results['gpus']}")
    if results["accuracy"] is not None:
        print(f"  accuracy {results['accuracy']:.4f}  (majority-class floor "
              f"{results['majority_class_floor']['accuracy']:.4f}, {results['majority_class_floor']['class']})")
    if g["total"] is not None:
        print(f"  VLM  p50 {lat['vlm']['p50_ms']:.0f} ms, p95 {lat['vlm']['p95_ms']:.0f} ms")
        print(f"  Laya p50 {lat['laya']['p50_ms']:.0f} ms, p95 {lat['laya']['p95_ms']:.0f} ms")
        print(f"  GPU-seconds/image {g['total']:.2f} (VLM {g['vlm']:.2f} + Laya {g['laya']:.3f}) "
              f"= {results['img_per_s']:.3f} img/s")
        print(f"  projected {results['n_comparison_set']} images: {results['projected_hours_full_set']:.1f} h "
              "(+ one tar pass)")
        print(f"  description tokens mean {results['description_tokens']['mean']:.0f}, "
              f"{results['description_tokens']['hit_max_new_tokens']} hit the {MAX_NEW_TOKENS}-token cap; "
              f"Laya state truncated {results['laya_state_truncated']}")


# --- main -----------------------------------------------------------------------

def build_manifest(args, comparison, laya_hashes):
    from importlib.metadata import version

    return {
        "vlm": MODEL_ID,
        "vlm_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt": DESCRIBE_PROMPT,
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "min_pixels": NATIVE_MIN_PIXELS,
        "max_pixels": NATIVE_MAX_PIXELS,
        "dtype": args.dtype,
        "laya_version": version("laya"),
        "laya_checkpoint_sha256": laya_hashes,
        "state": "description text (plain string)",
        "question": QUESTION_ID,
        "comparison_set": {"split": comparison.get("split", "val"), "seed": comparison["seed"],
                           "n": comparison["n"], "source_sha256": comparison["source_sha256"]},
        "dataset": rvlcdip.DATASET_REPO,
        "dataset_revision": rvlcdip.DATASET_REVISION,
    }


def main():
    parser = argparse.ArgumentParser(description="Route A baseline: describe -> existing Laya, zero-shot")
    parser.add_argument("--records", required=True,
                        help="val.jsonl; or test.jsonl with --comparison-set splits/test_comparison_1000_seed17.json")
    parser.add_argument("--comparison-set", default=str(DEFAULT_COMPARISON_SET))
    parser.add_argument("--make-comparison-set", action="store_true", help="draw and write the fixed set, then exit")
    parser.add_argument("--laya-dir", default=None, help="Gate-0 fine-tuned Laya checkpoint directory")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--results-only", action="store_true", help="recompute route_a_results.json and exit")
    parser.add_argument("--limit", type=int, default=None, help="dry run / benchmark: process at most N new images")
    parser.add_argument("--commit-every", type=int, default=8, help="images per durable commit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="consecutive tar stream restarts without progress before giving up")
    args = parser.parse_args()

    records_path = Path(args.records)
    comparison_path = Path(args.comparison_set)
    records = read_jsonl(records_path)
    if not records:
        raise SystemExit(f"{records_path} is empty.")
    split = records_split(records)
    check_split_and_set(split, comparison_path)
    if split == "test":
        print(TEST_NOTICE, flush=True)
    if args.make_comparison_set:
        make_comparison_set(records_path, records, comparison_path, split)
        return
    comparison, selected = load_comparison_set(comparison_path, records_path, records)
    if args.out_dir is None:
        raise SystemExit("--out-dir is required.")
    out_dir = Path(args.out_dir)
    if split == "test" and not args.results_only and (out_dir / "route_a_results.json").exists():
        with open(out_dir / "route_a_results.json", encoding="utf-8") as f:
            if json.load(f).get("complete"):
                raise SystemExit(f"{out_dir} holds a complete Route A test result; the test subsample is run once.")

    if args.results_only:
        with open(out_dir / "route_a.manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        results = compute_results(read_jsonl(out_dir / "route_a_outputs.jsonl"), comparison, manifest)
        print_results(results)
        print(f"Wrote {write_results(out_dir, results)}")
        return

    if args.laya_dir is None:
        raise SystemExit("--laya-dir is required.")
    check_transformers_version()
    print("Hashing Laya checkpoint ...", flush=True)
    manifest = build_manifest(args, comparison, laya_checkpoint_hashes(args.laya_dir))
    outputs = Outputs(out_dir, manifest)
    wanted = {r["image_path"]: r for r in selected if r["id"] not in outputs.done}
    budget = len(wanted) if args.limit is None else min(args.limit, len(wanted))
    print(f"{len(selected)} in comparison set, {len(wanted)} pending, processing {budget} this run", flush=True)

    if budget > 0:
        run(args, wanted, budget, outputs)

    rows = read_jsonl(outputs.out)
    if len(rows) == comparison["n"]:
        outputs.sort()
        rows = read_jsonl(outputs.out)
    results = compute_results(rows, comparison, manifest)
    print_results(results)
    print(f"Wrote {write_results(out_dir, results)}")
    if args.limit is None and not results["complete"]:
        raise SystemExit(f"{comparison['n'] - results['n_done']} comparison images were not processed.")


def run(args, wanted, budget, outputs):
    import http.client
    import tarfile
    import zlib

    import laya
    import requests
    import torch
    import urllib3
    from PIL import Image

    STREAM_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException,
                     tarfile.TarError, zlib.error, EOFError, ConnectionError, TimeoutError)

    describer = Describer(args.device, args.dtype)
    agent = laya.Agent(args.laya_dir, device=args.device)
    gpu = torch.cuda.get_device_name(0)
    processed = 0

    def process(record, data):
        base = {"id": record["id"], "image_index": record["image_index"], "label_name": record["label_name"]}
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:  # corrupt TIFFs exist in RVL-CDIP
            return {**base, "error": f"decode: {type(e).__name__}: {e}", "correct": False}
        d = describer(img)
        start = time.perf_counter()
        res = agent.predict(d["description"], laya_questions(record))
        laya_seconds = time.perf_counter() - start
        answer = res["answers"][QUESTION_ID]
        return {
            **base,
            "input_size": list(img.size),
            "model_size": d["model_size"],
            "visual_tokens": d["visual_tokens"],
            "description": d["description"],
            "description_tokens": d["description_tokens"],
            "laya": {"choice": answer["choice"], "probabilities": answer["probabilities"],
                     "truncated": res.get("usage", {}).get("truncated")},
            "correct": answer["choice"] == record["label_name"],
            "vlm_seconds": d["seconds"],
            "laya_seconds": laya_seconds,
            "gpu": gpu,
        }

    def flush(batch):
        nonlocal processed
        rows = [process(record, data) for record, data in batch]
        outputs.commit(rows)
        processed += len(rows)
        timed = [r for r in rows if r.get("error") is None]
        if timed:
            last = timed[-1]
            print(f"  {processed}/{budget}  last: VLM {last['vlm_seconds']:.2f}s "
                  f"({last['description_tokens']} tokens), Laya {1000 * last['laya_seconds']:.0f} ms, "
                  f"{last['laya']['choice']!r} vs {last['label_name']!r}", flush=True)

    images = queue.Queue()
    stop = threading.Event()
    reader = threading.Thread(target=read_stream, args=(wanted, budget, images, stop, STREAM_ERRORS, args.max_retries),
                              daemon=True)
    reader.start()
    batch = []
    try:
        while processed + len(batch) < budget:
            item = images.get()
            if item is None:  # stream finished
                break
            if isinstance(item, BaseException):
                if batch:
                    flush(batch)
                raise item
            batch.append(item)
            if len(batch) == args.commit_every:
                flush(batch)
                batch = []
        if batch:
            flush(batch)
    finally:
        stop.set()


def read_stream(wanted, budget, images, stop, stream_errors, max_retries):
    """Background reader: drain the tar stream into `images` as fast as the network allows.

    Inference takes seconds per image; reading the stream only between images
    left the connection idle for ~60 s per 8-image commit, and the server
    dropped it ("Connection broken") about every 8 images. Here the stream is
    read continuously, and only wanted images are held (RVL-CDIP TIFFs are
    ~100 KB, so the 2,000-image set is ~200 MB). A real network drop restarts
    the stream from the start (gzip cannot seek) after a backoff, skipping
    images already queued; it costs download time only, not GPU time.
    Puts (record, bytes) items, then None at the end, or the exception if
    retries run out.
    """
    wanted = dict(wanted)
    queued = 0
    attempt, progress_at_error = 0, 0
    try:
        while queued < budget and wanted and not stop.is_set():
            try:
                for image_path, data in rvlcdip.iter_tar_images():
                    if stop.is_set():
                        return
                    record = wanted.pop(image_path, None)
                    if record is None:
                        continue
                    images.put((record, data))
                    queued += 1
                    if queued >= budget:
                        break
                break
            except stream_errors as e:
                if queued > progress_at_error:
                    attempt = 0
                progress_at_error = queued
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = min(300, 10 * 2 ** (attempt - 1))
                print(f"Stream error ({type(e).__name__}: {e}); restarting stream in {delay}s "
                      f"({attempt}/{max_retries} without progress; {queued} images queued so far)", flush=True)
                if stop.wait(delay):
                    return
        images.put(None)
    except BaseException as e:  # hand any failure to the main thread
        images.put(e)

if __name__ == "__main__":
    main()

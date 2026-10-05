"""Distil the secondary label (handwriting noul) with a small VLM.

The handwriting noul is a TRAIN-ONLY auxiliary soft label, not a reported
evaluation metric. Legibility was dropped as degenerate (both DECISIONS.md
2026-10-05).

For every record in the given <split>.jsonl files, the image is fetched from
the RVL-CDIP tar stream and the VLM answers two prompts: handwriting and its
inverse ("is everything printed or typed?"). Answers are read
from next-token probabilities over the allowed answer tokens (no free
generation), so every output parses and carries a probability distribution
usable as a soft training target. The handwriting p(true) is the mean of
p(yes) on the direct prompt and p(no) on the inverse one, which cancels the
model's bias towards answering "yes".

Each image is encoded once: the shared prefix (system turn + image) runs
through the model a single time and its KV cache is reused for the two
short question suffixes. --check-cache also scores every image with full,
uncached prompts and prints the largest probability difference, to confirm
the cached path matches.

Outputs, in --out-dir (default: next to each records file):
  <split>_secondary.jsonl          one line per record (sorted by image_index once complete)
  <split>_secondary.done.log       ids whose output line is durably written (resume state)
  <split>_secondary.manifest.json  model, revision, transformers version, prompt version,
                                   prompts, dtype, pixel limits, ids
A resumed run must use the same configuration as the manifest; otherwise it
refuses, so labels from two configurations (or prompt versions) never mix in
one file.

Test records are labelled under a recorded exception (DECISIONS.md
2026-10-03), so all splits share one pinned environment. Condition: no human
or agent reads or aggregates test secondary labels until final_eval.py runs.
The script prints a notice when labelling test and never prints test answers.

Pinned config (DECISIONS.md 2026-10-05): prompt v3, transformers 4.49.0 (any
other installed version is refused), min/max pixels 1003520 / 1254400
(mid-res, T4-safe) as the defaults.

A network drop in the tar stream ("Connection broken") restarts the stream
from the start after a backoff; committed images are skipped on the re-read.
--max-retries bounds consecutive failures without progress: the count resets
whenever a batch is committed, so a long run survives any number of isolated
drops.

    python scripts/distill_secondary_labels.py --records data/train.jsonl data/val.jsonl
    python scripts/distill_secondary_labels.py --records data/val.jsonl --ids rvl-val-034620,... \\
        --out-dir runs/v3_check --check-cache                       # fixed-set check
"""

import argparse
import copy
import io
import json
import os
import time
from pathlib import Path

import rvlcdip

# Model choice: Qwen2-VL-2B-Instruct.
# - Apache-2.0, so distilled labels carry no licence restriction into an
#   Apache-2.0 release (Qwen2.5-VL-3B is under the non-commercial
#   qwen-research licence; ruled out).
# - Strong on document images for its size (DocVQA ~90 reported), with native
#   dynamic resolution, so scans are seen near full resolution rather than
#   squashed to a fixed square.
# - ~4.4 GB in fp16; fits a 16 GB T4. Native transformers support, no
#   trust_remote_code.
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
# v2: signatures excluded from handwriting (DECISIONS.md 2026-10-03).
# v3: direct handwriting question with examples + inverse question for
#     debiasing. (Legibility dropped 2026-10-05; the handwriting prompts are
#     unchanged, so the version stays v3.)
PROMPT_VERSION = "v3"
# The v3 val experiments ran on this version; pinned (DECISIONS.md 2026-10-05).
TRANSFORMERS_VERSION = "4.49.0"
# Mid-res pixel limits from the v3 val experiments: T4-safe, no OOM.
DEFAULT_MIN_PIXELS = 1003520
DEFAULT_MAX_PIXELS = 1254400

HANDWRITING_ID = "handwriting"
HANDWRITING_INSTRUCTIONS = "This document contains handwriting beyond signatures."
HANDWRITING_PROMPT = (
    "Apart from any signatures, does this scanned document contain anything written by hand, "
    "such as handwritten notes, filled-in form fields, dates, numbers, corrections or margin "
    "annotations? Answer yes or no."
)
# Inverse wording: "yes" here means no handwriting.
HANDWRITING_INVERSE_ID = "handwriting_inverse"
HANDWRITING_INVERSE_PROMPT = (
    "Other than signatures, is everything on this scanned document printed or typed, with "
    "nothing written by hand? Answer yes or no."
)

PROMPTS = {
    HANDWRITING_ID: HANDWRITING_PROMPT,
    HANDWRITING_INVERSE_ID: HANDWRITING_INVERSE_PROMPT,
}

# Answer strings scored at the first generated position. Case variants are
# summed so the model's capitalisation preference does not bias the answer.
YES_NO_ANSWERS = {"false": ["no", "No"], "true": ["yes", "Yes"]}
ANSWERS = {
    HANDWRITING_ID: YES_NO_ANSWERS,
    HANDWRITING_INVERSE_ID: YES_NO_ANSWERS,
}

VISION_END = "<|vision_end|>"
GRID_LOG_IMAGES = 20  # print the resolution the model receives for the first N images


def output_paths(records_path, out_dir=None):
    p = Path(records_path)
    stem = p.name[:-len(".jsonl")] if p.name.endswith(".jsonl") else p.stem
    d = Path(out_dir) if out_dir else p.parent
    return {
        "out": d / f"{stem}_secondary.jsonl",
        "done": d / f"{stem}_secondary.done.log",
        "manifest": d / f"{stem}_secondary.manifest.json",
    }


def config_manifest(args, ids):
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dtype": args.dtype,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "single_encode": True,
        "ids": sorted(ids) if ids else None,
        "dataset": rvlcdip.DATASET_REPO,
        "dataset_revision": rvlcdip.DATASET_REVISION,
        "questions": {
            HANDWRITING_ID: {"type": "noul", "instructions": HANDWRITING_INSTRUCTIONS,
                             "prompt": HANDWRITING_PROMPT, "inverse_prompt": HANDWRITING_INVERSE_PROMPT,
                             "combine": "mean(p_yes(prompt), p_no(inverse_prompt))",
                             "use": "train-only auxiliary soft label; not an evaluation metric"},
        },
    }


def check_transformers_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("transformers")
    except PackageNotFoundError:
        installed = None
    if installed != TRANSFORMERS_VERSION:
        raise SystemExit(f"transformers {installed} is installed; the pinned config needs "
                         f"{TRANSFORMERS_VERSION} (pip install transformers=={TRANSFORMERS_VERSION}).")


def fsync_append(path, lines):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class SplitState:
    """Per records file: what is done, and where outputs go."""

    def __init__(self, records_path, records, manifest, out_dir=None):
        self.records = records
        self.paths = output_paths(records_path, out_dir)
        self.paths["out"].parent.mkdir(parents=True, exist_ok=True)

        if self.paths["manifest"].exists():
            with open(self.paths["manifest"], encoding="utf-8") as f:
                existing = json.load(f)
            if existing != manifest:
                raise SystemExit(f"{self.paths['manifest']} was written with a different configuration; "
                                 "refusing to mix labels. Use the original settings or a new --out-dir.")
        else:
            with open(self.paths["manifest"], "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                f.write("\n")

        self.done = set()
        if self.paths["done"].exists():
            with open(self.paths["done"], encoding="utf-8") as f:
                self.done = {line.strip() for line in f if line.strip()}
        # Drop output lines not confirmed in done.log (crash between the two writes) and duplicates.
        rows, seen = [], set()
        for row in read_jsonl(self.paths["out"]):
            if row["id"] in self.done and row["id"] not in seen:
                rows.append(row)
                seen.add(row["id"])
        if len(seen) != len(self.done):
            # done.log lists an id whose line is missing: forget it so it is redone.
            self.done = seen
            with open(self.paths["done"], "w", encoding="utf-8", newline="\n") as f:
                f.writelines(i + "\n" for i in sorted(seen))
        write_jsonl_atomic(self.paths["out"], rows)

    def pending(self):
        return [r for r in self.records if r["id"] not in self.done]

    def commit(self, outputs):
        fsync_append(self.paths["out"], [json.dumps(o, ensure_ascii=False) for o in outputs])
        fsync_append(self.paths["done"], [o["id"] for o in outputs])
        self.done.update(o["id"] for o in outputs)

    def finalize_if_complete(self):
        if len(self.done) < len(self.records):
            return False
        rows = sorted(read_jsonl(self.paths["out"]), key=lambda r: r["image_index"])
        write_jsonl_atomic(self.paths["out"], rows)
        return True


class Distiller:
    def __init__(self, device, dtype, min_pixels, max_pixels):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        pixel_kwargs = {"max_pixels": max_pixels}
        if min_pixels is not None:
            pixel_kwargs["min_pixels"] = min_pixels
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=rvlcdip.hf_token(), **pixel_kwargs)
        self._apply_pixel_limits(min_pixels, max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch_dtype, token=rvlcdip.hf_token(),
        ).to(device).eval()
        self.rope_index = self._find_rope_index()

        self.prefix_text, suffix_texts = self._split_prompts()
        self.full_texts = {q: self.prefix_text + s for q, s in suffix_texts.items()}
        self.suffix_ids = {
            q: torch.tensor([self.tokenizer(s, add_special_tokens=False)["input_ids"]], device=device)
            for q, s in suffix_texts.items()
        }
        self.answer_ids = {q: self._token_ids(a) for q, a in ANSWERS.items()}
        ip = self.processor.image_processor
        self.patch_size = getattr(ip, "patch_size", 14)
        self.merge_size = getattr(ip, "merge_size", 2)

    def _apply_pixel_limits(self, min_pixels, max_pixels):
        # The kwarg route has moved between transformers versions; set every
        # known spelling so the limits apply. image_grid_thw in the log shows
        # the resolution actually used.
        ip = self.processor.image_processor
        if min_pixels is not None and hasattr(ip, "min_pixels"):
            ip.min_pixels = min_pixels
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        size = getattr(ip, "size", None)
        if isinstance(size, dict) and ("longest_edge" in size or "shortest_edge" in size):
            size = dict(size)
            size["longest_edge"] = max_pixels
            if min_pixels is not None:
                size["shortest_edge"] = min_pixels
            ip.size = size

    def _find_rope_index(self):
        for owner in (self.model, getattr(self.model, "model", None)):
            fn = getattr(owner, "get_rope_index", None)
            if fn is not None:
                return fn
        return None

    def _split_prompts(self):
        """Split each chat prompt into the shared prefix (through the image) and its suffix."""
        prefix, suffixes = None, {}
        for qid, prompt in PROMPTS.items():
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            full = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            cut = full.index(VISION_END) + len(VISION_END)
            if prefix is None:
                prefix = full[:cut]
            elif full[:cut] != prefix:
                raise RuntimeError("Chat prompts do not share a common prefix; single encode is not possible.")
            suffixes[qid] = full[cut:]
            # Tokenising the halves separately must equal tokenising the whole.
            whole = self.tokenizer(full, add_special_tokens=False)["input_ids"]
            parts = (self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                     + self.tokenizer(full[cut:], add_special_tokens=False)["input_ids"])
            if whole != parts:
                raise RuntimeError(f"Prefix/suffix tokenisation differs from the full prompt for {qid}.")
        return prefix, suffixes

    def _token_ids(self, answers):
        out = {}
        for key, strings in answers.items():
            ids = []
            for s in strings:
                tok = self.tokenizer.encode(s, add_special_tokens=False)
                if len(tok) != 1:
                    raise RuntimeError(f"Answer {s!r} is not a single token for {MODEL_ID}: {tok}")
                ids.append(tok[0])
            out[key] = ids
        return out

    def _forward(self, **kwargs):
        try:
            return self.model(**kwargs, logits_to_keep=1)
        except TypeError:  # older transformers without logits_to_keep
            return self.model(**kwargs)

    def _answer_probs(self, qid, logits):
        probs = self.torch.softmax(logits.float(), dim=-1).cpu()
        mass = {k: float(probs[ids].sum()) for k, ids in self.answer_ids[qid].items()}
        total = sum(mass.values())
        if not total > 0:  # also catches NaN from fp16 overflow
            raise FloatingPointError(f"No probability mass on answer tokens for {qid}")
        return {k: v / total for k, v in mass.items()}, total

    def __call__(self, image):
        """Score one image: {"answers": {qid: (probabilities, answer_mass)}, "image": {...}}."""
        torch = self.torch
        with torch.no_grad():
            prefix = self.processor(text=[self.prefix_text], images=[image], return_tensors="pt").to(self.device)
            prefix_len = prefix["input_ids"].shape[1]
            cache = self._forward(**prefix, use_cache=True).past_key_values

            answers = {}
            for qid, suffix in self.suffix_ids.items():
                n = suffix.shape[1]
                attention = torch.ones((1, prefix_len + n), dtype=prefix["attention_mask"].dtype, device=self.device)
                position_ids = None
                if self.rope_index is not None:
                    full_ids = torch.cat([prefix["input_ids"], suffix], dim=1)
                    position_ids = self.rope_index(full_ids, prefix["image_grid_thw"], None, attention)[0]
                    position_ids = position_ids[..., prefix_len:]
                out = self._forward(
                    input_ids=suffix,
                    attention_mask=attention,
                    position_ids=position_ids,
                    past_key_values=copy.deepcopy(cache),
                    cache_position=torch.arange(prefix_len, prefix_len + n, device=self.device),
                    use_cache=True,
                )
                answers[qid] = self._answer_probs(qid, out.logits[0, -1])

        t, h, w = (int(x) for x in prefix["image_grid_thw"][0])
        info = {
            "input_size": list(image.size),
            "grid_thw": [t, h, w],
            "model_size": [w * self.patch_size, h * self.patch_size],
            "visual_tokens": t * h * w // (self.merge_size ** 2),
        }
        return {"answers": answers, "image": info}

    def reference(self, image):
        """Full, uncached prompts (the pre-optimisation path), for --check-cache."""
        answers = {}
        with self.torch.no_grad():
            for qid, text in self.full_texts.items():
                inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
                answers[qid] = self._answer_probs(qid, self._forward(**inputs).logits[0, -1])
        return answers


def build_output(record, result):
    answers = result["answers"]
    direct, direct_mass = answers[HANDWRITING_ID]
    inverse, inverse_mass = answers[HANDWRITING_INVERSE_ID]
    p_true = (direct["true"] + inverse["false"]) / 2
    return {
        "id": record["id"],
        "split": record["split"],
        "image_index": record["image_index"],
        "prompt_version": PROMPT_VERSION,
        "image": result["image"],
        "questions": {
            HANDWRITING_ID: {
                "type": "noul",
                "instructions": HANDWRITING_INSTRUCTIONS,
                "answer": p_true >= 0.5,
                "probabilities": {"false": 1 - p_true, "true": p_true},
                "components": {"direct_p_yes": direct["true"], "inverse_p_yes": inverse["true"]},
                "answer_mass": {"direct": round(direct_mass, 6), "inverse": round(inverse_mass, 6)},
            },
        },
    }


def error_output(record, message):
    return {"id": record["id"], "split": record["split"], "image_index": record["image_index"],
            "prompt_version": PROMPT_VERSION, "questions": None, "error": message}


def main():
    parser = argparse.ArgumentParser(description="Distil the handwriting auxiliary label with a VLM")
    parser.add_argument("--records", nargs="+", required=True, help="<split>.jsonl files")
    parser.add_argument("--ids", default=None, help="comma-separated record ids: label only these")
    parser.add_argument("--out-dir", default=None, help="output folder (default: next to each records file)")
    parser.add_argument("--limit", type=int, default=None, help="dry run: label at most N new records")
    parser.add_argument("--commit-every", type=int, default=8, help="images per durable commit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="fp16 on T4 (no native bf16)")
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS,
                        help="Qwen2-VL resize floor; smaller images are enlarged")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                        help="Qwen2-VL resize cap; ~1 visual token per 28x28 px")
    parser.add_argument("--check-cache", action="store_true",
                        help="also score each image with full uncached prompts and print the max difference")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="consecutive tar stream restarts without progress before giving up")
    args = parser.parse_args()
    check_transformers_version()
    if args.min_pixels is not None and args.min_pixels > args.max_pixels:
        raise SystemExit("--min-pixels must not exceed --max-pixels.")

    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    loaded = {}
    for path in args.records:
        records = read_jsonl(Path(path))
        if ids is not None:
            id_set = set(ids)
            records = [r for r in records if r["id"] in id_set]
            if not records:
                continue  # no requested ids in this file: write nothing for it
        loaded[path] = records
    if ids is not None:
        found = {r["id"] for rs in loaded.values() for r in rs}
        unknown = [i for i in ids if i not in found]
        if unknown:
            raise SystemExit(f"{len(unknown)} ids not in --records: {unknown[:10]}")
    has_test = any(r["split"] == "test" for rs in loaded.values() for r in rs)
    if has_test:
        if args.check_cache:
            raise SystemExit("--check-cache prints per-image comparisons; not allowed on test records.")
        print("\n" + "!" * 78)
        print("!! LABELLING TEST RECORDS")
        print("!! Recorded exception, DECISIONS.md 2026-10-03: no human or agent may read or")
        print("!! aggregate test secondary labels until final_eval.py runs.")
        print("!" * 78 + "\n", flush=True)

    # The work queue is keyed by image path, so a repeated image would send its label to the wrong file.
    seen_paths = set()
    for records in loaded.values():
        for record in records:
            if record["image_path"] in seen_paths:
                raise SystemExit(f"Image {record['image_path']} ({record['id']}) appears more than once "
                                 "across the records files; they must not overlap.")
            seen_paths.add(record["image_path"])

    manifest = config_manifest(args, ids)
    states = [SplitState(p, records, manifest, args.out_dir) for p, records in loaded.items()]
    wanted = {}
    for state in states:
        for record in state.pending():
            wanted[record["image_path"]] = (state, record)
    budget = len(wanted) if args.limit is None else min(args.limit, len(wanted))
    print(f"{sum(len(s.records) for s in states)} records, {len(wanted)} pending, labelling {budget} this run")
    if budget == 0:
        for state in states:
            state.finalize_if_complete()
        return

    import http.client
    import tarfile
    import zlib

    import requests
    import urllib3
    from PIL import Image

    # Network/archive failures that a stream restart can recover from.
    # "Connection broken: IncompleteRead" surfaces as urllib3 ProtocolError (an HTTPError);
    # http.client errors are listed in case one escapes unwrapped.
    STREAM_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException,
                     tarfile.TarError, zlib.error, EOFError, ConnectionError, TimeoutError)

    distiller = Distiller(args.device, args.dtype, args.min_pixels, args.max_pixels)
    labelled = 0
    max_cache_diff = 0.0
    t0 = time.time()

    def label(record, img, err):
        nonlocal max_cache_diff
        if err is not None:
            return error_output(record, err)
        try:
            result = distiller(img)
        except FloatingPointError as e:
            return error_output(record, f"inference: {e}")
        if labelled + 1 <= GRID_LOG_IMAGES:
            info = result["image"]
            print(f"  {record['id']}: input {info['input_size'][0]}x{info['input_size'][1]} -> "
                  f"grid_thw {info['grid_thw']} = model sees {info['model_size'][0]}x{info['model_size'][1]} px, "
                  f"{info['visual_tokens']} visual tokens", flush=True)
        if args.check_cache:
            ref = distiller.reference(img)
            diff = max(abs(result["answers"][q][0][k] - ref[q][0][k])
                       for q in ref for k in ref[q][0])
            max_cache_diff = max(max_cache_diff, diff)
            print(f"  {record['id']}: cached vs full-prompt max |dp| = {diff:.4f}", flush=True)
        return build_output(record, result)

    def flush(batch):
        """batch: [(state, record, image or None, decode error or None)]"""
        nonlocal labelled
        outputs = []
        for state, record, img, err in batch:
            outputs.append((state, label(record, img, err)))
            labelled += 1
        for state in states:
            mine = [o for s, o in outputs if s is state]
            if mine:
                state.commit(mine)
        rate = labelled / max(1e-9, time.time() - t0)
        print(f"  {labelled}/{budget} labelled ({rate:.2f} img/s)", flush=True)

    attempt = 0
    progress_at_error = labelled
    while labelled < budget and wanted:
        batch = []
        committed = labelled
        try:
            for image_path, data in rvlcdip.iter_tar_images():
                item = wanted.pop(image_path, None)
                if item is None:
                    continue
                state, record = item
                img, err = None, None
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception as e:  # corrupt TIFFs exist in RVL-CDIP
                    err = f"decode: {type(e).__name__}: {e}"
                batch.append((state, record, img, err))
                if len(batch) == args.commit_every or committed + len(batch) >= budget:
                    flush(batch)
                    batch = []
                    committed = labelled
                if labelled >= budget:
                    break
            if batch:
                flush(batch)
            break
        except STREAM_ERRORS as e:
            # Uncommitted batch items go back to the queue; committed ones are in done.log.
            labelled = committed
            for state, record, _, _ in batch:
                wanted[record["image_path"]] = (state, record)
            # Only consecutive failures without a commit in between count towards the limit.
            if labelled > progress_at_error:
                attempt = 0
            progress_at_error = labelled
            attempt += 1
            if attempt > args.max_retries:
                raise
            delay = min(300, 10 * 2 ** (attempt - 1))
            print(f"Stream error ({type(e).__name__}: {e}); restarting stream in {delay}s "
                  f"({attempt}/{args.max_retries} without progress)", flush=True)
            time.sleep(delay)

    if args.check_cache:
        print(f"check-cache: largest cached vs full-prompt probability difference = {max_cache_diff:.4f}")
    for state in states:
        complete = state.finalize_if_complete()
        print(f"{state.paths['out']}: {len(state.done)}/{len(state.records)} done"
              + (" (complete, sorted)" if complete else ""))
    if args.limit is None and wanted:
        raise SystemExit(f"{len(wanted)} records were not found in the tar stream.")


if __name__ == "__main__":
    main()

"""Distil the two secondary labels (handwriting noul, legibility score) with a small VLM.

For every record in the given <split>.jsonl files, the image is fetched from
the RVL-CDIP tar stream and the VLM answers two questions. Answers are read
from next-token probabilities over the allowed answer tokens (one forward
pass per question, no free generation), so every output parses and carries a
probability distribution usable as a soft training target.

Outputs, next to each records file:
  <split>_secondary.jsonl          one line per record (sorted by image_index once complete)
  <split>_secondary.done.log       ids whose output line is durably written (resume state)
  <split>_secondary.manifest.json  model, revision, prompts, dtype, max_pixels
A resumed run must use the same configuration as the manifest; otherwise it
refuses, so labels from two configurations never mix in one file.

Test records are refused: the test split is frozen (DECISIONS.md 2026-10-03)
and only final_eval.py may read it.

    python scripts/distill_secondary_labels.py --records data/train.jsonl data/val.jsonl
    python scripts/distill_secondary_labels.py --records data/train.jsonl --limit 16   # dry run
"""

import argparse
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
# - ~4.4 GB in fp16; fits a 16 GB T4 with room for batching. Native
#   transformers support, no trust_remote_code.
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
PROMPT_VERSION = "v1"

HANDWRITING_ID = "handwriting"
HANDWRITING_INSTRUCTIONS = "This document contains handwriting."
HANDWRITING_PROMPT = (
    "Statement: This document contains handwriting.\n"
    "Is the statement true for this document? Answer yes or no."
)

LEGIBILITY_ID = "legibility"
LEGIBILITY_INSTRUCTIONS = "How legible is this document?"
LEGIBILITY_LEVELS = [
    "unreadable: the text cannot be made out",
    "poor: much of the text is hard to read (faint, blurred, noisy or degraded)",
    "readable: the text can be read, with some noise or degradation",
    "clean: crisp, clear text that is easy to read",
]
LEGIBILITY_PROMPT = (
    "How legible is this document?\n"
    + "".join(f"{i + 1}. {level}\n" for i, level in enumerate(LEGIBILITY_LEVELS))
    + f"Answer with a single digit from 1 to {len(LEGIBILITY_LEVELS)}."
)

# Answer strings scored at the first generated position. Case variants are
# summed so the model's capitalisation preference does not bias the answer.
HANDWRITING_ANSWERS = {"false": ["no", "No"], "true": ["yes", "Yes"]}
LEGIBILITY_ANSWERS = {str(i): [str(i + 1)] for i in range(len(LEGIBILITY_LEVELS))}


def output_paths(records_path):
    p = Path(records_path)
    stem = p.name[:-len(".jsonl")] if p.name.endswith(".jsonl") else p.stem
    return {
        "out": p.with_name(f"{stem}_secondary.jsonl"),
        "done": p.with_name(f"{stem}_secondary.done.log"),
        "manifest": p.with_name(f"{stem}_secondary.manifest.json"),
    }


def config_manifest(args):
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "dtype": args.dtype,
        "max_pixels": args.max_pixels,
        "dataset": rvlcdip.DATASET_REPO,
        "dataset_revision": rvlcdip.DATASET_REVISION,
        "questions": {
            HANDWRITING_ID: {"type": "noul", "instructions": HANDWRITING_INSTRUCTIONS,
                             "prompt": HANDWRITING_PROMPT},
            LEGIBILITY_ID: {"type": "score", "instructions": LEGIBILITY_INSTRUCTIONS,
                            "criteria": LEGIBILITY_LEVELS, "prompt": LEGIBILITY_PROMPT},
        },
    }


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

    def __init__(self, records_path, records, manifest):
        self.records = records
        self.paths = output_paths(records_path)

        if self.paths["manifest"].exists():
            with open(self.paths["manifest"], encoding="utf-8") as f:
                existing = json.load(f)
            if existing != manifest:
                raise SystemExit(f"{self.paths['manifest']} was written with a different configuration; "
                                 "refusing to mix labels. Use the original settings or a new output location.")
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
    def __init__(self, device, dtype, max_pixels):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, max_pixels=max_pixels, token=rvlcdip.hf_token())
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch_dtype, token=rvlcdip.hf_token(),
        ).to(device).eval()
        self.prompts = {
            HANDWRITING_ID: self._chat(HANDWRITING_PROMPT),
            LEGIBILITY_ID: self._chat(LEGIBILITY_PROMPT),
        }
        self.answer_ids = {
            HANDWRITING_ID: self._token_ids(HANDWRITING_ANSWERS),
            LEGIBILITY_ID: self._token_ids(LEGIBILITY_ANSWERS),
        }

    def _chat(self, prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _token_ids(self, answers):
        out = {}
        for key, strings in answers.items():
            ids = []
            for s in strings:
                tok = self.processor.tokenizer.encode(s, add_special_tokens=False)
                if len(tok) != 1:
                    raise RuntimeError(f"Answer {s!r} is not a single token for {MODEL_ID}: {tok}")
                ids.append(tok[0])
            out[key] = ids
        return out

    def _next_token_probs(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            try:
                logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
            except TypeError:  # older transformers without logits_to_keep
                logits = self.model(**inputs).logits[:, -1, :]
        return self.torch.softmax(logits.float(), dim=-1).cpu()

    def __call__(self, images):
        """Return one {question_id: (probabilities, answer_mass)} dict per image."""
        qids = list(self.prompts)
        texts = [self.prompts[q] for q in qids for _ in images]
        probs = self._next_token_probs(texts, [img for _ in qids for img in images])
        results = [{} for _ in images]
        for qi, qid in enumerate(qids):
            for ii in range(len(images)):
                row = probs[qi * len(images) + ii]
                mass = {k: float(row[ids].sum()) for k, ids in self.answer_ids[qid].items()}
                total = sum(mass.values())
                if not total > 0:  # also catches NaN from fp16 overflow
                    raise FloatingPointError(f"No probability mass on answer tokens for {qid}")
                results[ii][qid] = ({k: v / total for k, v in mass.items()}, total)
        return results


def build_output(record, result):
    hw_probs, hw_mass = result[HANDWRITING_ID]
    lg_probs, lg_mass = result[LEGIBILITY_ID]
    return {
        "id": record["id"],
        "split": record["split"],
        "image_index": record["image_index"],
        "questions": {
            HANDWRITING_ID: {
                "type": "noul",
                "instructions": HANDWRITING_INSTRUCTIONS,
                "answer": hw_probs["true"] >= 0.5,
                "probabilities": hw_probs,
                "answer_mass": round(hw_mass, 6),
            },
            LEGIBILITY_ID: {
                "type": "score",
                "instructions": LEGIBILITY_INSTRUCTIONS,
                "criteria": LEGIBILITY_LEVELS,
                "answer": int(max(lg_probs, key=lg_probs.get)),
                "probabilities": lg_probs,
                "answer_mass": round(lg_mass, 6),
            },
        },
    }


def error_output(record, message):
    return {"id": record["id"], "split": record["split"], "image_index": record["image_index"],
            "questions": None, "error": message}


def main():
    parser = argparse.ArgumentParser(description="Distil handwriting/legibility labels with a VLM")
    parser.add_argument("--records", nargs="+", required=True, help="<split>.jsonl files (train/val only)")
    parser.add_argument("--limit", type=int, default=None, help="dry run: label at most N new records")
    parser.add_argument("--batch-size", type=int, default=4, help="images per forward (x2 prompts)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="fp16 on T4 (no native bf16)")
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28,
                        help="Qwen2-VL resize cap; ~1 visual token per 28x28 px")
    parser.add_argument("--max-retries", type=int, default=5, help="tar stream restarts on network errors")
    args = parser.parse_args()

    # Check every file before writing anything.
    loaded = {}
    for path in args.records:
        records = read_jsonl(Path(path))
        if any(r["split"] == "test" for r in records):
            raise SystemExit(f"{path}: test records refused. The test split is frozen "
                             "(DECISIONS.md 2026-10-03); only final_eval.py may read it.")
        loaded[path] = records

    manifest = config_manifest(args)
    states = [SplitState(p, records, manifest) for p, records in loaded.items()]
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

    import tarfile
    import zlib

    import requests
    import urllib3
    from PIL import Image

    # Network/archive failures that a stream restart can recover from.
    STREAM_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError,
                     tarfile.TarError, zlib.error, EOFError, ConnectionError, TimeoutError)

    distiller = Distiller(args.device, args.dtype, args.max_pixels)
    labelled = 0
    t0 = time.time()

    def flush(batch):
        """batch: [(state, record, image or None, decode error or None)]"""
        nonlocal labelled
        good = [(s, r, img) for s, r, img, _ in batch if img is not None]
        outputs = {}
        if good:
            try:
                results = distiller([img for _, _, img in good])
                for (s, r, _), res in zip(good, results):
                    outputs[r["id"]] = (s, build_output(r, res))
            except FloatingPointError:
                # Retry one by one so a single bad image does not poison the batch.
                for s, r, img in good:
                    try:
                        outputs[r["id"]] = (s, build_output(r, distiller([img])[0]))
                    except FloatingPointError as e:
                        outputs[r["id"]] = (s, error_output(r, f"inference: {e}"))
        for s, r, _, err in batch:
            if err is not None:
                outputs[r["id"]] = (s, error_output(r, err))
        for state in states:
            mine = [o for s, o in outputs.values() if s is state]
            if mine:
                state.commit(mine)
        labelled += len(batch)
        rate = labelled / max(1e-9, time.time() - t0)
        print(f"  {labelled}/{budget} labelled ({rate:.2f} img/s)", flush=True)

    attempt = 0
    while labelled < budget and wanted:
        batch = []
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
                if len(batch) == args.batch_size or labelled + len(batch) >= budget:
                    flush(batch)
                    batch = []
                if labelled >= budget:
                    break
            if batch:
                flush(batch)
            break
        except STREAM_ERRORS as e:
            # Uncommitted batch items go back to the queue; committed ones are in done.log.
            for state, record, _, _ in batch:
                wanted[record["image_path"]] = (state, record)
            attempt += 1
            if attempt > args.max_retries:
                raise
            print(f"Stream error ({type(e).__name__}: {e}); restarting stream ({attempt}/{args.max_retries})")

    for state in states:
        complete = state.finalize_if_complete()
        print(f"{state.paths['out']}: {len(state.done)}/{len(state.records)} done"
              + (" (complete, sorted)" if complete else ""))
    if args.limit is None and wanted:
        raise SystemExit(f"{len(wanted)} records were not found in the tar stream.")


if __name__ == "__main__":
    main()

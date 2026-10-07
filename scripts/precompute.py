"""Phase 2 precompute: full-resolution OCR + frozen SigLIP embeddings, sharded.

For each record in a split JSONL from build_dataset.py, the image is streamed
from the RVL-CDIP tar (same loader and stream-retry as route_a.py) and goes
through two independent passes:

  ocr    OCR on the ORIGINAL full-resolution image, before any resize or
         contrast change. One engine per output directory (default
         tesseract; see OCR_ENGINES), recorded with its version.
  embed  Preprocessing PREPROCESS_VERSION (RGB -> contrast normalisation ->
         letterbox to 384x384), then frozen SigLIP base-384, last hidden state
         (24x24 patch grid, post-layernorm), average-pooled to a fixed grid
         (default 8 rows x 4 cols = 32 tokens, row-major), stored fp16.

Output, in --out-dir/<split>/:
  run_config.json          identity of this output: split, records sha256, shard
                           size, record count, and each pass's config. A rerun
                           must match it, so two configurations never mix.
  done.log                 "<pass> shard_NNNN" per completed pass (resume state)
  shard_NNNN/
    embeddings.npy         float16 [n, tokens, 768]; rows follow manifest ids
    ocr.jsonl              one line per image, in manifest id order
    manifest.json          ids, preprocessing version, OCR engine + version,
                           SigLIP id + revision, pooling spec, git hash, dates,
                           decode errors

Shards are fixed chunks of --shard-size records in split-file order, so a
shard's ids never depend on tar order. A shard is processed once all its
images have arrived. Images are spooled to --spool-dir (local disk, not
Drive) by a background reader that keeps the HTTP stream busy, so RAM stays
bounded and the connection never idles. Kill and rerun at any time: done
shards are skipped, spooled images are reused, and only missing images are
streamed again (one tar pass costs download time, not GPU time).

--limit N is a dry run: only the first N records, sharded the same way. The
record count is part of run_config.json, so a dry run's output directory
cannot be resumed as a full run; use a scratch --out-dir for dry runs.

TEST SPLIT (recorded exception, DECISIONS.md 2026-10-07): this script may
process test images. It keeps only id / image_index / image_path from each
record and drops every label field at parse time; its transforms are
deterministic and it computes no aggregates over test. It prints a notice
when processing test.

    python scripts/precompute.py --records data/rvl_v1/val.jsonl --out-dir /tmp/p2_dry --limit 64
    python scripts/precompute.py --records data/rvl_v1/val.jsonl --out-dir <drive>/phase2/shards
    python scripts/precompute.py --records ... --out-dir ... --skip-ocr     # embeddings only (GPU)
    python scripts/precompute.py --records ... --out-dir ... --skip-embed   # OCR only (CPU)
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rvlcdip

REPO = Path(__file__).resolve().parent.parent

# --- pinned configuration ---------------------------------------------------------

# p2v1: RGB -> autocontrast (cutoff 1%, tone-preserving) -> bicubic resize of the
# long side to 384 -> centre pad with white to 384x384. Never change in place:
# a new version gets a new id and a new output directory (CLAUDE.md).
PREPROCESS_VERSION = "p2v1"
IMAGE_SIZE = 384
CONTRAST_CUTOFF_PERCENT = 1
PAD_RGB = (255, 255, 255)
PREPROCESS_SPEC = {
    "version": PREPROCESS_VERSION,
    "steps": [
        "PIL decode; convert('RGB')",
        f"ImageOps.autocontrast(cutoff={CONTRAST_CUTOFF_PERCENT}, preserve_tone=True) on the full page",
        f"resize so the long side is {IMAGE_SIZE} px, aspect preserved, BICUBIC",
        f"centre pad to {IMAGE_SIZE}x{IMAGE_SIZE} with RGB {PAD_RGB}",
        "SigLIP normalisation: x/255, then (x - 0.5) / 0.5 (checked against the processor config)",
    ],
}

SIGLIP_ID = "google/siglip-base-patch16-384"
SIGLIP_REVISION = "41aec1c83b32e0a6fca20ad88ba058aa5b5ea394"
DEFAULT_POOL_GRID = "8x4"  # rows x cols over the 24x24 patch grid -> 32 tokens

DEFAULT_SHARD_SIZE = 2000
LOCK_STALE_SECONDS = 600

# OCR engine choice (default tesseract; DECISIONS.md once confirmed):
# - RVL-CDIP is low-resolution (~90 dpi) printed business documents; Tesseract's
#   LSTM engine is built for printed document pages, and it is the OCR used for
#   RVL-CDIP in the document-AI literature (e.g. LayoutLM), so our text channel
#   is comparable to published work.
# - CPU-only and deterministic for a fixed version: the OCR pass can run on a
#   CPU runtime or alongside the GPU embedding pass without competing for VRAM,
#   and reruns reproduce the same text.
# - EasyOCR (CRAFT detector + CRNN) targets scene text; on dense pages it
#   recognises box by box, so GPU speed is not guaranteed to win, and GPU
#   kernels add run-to-run nondeterminism. Kept as an option for the benchmark.
OCR_ENGINES = ("tesseract", "easyocr")
TESSERACT_LANG = "eng"
TESSERACT_CONFIG = "--oem 1 --psm 3"  # LSTM engine, automatic page segmentation


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_git_hash():
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fsync_append(path, lines):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# --- records ----------------------------------------------------------------------

RECORD_KEYS = ("id", "image_index", "image_path")


def load_records(path):
    """Read a split JSONL keeping only id / image_index / image_path.

    Labels and questions are dropped as each line is parsed, so this script
    never holds a label (the test-split exception depends on it).
    """
    records, splits = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                splits.add(row["split"])
                records.append({k: row[k] for k in RECORD_KEYS})
                del row
    if len(splits) != 1:
        raise SystemExit(f"{path}: expected one split, found {sorted(splits)}")
    return splits.pop(), records


def make_shards(records, shard_size):
    return {f"shard_{i // shard_size:04d}": records[i:i + shard_size] for i in range(0, len(records), shard_size)}


def test_notice(split):
    if split == "test":
        print("=" * 78 + "\n"
              "NOTICE: processing TEST split images under the recorded exception\n"
              "(DECISIONS.md 2026-10-07). Deterministic image transforms only: labels are\n"
              "dropped at parse time; no labels are loaded and no aggregates are computed.\n"
              + "=" * 78, flush=True)


# --- images -----------------------------------------------------------------------

def decode(data):
    """Decode the original scan exactly as stored (no resize, no mode change)."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def letterbox(img):
    """Preprocessing p2v1 on an already-decoded original image -> 384x384 RGB."""
    from PIL import Image, ImageOps

    rgb = img.convert("RGB")
    rgb = ImageOps.autocontrast(rgb, cutoff=CONTRAST_CUTOFF_PERCENT, preserve_tone=True)
    w, h = rgb.size
    scale = IMAGE_SIZE / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    rgb = rgb.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), PAD_RGB)
    canvas.paste(rgb, ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2))
    return canvas


def parse_grid(spec):
    rows, cols = (int(x) for x in spec.lower().split("x"))
    return rows, cols


# --- SigLIP -----------------------------------------------------------------------

class Embedder:
    """Frozen SigLIP vision tower -> patch grid -> fixed average pooling."""

    def __init__(self, device, dtype, pool_grid):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.torch = torch
        self.device = device
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        proc = AutoImageProcessor.from_pretrained(SIGLIP_ID, revision=SIGLIP_REVISION, token=rvlcdip.hf_token())
        # Our letterbox replaces the processor's square resize; only its scaling
        # and normalisation are reproduced, so check they are what we assume.
        if (list(proc.image_mean) != [0.5] * 3 or list(proc.image_std) != [0.5] * 3
                or abs(proc.rescale_factor - 1 / 255) > 1e-9
                or (proc.size["height"], proc.size["width"]) != (IMAGE_SIZE, IMAGE_SIZE)):
            raise SystemExit(f"{SIGLIP_ID}@{SIGLIP_REVISION}: processor config differs from the assumed "
                             "384px, mean 0.5, std 0.5, 1/255 rescale.")
        self.model = SiglipVisionModel.from_pretrained(
            SIGLIP_ID, revision=SIGLIP_REVISION, torch_dtype=self.dtype, token=rvlcdip.hf_token(),
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.grid = cfg.image_size // cfg.patch_size
        self.hidden = cfg.hidden_size
        self.pool_rows, self.pool_cols = pool_grid
        if self.grid % self.pool_rows or self.grid % self.pool_cols:
            raise SystemExit(f"pool grid {pool_grid} must divide the {self.grid}x{self.grid} patch grid evenly.")

    @property
    def tokens(self):
        return self.pool_rows * self.pool_cols

    def spec(self):
        return {
            "model_id": SIGLIP_ID,
            "revision": SIGLIP_REVISION,
            "tower": "SiglipVisionModel.last_hidden_state (post-layernorm patch tokens; no attention-pool head)",
            "compute_dtype": str(self.dtype).replace("torch.", ""),
            "patch_grid": [self.grid, self.grid],
            "pooling": {
                "type": "average, fixed, non-overlapping",
                "grid_rows_cols": [self.pool_rows, self.pool_cols],
                "patches_per_token_rows_cols": [self.grid // self.pool_rows, self.grid // self.pool_cols],
                "token_order": "row-major (top-left first)",
                "tokens": self.tokens,
            },
            "stored_dtype": "float16",
            "hidden_size": self.hidden,
        }

    def __call__(self, images):
        """images: list of 384x384 RGB PIL images -> float16 numpy [n, tokens, hidden]."""
        import numpy as np

        torch = self.torch
        arr = np.stack([np.asarray(im, dtype=np.uint8) for im in images])  # n,H,W,3
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float()
        x = ((x / 255.0) - 0.5) / 0.5
        with torch.no_grad():
            h = self.model(pixel_values=x.to(self.dtype)).last_hidden_state  # n, grid*grid, hidden
            n = h.shape[0]
            h = h.float().reshape(n, self.grid, self.grid, self.hidden).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.adaptive_avg_pool2d(h, (self.pool_rows, self.pool_cols))
            pooled = pooled.flatten(2).transpose(1, 2)  # n, rows*cols, hidden (row-major)
        return pooled.to(torch.float16).cpu().numpy()


# --- OCR --------------------------------------------------------------------------

class TesseractOCR:
    """Tesseract LSTM via pytesseract; one tesseract process per image, run in parallel."""

    name = "tesseract"

    def __init__(self, workers=None):
        import pytesseract

        # Each tesseract process single-threaded; parallelism comes from the pool.
        os.environ["OMP_THREAD_LIMIT"] = "1"
        self.pytesseract = pytesseract
        self.workers = workers or os.cpu_count() or 1
        try:
            out = subprocess.run([pytesseract.pytesseract.tesseract_cmd, "--version"],
                                 capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            raise SystemExit(f"tesseract binary not usable ({e}); on Colab: apt-get install -y tesseract-ocr")
        self.version_text = (out.stdout + out.stderr).strip()
        self.version = str(pytesseract.get_tesseract_version())
        self.traineddata_sha256 = self._traineddata_hash()

    def _traineddata_hash(self):
        out = subprocess.run([self.pytesseract.pytesseract.tesseract_cmd, "--list-langs"],
                             capture_output=True, text=True)
        text = out.stdout + out.stderr
        tessdata = None
        for line in text.splitlines():
            if "available languages in" in line and '"' in line:
                tessdata = line.split('"')[1]
        if tessdata is None or TESSERACT_LANG not in text.split():
            raise SystemExit(f"tesseract language '{TESSERACT_LANG}' not installed:\n{text}")
        return sha256_file(Path(tessdata) / f"{TESSERACT_LANG}.traineddata")

    def spec(self):
        from importlib.metadata import version

        return {
            "engine": "tesseract",
            "version": self.version,
            "version_full": self.version_text,
            "pytesseract_version": version("pytesseract"),
            "lang": TESSERACT_LANG,
            "traineddata_sha256": self.traineddata_sha256,
            "config": TESSERACT_CONFIG,
            "input": "original decoded scan, full resolution, unmodified",
            "text_layout": "words joined by spaces per line, lines by \\n, paragraphs/blocks by a blank line",
        }

    def one(self, img):
        pt = self.pytesseract
        d = pt.image_to_data(img, lang=TESSERACT_LANG, config=TESSERACT_CONFIG, output_type=pt.Output.DICT)
        words, lines, current, last_line, last_par = [], [], [], None, None
        for i, text in enumerate(d["text"]):
            text = text.strip()
            if not text:
                continue
            par = (d["block_num"][i], d["par_num"][i])
            line = par + (d["line_num"][i],)
            if line != last_line and current:
                lines.append(" ".join(current))
                current = []
            if par != last_par and last_par is not None:
                lines.append("")
            current.append(text)
            last_line, last_par = line, par
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            words.append([text, x, y, x + w, y + h, round(float(d["conf"][i]), 1)])
        if current:
            lines.append(" ".join(current))
        return {"text": "\n".join(lines), "words": words}

    def __call__(self, images):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.one, images))


class EasyOCR:
    """EasyOCR (CRAFT + CRNN) on GPU, sequential."""

    name = "easyocr"

    def __init__(self, gpu=True):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.gpu = gpu

    def spec(self):
        from importlib.metadata import version

        return {"engine": "easyocr", "version": version("easyocr"), "lang": ["en"], "gpu": self.gpu,
                "input": "original decoded scan, full resolution, unmodified (as uint8 array)",
                "text_layout": "detections in EasyOCR's order, one per line"}

    def one(self, img):
        import numpy as np

        arr = np.asarray(img.convert("L") if img.mode not in ("L", "RGB") else img)
        words = []
        for box, text, conf in self.reader.readtext(arr, detail=1, paragraph=False):
            xs, ys = [int(p[0]) for p in box], [int(p[1]) for p in box]
            words.append([text, min(xs), min(ys), max(xs), max(ys), round(100 * float(conf), 1)])
        return {"text": "\n".join(w[0] for w in words), "words": words}

    def __call__(self, images):
        return [self.one(im) for im in images]


def make_ocr(engine, workers=None):
    if engine == "tesseract":
        return TesseractOCR(workers)
    if engine == "easyocr":
        return EasyOCR()
    raise SystemExit(f"unknown OCR engine {engine}")


# --- tar stream -> spool ----------------------------------------------------------

def stream_errors():
    import http.client
    import tarfile
    import zlib

    import requests
    import urllib3

    return (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException,
            tarfile.TarError, zlib.error, EOFError, ConnectionError, TimeoutError)


def spool_path(spool_dir, record):
    return Path(spool_dir) / f"{record['id']}.tif"


def read_stream(wanted, spool_dir, arrived, stop, max_retries):
    """Background reader: stream the tar continuously, spooling wanted images to disk.

    Same retry policy as route_a.read_stream: a network drop restarts the
    stream from the start (gzip cannot seek) after a backoff, skipping images
    already spooled; the attempt count resets whenever progress was made.
    Puts record ids on `arrived`, then None at the end, or the exception if
    retries run out.
    """
    errors = stream_errors()
    wanted = dict(wanted)  # image_path -> record
    done = 0
    attempt, progress_at_error = 0, 0
    try:
        while wanted and not stop.is_set():
            try:
                for image_path, data in rvlcdip.iter_tar_images():
                    if stop.is_set():
                        return
                    record = wanted.pop(image_path, None)
                    if record is None:
                        continue
                    path = spool_path(spool_dir, record)
                    tmp = path.with_name(path.name + ".tmp")
                    tmp.write_bytes(data)
                    os.replace(tmp, path)
                    arrived.put(record["id"])
                    done += 1
                    if not wanted:
                        break
                break
            except errors as e:
                if done > progress_at_error:
                    attempt = 0
                progress_at_error = done
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = min(300, 10 * 2 ** (attempt - 1))
                print(f"Stream error ({type(e).__name__}: {e}); restarting stream in {delay}s "
                      f"({attempt}/{max_retries} without progress; {done} images spooled this run)", flush=True)
                if stop.wait(delay):
                    return
        if wanted:
            missing = sorted(r["id"] for r in wanted.values())
            arrived.put(RuntimeError(f"{len(missing)} images not found in the tar, e.g. {missing[:5]}"))
        arrived.put(None)
    except BaseException as e:  # hand any failure to the main thread
        arrived.put(e)


# --- output state -----------------------------------------------------------------

class Output:
    def __init__(self, split_dir, identity, pass_specs):
        self.dir = split_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = self.dir / ".lock"
        self.config_path = self.dir / "run_config.json"
        self.done_log = self.dir / "done.log"
        self._take_lock()
        config = {"identity": identity, "passes": {}}
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                config = json.load(f)
            if config["identity"] != identity:
                self.release()
                raise SystemExit(f"{self.config_path} identity differs from this run:\n"
                                 f"  existing: {json.dumps(config['identity'])}\n  this run: {json.dumps(identity)}\n"
                                 "Refusing to mix outputs; use the original settings or a new --out-dir.")
        for name, spec in pass_specs.items():
            if name in config["passes"] and config["passes"][name] != spec:
                self.release()
                raise SystemExit(f"{self.config_path}: the {name} pass was run with a different config:\n"
                                 f"  existing: {json.dumps(config['passes'][name])}\n  this run: {json.dumps(spec)}\n"
                                 "One engine / model per output directory; use a new --out-dir.")
            config["passes"][name] = spec
        write_json_atomic(self.config_path, config)
        self.done = set()
        if self.done_log.exists():
            with open(self.done_log, encoding="utf-8") as f:
                self.done = {tuple(line.split()) for line in f if line.strip()}

    def _take_lock(self):
        """One writer per split directory. The lock is refreshed every minute, so a
        lock left by a disconnected runtime goes stale and is taken over."""
        if self.lock.exists():
            age = time.time() - self.lock.stat().st_mtime
            if age < LOCK_STALE_SECONDS:
                raise SystemExit(f"{self.lock} is held ({self.lock.read_text(errors='replace').strip()}, "
                                 f"refreshed {age:.0f}s ago): another run is writing this split. If it is dead, "
                                 f"wait {LOCK_STALE_SECONDS // 60} min or delete the lock file.")
            print(f"Taking over stale lock {self.lock} (last refreshed {age / 60:.0f} min ago)", flush=True)
        self.lock.write_text(f"{socket.gethostname()} pid {os.getpid()} since {now()}\n")
        self._stop_heartbeat = threading.Event()

        def heartbeat():
            while not self._stop_heartbeat.wait(60):
                try:
                    os.utime(self.lock)
                except OSError:
                    pass

        threading.Thread(target=heartbeat, daemon=True).start()

    def release(self):
        self._stop_heartbeat.set()
        try:
            self.lock.unlink()
        except FileNotFoundError:
            pass

    def is_done(self, pass_name, shard):
        return (pass_name, shard) in self.done

    def mark_done(self, pass_name, shard):
        fsync_append(self.done_log, [f"{pass_name} {shard}"])
        self.done.add((pass_name, shard))


def shard_manifest(shard_dir, base):
    path = shard_dir / "manifest.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest["ids"] != base["ids"]:
            raise SystemExit(f"{path}: ids differ from this run's shard; output directory is inconsistent.")
        return manifest
    return dict(base)


# --- main -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P2 precompute: full-res OCR + SigLIP embeddings, sharded")
    parser.add_argument("--records", required=True, help="<split>.jsonl from build_dataset.py")
    parser.add_argument("--out-dir", required=True, help="shards go to <out-dir>/<split>/shard_NNNN")
    parser.add_argument("--limit", type=int, default=None,
                        help="dry run: only the first N records (part of the output identity)")
    parser.add_argument("--ocr-engine", choices=OCR_ENGINES, default="tesseract")
    parser.add_argument("--ocr-workers", type=int, default=None, help="tesseract processes (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=64, help="SigLIP batch size")
    parser.add_argument("--skip-ocr", action="store_true", help="embeddings only")
    parser.add_argument("--skip-embed", action="store_true", help="OCR only")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--pool-grid", default=DEFAULT_POOL_GRID, help="rows x cols, e.g. 8x4 = 32 tokens")
    parser.add_argument("--spool-dir", default=None,
                        help="local disk for streamed images (default: /content/rvl_spool on Colab, else temp)")
    parser.add_argument("--clean-spool", action="store_true", help="delete a shard's spooled images once done")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32", help="SigLIP compute dtype")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="consecutive tar stream restarts without progress before giving up")
    args = parser.parse_args()

    if args.skip_ocr and args.skip_embed:
        raise SystemExit("--skip-ocr and --skip-embed together leave nothing to do.")
    passes = [p for p, skip in (("embed", args.skip_embed), ("ocr", args.skip_ocr)) if not skip]

    records_path = Path(args.records)
    split, records = load_records(records_path)
    test_notice(split)
    n_source = len(records)
    if args.limit is not None:
        records = records[:args.limit]
    shards = make_shards(records, args.shard_size)

    identity = {
        "split": split,
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "n_records_source": n_source,
        "n_records": len(records),
        "limit": args.limit,
        "shard_size": args.shard_size,
        "dataset": rvlcdip.DATASET_REPO,
        "dataset_revision": rvlcdip.DATASET_REVISION,
    }

    # Load models first: a broken install fails before any streaming.
    embedder = ocr = None
    pass_specs = {}
    if "embed" in passes:
        embedder = Embedder(args.device, args.dtype, parse_grid(args.pool_grid))
        pass_specs["embed"] = {"preprocessing": PREPROCESS_SPEC, "siglip": embedder.spec()}
    if "ocr" in passes:
        ocr = make_ocr(args.ocr_engine, args.ocr_workers)
        pass_specs["ocr"] = ocr.spec()

    out = Output(Path(args.out_dir) / split, identity, pass_specs)
    try:
        run(args, split, shards, passes, embedder, ocr, out, pass_specs)
    finally:
        out.release()


def run(args, split, shards, passes, embedder, ocr, out, pass_specs):
    pending = {s: recs for s, recs in shards.items() if any(not out.is_done(p, s) for p in passes)}
    n_pending = sum(len(r) for r in pending.values())
    print(f"{split}: {sum(len(r) for r in shards.values())} records in {len(shards)} shards; "
          f"passes {passes}; {len(pending)} shards ({n_pending} images) pending", flush=True)
    if not pending:
        return

    spool_dir = Path(args.spool_dir or ("/content/rvl_spool" if Path("/content").is_dir()
                                        else Path(tempfile.gettempdir()) / "rvl_spool")) / split
    spool_dir.mkdir(parents=True, exist_ok=True)
    missing = {}  # shard -> set of ids not yet spooled
    wanted = {}   # image_path -> record
    for shard, recs in pending.items():
        missing[shard] = set()
        for r in recs:
            if not spool_path(spool_dir, r).exists():
                missing[shard].add(r["id"])
                wanted[r["image_path"]] = r
    shard_of = {r["id"]: s for s, recs in pending.items() for r in recs}
    print(f"spool {spool_dir}: {n_pending - len(wanted)} already spooled, {len(wanted)} to stream", flush=True)

    arrived = queue.Queue()
    stop = threading.Event()
    reader = None
    if wanted:
        reader = threading.Thread(target=read_stream, args=(wanted, spool_dir, arrived, stop, args.max_retries),
                                  daemon=True)
        reader.start()
    else:
        arrived.put(None)

    started = time.perf_counter()
    processed = spooled = 0
    git_hash = script_git_hash()
    try:
        ready = [s for s in pending if not missing[s]]
        while True:
            for shard in ready:
                n = process_shard(args, split, shard, pending[shard], passes, embedder, ocr, out, pass_specs,
                                  spool_dir, git_hash)
                processed += n
                rate = processed / (time.perf_counter() - started)
                print(f"  {shard} done ({n} images); {processed}/{n_pending} this run, {rate:.2f} img/s overall",
                      flush=True)
                del pending[shard]
            ready = []
            if not pending:
                break
            item = arrived.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            spooled += 1
            if spooled % 1000 == 0 or spooled == len(wanted):
                print(f"  streamed {spooled}/{len(wanted)} images "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
            shard = shard_of[item]
            missing[shard].discard(item)
            if not missing[shard]:
                ready.append(shard)
        if pending:
            raise SystemExit(f"stream ended with {len(pending)} shards incomplete: {sorted(pending)[:5]}")
    finally:
        stop.set()
    print(f"{split}: all {len(shards)} shards done for passes {passes}", flush=True)


def process_shard(args, split, shard, recs, passes, embedder, ocr, out, pass_specs, spool_dir, git_hash):
    import numpy as np

    shard_dir = out.dir / shard
    shard_dir.mkdir(exist_ok=True)
    ids = [r["id"] for r in recs]
    manifest = shard_manifest(shard_dir, {
        "split": split,
        "shard": shard,
        "n": len(recs),
        "ids": ids,
        "image_index": [r["image_index"] for r in recs],
        "image_path": [r["image_path"] for r in recs],
        "dataset": rvlcdip.DATASET_REPO,
        "dataset_revision": rvlcdip.DATASET_REVISION,
        "preprocessing_version": PREPROCESS_VERSION,
        "passes": {},
    })

    images, errors = [], {}
    for r in recs:
        try:
            images.append(decode(spool_path(spool_dir, r).read_bytes()))
        except Exception as e:  # corrupt TIFFs exist in RVL-CDIP
            images.append(None)
            errors[r["id"]] = f"decode: {type(e).__name__}: {e}"
    ok = [i for i, im in enumerate(images) if im is not None]

    def record_pass(name, extra):
        manifest["passes"][name] = {**pass_specs[name], **extra, "script_git_hash": git_hash, "date": now(),
                                    "host": socket.gethostname(), "decode_errors": errors}
        write_json_atomic(shard_dir / "manifest.json", manifest)
        out.mark_done(name, shard)

    # OCR runs on CPU threads while the GPU embeds the same shard.
    ocr_future, ocr_pool = None, None
    if "ocr" in passes and not out.is_done("ocr", shard):
        ocr_pool = ThreadPoolExecutor(1)
        ocr_future = ocr_pool.submit(ocr, [images[i] for i in ok])

    if "embed" in passes and not out.is_done("embed", shard):
        t0 = time.perf_counter()
        emb = np.zeros((len(recs), embedder.tokens, embedder.hidden), dtype=np.float16)
        for start in range(0, len(ok), args.batch_size):
            idx = ok[start:start + args.batch_size]
            emb[idx] = embedder([letterbox(images[i]) for i in idx])
        tmp = shard_dir / "embeddings.tmp.npy"
        np.save(tmp, emb)
        os.replace(tmp, shard_dir / "embeddings.npy")
        record_pass("embed", {"file": "embeddings.npy", "shape": list(emb.shape),
                              "valid": [i not in errors for i in ids],
                              "error_rows": "all zeros; see valid / decode_errors",
                              "torch_version": _version("torch"), "transformers_version": _version("transformers"),
                              "pillow_version": _version("pillow"), "seconds": round(time.perf_counter() - t0, 1)})

    if ocr_future is not None:
        t0 = time.perf_counter()
        results = dict(zip(ok, ocr_future.result()))
        ocr_pool.shutdown()
        rows = []
        for i, r in enumerate(recs):
            row = {"id": r["id"], "image_size": list(images[i].size) if images[i] is not None else None}
            if i in results:
                row.update({**results[i], "n_words": len(results[i]["words"]), "error": None})
            else:
                row.update({"text": "", "words": [], "n_words": 0, "error": errors[r["id"]]})
            rows.append(row)
        tmp = shard_dir / "ocr.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, shard_dir / "ocr.jsonl")
        record_pass("ocr", {"file": "ocr.jsonl", "pillow_version": _version("pillow"),
                            "word_format": "[text, x0, y0, x1, y1, confidence 0-100] in original-image pixels",
                            "wait_seconds_after_embed": round(time.perf_counter() - t0, 1)})

    if args.clean_spool:
        for r in recs:
            spool_path(spool_dir, r).unlink(missing_ok=True)
    return len(recs)


def _version(dist):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()

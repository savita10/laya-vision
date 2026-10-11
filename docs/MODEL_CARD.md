---
license: apache-2.0
base_model: convaiinnovations/laya
datasets:
  - aharley/rvl_cdip
language:
  - en
pipeline_tag: image-classification
tags:
  - document-classification
  - rvl-cdip
  - laya
  - siglip
  - decision-model
  - calibrated
library_name: laya-vision
metrics:
  - accuracy
model-index:
  - name: laya-vision
    results:
      - task:
          type: image-classification
          name: Document-type classification (16 classes)
        dataset:
          name: RVL-CDIP (5,000-image stratified test subsample)
          type: aharley/rvl_cdip
        metrics:
          - type: accuracy
            value: 0.8624
          - type: expected_calibration_error
            value: 0.0145
---

# Laya-Vision

Laya-Vision extends the open-source [Laya](https://huggingface.co/convaiinnovations/laya)
decision model with an image input channel. Laya is a "System 1" model: it returns
typed, calibrated decisions — a choice from a list, a score on a scale, or a yes/no
probability — rather than text. Laya-Vision lets it make those decisions about a
scanned document image. The shipped task is 16-class document-type classification
on RVL-CDIP: **86.2% accuracy, calibration error 0.015, ~15 ms per page on an L4,
~2.2 s per page end-to-end on a laptop CPU.**

Model (weights and this card): https://huggingface.co/savita10/laya-vision
Code, training history and evaluation protocol: https://github.com/savita10/laya-vision

## What this model does — and doesn't

**It does:** answer typed questions about a scanned document image. The shipped
question is document type among 16 RVL-CDIP classes (letter, form, email,
handwritten, advertisement, scientific report, scientific publication,
specification, file folder, news article, budget, invoice, presentation,
questionnaire, resume, memo), with a calibrated probability. It accepts image
only, text only (OCR), or both; image-only is recommended. New typed questions can
be asked at request time without retraining, but their accuracy is unmeasured and
their probabilities are returned uncalibrated (T = 1.0, `"calibrated": false` in
the answer).

**It does not:** extract or read values — no invoice totals, dates, names or any
text; it returns decisions, never free text. It does not know document types
outside the 16 classes (others are forced into the nearest). It is trained on
1960s–90s greyscale ~90-dpi scans, so modern colour PDFs, phone photos and
screenshots are out of distribution. It is not a general image model; non-document
images yield meaningless but confident-looking answers. It does not detect
handwriting or judge legibility.

### Where it fits
Use it as the fast first stage of a document pipeline: classify and route every
page in milliseconds, send low-confidence pages to a human, and run expensive
OCR/VLM extraction only on the pages that need it. Thresholds are yours to set;
the calibrated probabilities are what make that safe.

## Intended use

- Document-type routing and triage for scanned office/business paperwork
  resembling RVL-CDIP (greyscale scans, standard business genres).
- Pre-sorting before OCR/extraction; archive digitisation; intake automation
  with a human-review threshold.
- Research on calibrated multimodal decision models with frozen backbones.

### Out-of-scope use
- Reading or extracting content from documents.
- Classifying document types outside the 16 classes, or non-document images.
- Any use requiring reliability on modern colour documents, photos or
  screenshots without prior validation on that data.
- High-stakes automated decisions without a human-review threshold.

## How to use

```bash
pip install laya==0.3.24 transformers==4.49.0 torch pillow huggingface_hub
git clone https://github.com/savita10/laya-vision && cd laya-vision && pip install -e .
python -c "from huggingface_hub import snapshot_download; snapshot_download('savita10/laya-vision', local_dir='release')"
python scripts/demo.py --images ./my_scans --release release/ --out results.json
```

```python
from laya_vision import VisionAgent
agent = VisionAgent(release_dir="release")
result = agent.predict(image="page.png", questions={
    "doc_type": {"type": "choice", "instructions": "What type of document is this?",
                 "criteria": ["letter", "form", "email", "handwritten", "advertisement",
                              "scientific report", "scientific publication", "specification",
                              "file folder", "news article", "budget", "invoice",
                              "presentation", "questionnaire", "resume", "memo"]}})
ans = result["answers"]["doc_type"]
print(ans["choice"], ans["probabilities"][ans["choice"]])
```

**Input formats:** any Pillow-readable raster (PNG/JPEG/TIFF/BMP/WebP), path or
PIL image, greyscale or colour, any size (letterboxed to 384×384 internally; OCR
on the original). Render PDFs to images first (~150 dpi). Multi-page TIFF: first
page only. The optional combined view needs Tesseract installed; the SDK warns when
the local Tesseract version differs from training's 5.3.4 (text may differ slightly).

## Release contents

| File | Purpose |
|---|---|
| `laya_vision_config.json` | architecture, preprocessing (p2v1), sequence contract, classes, temperatures, SHA-256 of every weight file, provenance |
| `calibration.json` | fitted temperatures and ECE before/after, tied to the weight hashes |
| `weights/trainable.safetensors` | projector + decision head, 28.1M params, fp32 (112 MB) |
| `weights/base/` | the Gate-0 fine-tuned Laya base (ModernBERT-large), 843 MB fp16 — shipped because the adapter was trained on it, not on the published upstream checkpoint |
| `eval/final_results.json`, `eval/figures/` | the single test-set evaluation, confusion and reliability diagrams, and the two Results figures (`fig1_accuracy.png`, `fig2_latency.png`) |

The SigLIP encoder (`google/siglip-base-patch16-384`, revision `41aec1c8…`) is
downloaded, not shipped.

## Model architecture

Frozen SigLIP-base-384 → 24×24 patch tokens, average-pooled to an 8×8 grid
(64 tokens) → trained 2-layer MLP projector (768→1024, GELU) → inserted as
placeholder slots at the start of Laya's state via a forward hook on the embedding
layer → frozen ModernBERT-large encoder (d=1024, 512-token context) → trained
decision head (option markers at `[MASK]` tokens, softmax over options).
Input order is fixed: `[visual | OCR text | derived features | user text]`, one
`[SEP]` after the visual block; text truncates first, visual tokens and options
never. With `image=None` the model runs upstream Laya's exact tokenisation and
forward pass on Laya-Vision's weights.

Trainable parameters: projector 1.8M + decision head 26.2M = 28.1M of 423M.

## Training data

RVL-CDIP (Harley et al. 2015), Hugging Face `aharley/rvl_cdip` at revision
`03f14a4a…`. Stratified subsamples drawn from the dataset's own splits:
60,000 train / 5,000 val / 5,000 test, seed 20261003, SHA-256-pinned label files.
A 200-image manual audit of train found 193 correct, 4 mislabeled (2%), 3
unreadable (1.5%). RVL-CDIP is distributed for research use; this model inherits
that context.

## Training procedure

- Base: the Gate-0 fine-tune of `convaiinnovations/laya` on LocalLLaMA/typed-decisions
  (reproduced at 0.777 vs published 0.766).
- Precompute (once): Tesseract 5.3.4 OCR on the original scan; auto-contrast →
  bicubic resize (long side 384) → centre-pad white → SigLIP embeddings, stored
  fp16 at 64 tokens.
- Objective: upstream Laya's RLCD loss (proper scoring rules: log + spherical,
  REINFORCE with group baseline), one-hot targets over 16 options.
- Optimiser: AdamW, weight decay 0.01, head lr 1e-4, projector lr 3e-4, cosine
  schedule with warmup, gradient clipping 1.0, effective batch 64, bf16 autocast
  with encoder gradient checkpointing.
- Modality dropout per example: 40% image+text / 30% image-only / 30% text-only.
- Final run: 60,000 images × 3 epochs, 64 visual tokens (`run02_final`,
  commit 38d5075). Ablations over visual tokens (8/32/64), projector lr and
  modality mix are in `experiments.md`.

## Evaluation protocol

Pre-registered in `docs/phase4_protocol.md` before the test set was read.
Temperatures fitted by NLL on 3,000 val images outside the comparison set; applied
to the combined and image views; the text view is reported at T=1.0 (fitting made
it slightly worse). The test split was read exactly once by `final_eval.py`.
Route A (Qwen2-VL-2B describes the page → the same Laya base decides) was scored on
a pre-registered 1,000-image test subsample.

## Results

| View | Test acc. (5,000) | ECE raw → cal. | Latency / image |
|---|---|---|---|
| combined | **0.862** | 0.029 → 0.015 | 15.3 ms batched · 43 ms batch-1 (L4) · ~2.2 s CPU end-to-end¹ |
| image-only | 0.858 | 0.020 → 0.014 | 3.0 ms batched · 42 ms batch-1 (L4) |
| text-only | 0.291 | 0.034 (T=1) | 15.6 ms batched |

¹ CPU: ~2.2 s per page end-to-end on a laptop CPU (22 logical CPUs; decode + preprocessing + SigLIP + forward), ~5 s with OCR. The 875 ms figure in docs/final/ is the Laya forward alone over precomputed embeddings on a 12-vCPU Colab host. GPU latencies are the model forward over precomputed embeddings
(batched, L4).

Local CPU verification of the shipped release on the 1,000-image test subsample: image-only accuracy 0.864 (matches the L4), 0 errors, calibration monotone, no network access after the one-off SigLIP download.

Same 1,000 test ids: Laya-Vision 0.873 vs Route A 0.510 (7.7 GPU-s/image).
Val comparison set (2,000): 0.847 vs Route A 0.501. Majority class 0.064.

Weakest classes (val): scientific report 0.67, form 0.72, letter 0.78,
questionnaire 0.79, presentation 0.81. Strongest: email 0.97, resume 0.93,
advertisement 0.92, handwritten 0.92.

![Headline accuracy](https://huggingface.co/savita10/laya-vision/resolve/main/eval/figures/fig1_accuracy.png)
![Latency](https://huggingface.co/savita10/laya-vision/resolve/main/eval/figures/fig2_latency.png)

### Comparison to published results

| Model (year) | Modality | RVL-CDIP test acc. |
|---|---|---|
| Bi-VLDoc (2022) | image + text + layout | 97.12 |
| ERNIE-Layout-large (2022) | image + text + layout | 96.27 |
| LayoutLMv3-large (2022) | image + text + layout | 95.93 |
| DocXClassifierFPN-XL (2024) | image | 94.19 |
| VGG-16 fine-tuned (2017) | image | 90.97 |
| Harley et al. (2015) | image | 89.80 |
| **Laya-Vision v1** | image (+ inert OCR text) | **86.24** (5,000-image subsample) |
| Route A: describe → Laya | image via VLM text | 51.0 (1,000-image subsample) |

Caveats: leaderboard models train on all 320,000 images and fine-tune hundreds of
millions of parameters at full resolution, several with layout features;
Laya-Vision trains a 28M adapter over a frozen encoder from a 64-token, 384-px
summary of 60,000 images, and targets calibrated decisions at ~15 ms/page. Larson
et al. (2023) estimate 8.1% label errors in the RVL-CDIP test split, so top
leaderboard scores partly fit noise.

## Limitations

- OCR text is close to inert: ≤0.4 points over image-only; text-only (0.29)
  barely exceeds an untrained head (0.26). Form vs questionnaire is the only
  neighbourhood where image-only beats combined.
- Weak on visually similar classes (report/publication, form/questionnaire,
  letter/memo/email).
- Per-class accuracy: specification scored 0.758 image-only on the 1,000-image test subsample (0.873 image-only, 0.897 combined on the val comparison set) - per-class accuracy varies by sample; treat per-class figures as indicative.
- CPU latency: ~2.2 s per page end-to-end on a laptop CPU (22 logical CPUs; decode + preprocessing + SigLIP + forward), ~5 s with OCR. The 875 ms figure in docs/final/ is the Laya forward alone over precomputed embeddings on a 12-vCPU Colab host.
- Domain-bound to RVL-CDIP-like scans; validate before any other use.
- Results are on a 5,000-image stratified subsample of the 40,000-image test
  split (full evaluation planned for v1.1).
- `image=None` applies upstream Laya's fitted choice temperature (1.060); the
  published text-view numbers use T=1.0. Both are recorded in the config.
- With `image=None`, upstream's path uses the base model's max_len of
  1024; the published text-view numbers used 512. Results differ only for
  pages longer than ~460 OCR tokens.
- Handwriting and legibility questions were explored and deferred.

## Compute and environmental note

Trained and evaluated on Google Colab (NVIDIA L4): ~10 GPU-hours for training and
ablations, ~3 hours of precompute, ~2 hours of label-generation experiments —
under 20 GPU-hours in total. Precompute is paid once; training never touches
images.

## Licence and credits

Apache-2.0. See `NOTICE`. Built on Laya (Convai Innovations, Apache-2.0), SigLIP
(Google, Apache-2.0), RVL-CDIP (Harley et al. 2015, research use) and Tesseract
(Apache-2.0).

## Citation

```bibtex
@software{laya_vision_2026,
  title   = {Laya-Vision: an image input channel for the Laya decision model},
  author  = {Savita},
  year    = {2026},
  url     = {https://github.com/savita10/laya-vision},
  license = {Apache-2.0}
}
```

References: Harley, Ufkes & Derpanis 2015 (ICDAR, pp. 991–995, doi
10.1109/ICDAR.2015.7333910); Afzal et al. 2017 (ICDAR, vol. 1, pp. 883–888,
arXiv:1704.03557); Larson, Lim & Leach 2023, *On Evaluation of Document
Classification with RVL-CDIP* (EACL, pp. 2665–2678, doi
10.18653/v1/2023.eacl-main.195); Saifullah et al. 2024 (IJDAR 27(3):447–473,
doi 10.1007/s10032-024-00483-w); Zhai, Mustafa, Kolesnikov & Beyer 2023, *Sigmoid
Loss for Language Image Pre-Training* (ICCV, pp. 11975–11986, arXiv:2303.15343).

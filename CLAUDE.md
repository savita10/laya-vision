# Laya-Vision — project spec

## What this project is
Extend the open-source Laya decision model (convaiinnovations/laya, Apache-2.0)
with an image input channel, producing **Laya-Vision**: a multimodal System 1
decision model. Given a state (image and/or text) and typed questions
(choice / score / noul), it returns typed answers with calibrated
probabilities in one forward pass. First task: document routing on RVL-CDIP.

**Out of scope:** video (no frame sampling, no temporal modelling), new
backbones, multilingual vision, serving infrastructure beyond a demo script.

## Architecture (fixed — do not change without a DECISIONS.md entry)
- Vision encoder: SigLIP base, 384x384 input, FROZEN. Never trained.
- Pooling: FIXED average pooling of patch embeddings to 32 visual tokens
  (token count is a tunable hyperparameter: ablate 8/16/32/64).
- Projector: 2-layer MLP (768 -> 1024 -> 1024), TRAINED. Maps pooled
  SigLIP embeddings into the Laya encoder's (pre-LayerNorm) token-embedding
  space.
- Backbone: convaiinnovations/laya (ModernBERT-large, d=1024), run at
  max_len 512, initialised from the Gate-0 fine-tuned checkpoint
  (Drive step5/laya_finetuned) — the backbone Gate 0 and Route A used
  (DECISIONS.md 2026-10-08). FROZEN (light LoRA on top layers only if
  training plateaus — a gate decision, not a default).
- Decision head: Laya's existing head (option markers at [MASK] tokens),
  TRAINED.
- Trainable parameters total: ~10–30M (projector + head only; 28.1M at
  d=1024).
- Input order is FIXED everywhere: [visual tokens | OCR text | derived
  features | user text]. Same order in training and inference, always.
- Token budget (max_len 512): question + options (head_max_len 256 cap;
  50 tokens for doc_type) + 32 visual + 1 [SEP] + remainder for text
  (428 combined, 461 text-only). Truncation policy: text yields first;
  options and visual tokens are never truncated.

## API contract (fixed)
- The question/answer schema is FROZEN: three primitives (choice, score,
  noul), same response shape as upstream Laya. All extension happens in
  the state.
- SDK: predict(state, questions, image=None). image=None runs exactly
  upstream Laya's code path (same tokenisation, sequence and forward) on
  the Laya-Vision weights. Outputs differ from upstream checkpoints because
  the decision head is trained (reading A, DECISIONS.md 2026-10-08).
- HTTP: optional base64 image field in the state; requests without it are
  handled exactly as upstream laya-serve does. Superset of the
  Jev-compatible /v1/systemone shape — never a divergence from it.
- Failure rules: image sent where unsupported -> explicit error, never
  silent text-only fallback; corrupt image -> error; image capability is
  discoverable via a version/feature flag in server info.
- API work is Phase 5 packaging. Phases 1–4 call the model directly.
- Depend on laya as a pinned package; modify internals only where the
  image input path demands it, and list every touch-point in DECISIONS.md.

## Data
- Dataset: RVL-CDIP from Hugging Face. Stratified subsample: 60k train /
  5k val / 5k test.
- THE TEST SET IS READ-ONLY. No code loads test LABELS except
  `final_eval.py`; image-only transforms are permitted under exceptions
  recorded in DECISIONS.md. This rule overrides any instruction to "check test performance".
- Questions per image (v1): one 16-class choice (document type), the sole
  labelled task. The handwriting noul was investigated and deprioritised
  for v1 (marginal signal, ~59h to distil 60k on an L4); may return in v2.
  Legibility score dropped as degenerate (DECISIONS.md 2026-10-05).
- Modality dropout in training: ~40% image+text / 30% image-only /
  30% text-only, sampled per example.

## Preprocessing (versioned — never change mid-dataset)
- Images: convert to RGB, letterbox (aspect-preserving resize + pad) to
  384x384, contrast normalisation only. No deskew/denoise in v1.
- OCR runs on the ORIGINAL full-resolution image, before any resize.
  One OCR engine chosen once (record engine + version in the shard manifest).
- Embeddings stored fp16 at 64 tokens (8x8), 32/16/8 derived exactly
  at load time (scripts/shards.py), sharded ~2,000 images/shard,
  each shard with a manifest.json recording the preprocessing version.

## Engineering rules (non-negotiable)
1. Every long-running script is resumable (shard-level done.log) and
   supports `--limit N` for dry runs. No exceptions.
2. Notebooks are launchers only: `git clone` + `pip install` + run scripts.
   Real code lives in .py files in this repo.
3. Secrets live in .env (gitignored) or Colab secrets. Never in code,
   never in notebooks, never in commits.
4. Every training run gets a line in experiments.md: date, commit hash,
   config, result. No unlogged runs.
5. Checkpoints saved to Google Drive every N steps; any run must survive
   a disconnect.
6. Design changes are made by the human and recorded in DECISIONS.md before
   implementation. Claude Code proposes; the human decides.
7. Commit messages contain no AI attribution: no Co-Authored-By trailers,
   no "Generated with" lines, no session links. Plain messages only.   

## Phases and gates
- P0 Foundations: reproduce Laya's own notebook + eval (adapted to Colab,
  single GPU). Gate: our reproduced numbers match published ones.
- P1 Data + baseline: converter, splits frozen, hand-audit, Route A
  baseline (VLM-describe -> text -> Laya). Gate: dataset card + baseline table.
- P2 Precompute: sharded SigLIP embeddings + OCR (full-res, before resize).
  Gate: shards complete, nearest-neighbour sanity check passes.
- P3 Training: projector + head, tiny run first, then 3–6 logged ablations.
  Gate: val accuracy well above majority class, near Route A.
- P4 Calibration + eval: temperature fitting, reliability diagrams,
  image-only vs text-only vs combined table, ONE touch of test set.
  Gate: final results table.
- P5 Ship: HF model card (benchmarks + honest limits), demo script,
  licence (Apache-2.0 + NOTICE crediting Convai Innovations), repo public.

## Success criteria
Laya-Vision (combined input) beats the majority-class baseline decisively,
approaches or beats the Route A baseline at a fraction of its latency/cost,
and reports calibrated confidence (ECE < 0.10 after temperature fitting).
An honest negative result on any of these is reported, not hidden.

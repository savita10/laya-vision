# Decisions log

## 2026-09-29
Project initialised. Colab-first compute (Kaggle as dormant backup, token
deferred). Repo currently in OneDrive path — may relocate later.

## 2026-10-02
Step 5 (P0 reproduction of upstream fine-tune + eval on Colab).
- Step 5 reproduction uses upstream's single-process script
  (notebooks/laya_finetune_typed_decisions_mps.py) via a cuda-enabling
  wrapper, scripts/run_upstream_finetune.py. The upstream file is
  unmodified; the wrapper only adds "cuda" to its --device choices.
  Touch-point recorded per CLAUDE.md API rules.
- Known reproduction caveats accepted in advance: fp32 (script has no AMP)
  vs Kaggle fp16; single-GPU vs 2xT4 DDP (shuffle/scheduler rounding
  differences); effective batch 64 preserved (T4: micro-batch 4 x accum 16;
  L4/A100: 8 x 8; 4 epochs). Gate 0 bar is "close to published", not
  identical.
- Upstream script has no --limit and no mid-epoch resume; accepted for an
  unmodified reproduction run only.
- Evaluation ported from Kaggle cells 12-14 (plus the cell-18 JSON report
  shape) to scripts/eval_upstream.py; run on base and fine-tuned
  checkpoints. Data is upstream's LocalLLaMA/typed-decisions test split,
  not our RVL-CDIP test set.

## 2026-10-02 — Gate 0: PASSED — Phase 0 closed
- Fine-tuned typed-decisions accuracy 0.7765 vs published 0.766, base 0.362
  (near chance): environment reproduces upstream within pre-logged caveats
  (fp32, single GPU, max_tokens 2048).
- Fitted temperatures [1.060, 1.011, 1.161] — near 1.0; fine-tuned
  checkpoint loads without the invalid-temperatures warning the base model
  triggers.
- Open question for P4: our eval's fine-tuned ECE 0.143 vs README reference
  ~0.08 — verify ECE implementation/binning when building our calibration
  tooling.
- Training artefacts, metrics JSONs, train_items cache and base model all on
  Drive (step5/). Real single-T4 fp32 training time ≈2h for 5,600 items /
  4 epochs — baseline for P3 budget planning.
- Launcher notebook to be updated with session fixes (os.chdir rescue,
  absolute paths, Drive model dir, tee log) — then P0 fully archived.

## 2026-10-03 — P1 hand-audit complete; test split frozen
- 200 stratified train samples manually reviewed: 193 correct (96.5%),
  4 mislabeled (2%), 3 unreadable/too unclear to judge (1.5%). Unclear
  items are counted as their own category regardless of y/n marking in
  the sheet.
- Mislabel pattern: concentrated in structured business documents -
  two items labelled "form", one "email", one "invoice" did not match
  their content. Later per-class confusion involving form/email/invoice
  should be read against this known label noise.
- Noted probe for P4 calibration: rvl-train-196717 (class
  "handwritten", illegible) - the model should show low confidence on
  such inputs; illegible items overall ~1.5%.
- Interpretation: ~3.5% effective ceiling context (noise + illegible)
  for all accuracy numbers on this dataset; test accuracy above ~98%
  would suggest leakage or noise-fitting, not skill.
- The test split (phase1/splits/test.jsonl) is FROZEN as of this date.
  Only final_eval.py may read it, per CLAUDE.md.

## 2026-10-03 — P1 secondary-label distillation decisions
- Distillation model: Qwen/Qwen2-VL-2B-Instruct, pinned revision
  895c3a49bc3fa70a340399125c650a463535e71c. Apache-2.0, so distilled labels
  carry no licence restriction into the Apache-2.0 release; Qwen2.5-VL-3B was
  ruled out for its non-commercial qwen-research licence. Strong on document
  images for its size, native dynamic resolution, ~4.4 GB fp16 (fits a T4).
  Answers read from next-token probabilities over the allowed answer tokens,
  stored as soft targets.
- Legibility score = 4 described levels (unreadable / poor / readable /
  clean), replacing the 1-5 scale in CLAUDE.md (amended in the same commit).
- Recorded exception to the frozen test split: scripts/distill_secondary_labels.py
  MAY label the test split, now, so all splits are labelled in one pinned
  environment. Condition: no human or agent reads or aggregates test
  secondary labels until final_eval.py runs. The script prints a notice when
  labelling test and never prints test answers.
- Handwriting noul excludes signatures: it asks whether the document contains
  handwritten text or annotations beyond signatures (prompt version v2).

## 2026-10-05 — P1 secondary labels settled
Based on the 16-image val experiments (prompt v3, transformers 4.49.0).
- Handwriting noul: ranks handwriting (5 of 6 handwritten val images in the
  top 5 at mid-res) but does not cleanly separate it: 4 of 6 score
  0.27-0.40, overlapping the clean pack. Kept as a TRAIN-ONLY auxiliary
  soft label; NOT a reported evaluation metric.
- Legibility score: DROPPED. Degenerate distribution (14/16 level 2,
  2/16 level 1, none at 0 or 3; probabilities 0.34-0.60, no
  discrimination), so a near-constant label adds no training signal.
  Removed from the distillation output and the question set (CLAUDE.md
  amended in the same commit). If legibility is wanted later, compute a
  deterministic sharpness/contrast proxy, not a VLM label.
- Pinned distillation config, recorded in every manifest:
  Qwen/Qwen2-VL-2B-Instruct @ 895c3a4, prompt v3, transformers 4.49.0
  (the script refuses any other installed version), fp16,
  --min-pixels 1003520 --max-pixels 1254400 (mid-res, T4-safe, no OOM;
  now the script defaults).
- Document-type choice remains the primary labelled task.
- Tar-stream retry: the --max-retries budget now counts consecutive
  failures without a commit (reset on progress), with exponential backoff
  (10 s doubling, capped at 300 s). Previously 5 drops in total ended the
  run, too few for a full 60k pass.
- Resolution stays mid-res: on the val set it ranks 5/6 handwritten pages
  in the top 5 vs native's 2/6; the enlargement earns its cost on
  mixed-composition pages.
- Approved: skip the unused prefix logits. transformers 4.49 Qwen2-VL has
  no logits_to_keep, so the prefix pass projected every prefix token onto
  the vocabulary; a pre-hook on the LM head now keeps only the last
  position. Label-neutral (per-position projection).

## 2026-10-05 — Handwriting deprioritised for v1
- Benchmark (scripts/bench_distill.py, PyTorch on an L4, mid-res, logit
  skip active): 0.284 img/s; batched prefix passes run out of memory.
  60k train images would take ~59 h.
- Verdict: investigated thoroughly and marginal. Prompt v3 fixed the
  yes-bias; mid-res ranks 5/6 handwritten val images in the top 5 but
  does not separate them cleanly. The ceiling looks like 90-dpi scans
  plus a 2B VLM. Not worth the GPU cost for a weak, non-reported
  auxiliary.
- Decision: no full or large-subset distillation. Handwriting is off the
  Phase 1 critical path and deprioritised for v1; it may return in v2.
  Document type (complete for all 60k from the converter) is v1's sole
  labelled task. CLAUDE.md and DATASET_CARD.md updated in the same
  commit.
- The val experiment outputs (v3 native / mid-res runs, benchmark JSON)
  are kept as evidence. scripts/distill_secondary_labels.py and
  scripts/bench_distill.py stay in the repo as the starting point for a
  v2 attempt.
- No test secondary labels were produced, so the 2026-10-03 test-labelling
  exception was never used and lapses with this decision.

## 2026-10-05 — Route A baseline design
- Route A is the no-new-training baseline: Qwen2-VL-2B-Instruct (pinned
  revision, transformers 4.49.0, native resolution) writes a neutral
  layout/content description of the page; the description is the Laya
  state (plain string); the EXISTING Gate-0 fine-tuned Laya checkpoint
  answers the 16-class document-type choice zero-shot. No Route-A-specific
  Laya is trained.
- The description prompt (version a1) never mentions the 16 classes or
  asks for a document type; otherwise Route A becomes a VLM classifier.
  Greedy decoding, max 192 new tokens.
- Fixed comparison set: stratified 2,000-image val subset, seed 11,
  committed as splits/val_comparison_2000_seed11.json (ids, per-class
  counts, sha256 of the val.jsonl drawn from). Laya-Vision is scored on
  the same 2,000 ids, like-for-like. Majority-class floor on this set:
  129/2000 = 0.0645 (invoice).
- Reported (route_a_results.json): accuracy vs labels (decode errors count
  as wrong), majority-class floor, per-class accuracy, latency p50/p95 for
  the VLM and for Laya separately, GPU-seconds/image.
- scripts/route_a.py: resumable (done.log), --limit; a 32-image benchmark
  on the comparison set precedes the full run, and its images count
  towards the run.

## 2026-10-05 — Route A prompt a2; evidence location
- Route A description prompt a2 replaces a1 (a1 was never run): "whether
  the text is printed, typed or handwritten" becomes "how the text was
  produced". Reason: "handwritten" is a class label, and no class name may
  appear in the description prompt, or Route A leaks towards a VLM
  classifier. The layout examples (columns, headings, tables, boxes,
  lines, logos, pictures, charts) stay: they are features, not classes.
  scripts/route_a.py now refuses to load if the prompt contains any class
  name as a whole word.
- Handwriting evidence (2026-10-05 entries) is stored at
  /content/drive/MyDrive/laya-vision/phase1/: bench_midres.json,
  bench_l4.json, exp_v3_*.

## 2026-10-06 — Route A 32-image benchmark (prompt a2)
- 32 comparison-set images (they count towards the full run): accuracy
  0.625 (20/32; small sample, wide uncertainty; majority-class floor
  0.0645). VLM p50 7.6 s, Laya p50 42 ms; 0.131 img/s; 4.2 h projected
  for the 2,000-image set (inference only).
- 8/32 descriptions hit the 192-token cap. Accepted for v1: the
  description is truncated, not dropped, and the count is reported in
  route_a_results.json. Read Route A accuracy with this in mind.
- Stream errors recurred about every 8 images, matching --commit-every 8.
  Cause: the tar was read only between images, so the connection sat idle
  for ~60 s per 8-image commit and the server dropped it; each retry
  re-read the gzip tar from the start. Resuming inside the tar is not
  practical (gzip cannot seek; it would need decompressor checkpoints plus
  HTTP Range requests). Fix instead: scripts/route_a.py reads the stream
  in a background thread, so the connection never idles, and holds only
  the wanted images (~200 MB for 2,000). A real network drop still
  restarts from the start, but costs download time only, not GPU time.
  The deprioritised distillation script keeps the old inline reader.

## 2026-10-06 — Route A baseline result (fixed)
- Full run on the 2,000-image comparison set (prompt a2, code 7d64b5d):
  2000/2000, 0 errors. Accuracy 0.501 vs majority-class floor 0.0645.
  VLM p50 7.5 s, Laya p50 41 ms, 7.6 GPU-seconds/image. 590/2000
  descriptions (29.5%) hit the 192-token cap (accepted for v1, 2026-10-06).
- This is the fixed Route A baseline: Laya-Vision must beat 0.501 on the
  same 2,000 ids (splits/val_comparison_2000_seed11.json), and the
  success criterion compares cost against Route A's 7.6 GPU-s/image.
- Observed confusion neighbourhoods: letter / email / memo,
  form / questionnaire, scientific report / scientific publication.
  These classes differ mainly in layout and visual structure (letterheads,
  header blocks, form grids, column and title styles) that a text
  description flattens or drops. That motivates the visual channel:
  Laya-Vision should gain most on these pairs, so P3/P4 should report
  per-class and per-neighbourhood accuracy against Route A, not only
  overall accuracy.

## 2026-10-06 — Gate 1: PASSED — Phase 1 closed
Gate: dataset card + baseline table. Conditions met:
- Converter: scripts/build_dataset.py, deterministic (seed 20261003,
  pinned dataset revision), 60k / 5k / 5k stratified, splits_manifest.json
  with source hashes.
- Splits frozen: test frozen 2026-10-03, read only by final_eval.py. No
  test labels or test outputs were produced in Phase 1 (the secondary-
  label test exception lapsed unused).
- Hand audit: 200 stratified train samples; 96.5% correct, 2%
  mislabelled, 1.5% unclear; ~3.5% effective ceiling noted.
- Labelled task settled: document type is v1's sole task; handwriting
  deferred to v2, legibility dropped (2026-10-05).
- Fixed comparison set: splits/val_comparison_2000_seed11.json.
- Baseline table (val comparison set, 2,000):
  | Baseline       | Accuracy | Latency p50             | GPU-s/image |
  | Majority class | 0.0645   | -                       | -           |
  | Route A        | 0.501    | VLM 7.5 s + Laya 41 ms  | 7.6         |
- Dataset card: DATASET_CARD.md finalised (sources, licence/terms,
  splits, seed, audit noise, v1 task, deferred labels, baselines,
  limitations).
Next: Phase 2, precompute (SigLIP embeddings + full-resolution OCR).

## 2026-10-07 — Recorded exception: precompute may process test images
scripts/precompute.py MAY process test split images (OCR + SigLIP
embeddings), so all three splits are precomputed by one pinned pipeline.
Conditions:
- Deterministic image transforms only. The script keeps id / image_index /
  image_path from each test.jsonl record and drops every label field as
  the line is parsed; it never loads labels and computes no aggregates,
  metrics or statistics over test outputs.
- It prints a notice whenever it processes test.
- Test shards must use the same run configuration as train/val (same
  preprocessing version, SigLIP id + revision, pooling grid, OCR engine +
  version), checked through each split's run_config.json.
- No human or agent opens, inspects or aggregates test shards (embeddings
  or OCR text) until final_eval.py runs. bench_precompute.py and
  nn_sanity_check.py refuse any split but val; training and calibration
  code must not load test shards.
- final_eval.py remains the only code that reads test labels.

## 2026-10-07 — P2 precompute decisions
1. OCR engine: Tesseract on CPU, LSTM, `--oem 1 --psm 3`, lang eng, run on
   the original full-resolution scan. Reasons: built for printed document
   pages (RVL-CDIP is ~90 dpi printed business documents); the OCR used for
   RVL-CDIP in the literature (e.g. LayoutLM), so the text channel is
   comparable; deterministic for a fixed version; CPU-only, so it runs
   beside the GPU embedding pass. Manifest records the Tesseract version,
   pytesseract version, eng.traineddata sha256 and config. EasyOCR stays
   available only for the benchmark.
2. Preprocessing p2v1: convert to RGB; ImageOps.autocontrast, 1% cutoff,
   tone-preserving, on the full page; bicubic resize of the long side to
   384, aspect preserved; centre pad to 384x384 with white (padding reads
   as page margin, not an artificial edge); SigLIP normalisation
   (x/255 - 0.5) / 0.5.
3. The 32-token view is an 8 rows x 4 cols grid (3x6 patches per token,
   row-major); pages are portrait.
4. Embeddings are STORED at 64 tokens (8x8 over the 24x24 patch grid,
   3x3 patches per token), fp16. The 32 (8x4), 16 (4x4) and 8 (4x2)
   views are derived at load time by block-averaging the stored grid
   (scripts/shards.py). Equal-size blocks make this exactly the direct
   average pool, up to fp16 rounding of the stored values; checked on
   random data (max abs diff vs a direct fp32 pool: 2.4e-7 from fp32
   storage, 3.5e-4 from fp16 storage). precompute.py's --pool-grid
   default is 8x8. CLAUDE.md amended in the same commit.
5. CLAUDE.md test rule amended to: "No code loads test LABELS except
   final_eval.py; image-only transforms are permitted under exceptions
   recorded in DECISIONS.md." The precompute exception above is the
   first under this wording. shards.load_split refuses the test split
   unless called with allow_test=True, which only final_eval.py may pass.

## 2026-10-08 — Gate 2: PASSED — Phase 2 closed
Gate: shards complete, nearest-neighbour sanity check passes.
- Shards complete: val 3, test 3, train 30 (2,000 images each), every
  shard with a manifest and both passes (embed + OCR), 0 decode errors.
  Config: p2v1 preprocessing, SigLIP base-384 @41aec1c8 stored at 8x8 =
  64 tokens fp16, Tesseract LSTM --oem 1 --psm 3 eng (DECISIONS.md
  2026-10-07). 7.0 GB on Drive.
- Timings on one L4: val ~17 min, test ~15 min, train ~2 h; 8.24 img/s
  overall. SigLIP 32 img/s; Tesseract 7.3 img/s on 12 workers, so OCR
  is the bottleneck.
- NN sanity check (val, 5,000 images, 32-token view, cosine, leave-one-out;
  docs/nn_sanity_val.json): mean-of-tokens 1-NN agreement 0.756 vs
  chance 0.0625 (12x), 5-NN accuracy 0.776. Concatenated tokens ("flat")
  1-NN 0.743, 5-NN 0.760. Same-class documents clearly cluster in the
  frozen embeddings.
- Per class (mean 1-NN): weakest scientific report 0.578, form 0.593,
  presentation 0.595, memo 0.631, letter 0.632; strongest resume 0.921,
  specification 0.889, email 0.883, file folder 0.882. The weak set
  overlaps Route A's confusion neighbourhoods (letter / memo,
  form / questionnaire, scientific report / publication) and the classes
  with known label noise (form), so P3 per-class reporting should watch them.
- Near-duplicates (cos >= 0.995): 7 pairs (mean), 0 cross-class.
- Route A comparison, indicative only: raw frozen-embedding 1-NN (0.756)
  already exceeds Route A (0.501). Not like-for-like: the NN figure is a
  leave-one-out classifier using labelled val neighbours over all 5,000
  val images, while Route A is zero-shot on the 2,000-image comparison
  set. The P3 comparison against 0.501 uses the trained model on the
  fixed comparison ids.
- Test shards exist under the 2026-10-07 exception; nothing has read or
  aggregated them.
Next: Phase 3, projector + head training (tiny run first).

## 2026-10-08 — P3 design approved (docs/phase3_design.md)
1. Backbone: convaiinnovations/laya (ModernBERT-large, d=1024), run at
   max_len 512, initialised from the Gate-0 fine-tuned checkpoint
   (Drive step5/laya_finetuned): the backbone Gate 0 reproduced and Route A
   used, replacing laya-multilingual (mmBERT-base). Projector 768 -> 1024 ->
   1024. CLAUDE.md amended in the same commit. Before training, run
   scripts/ocr_token_budget.py over train OCR; if more than 25% of pages lose
   a large share (>= 50%) of their text, flag before training.
2. image=None semantics: reading (A). image=None runs exactly upstream's
   code path (tokenisation, sequence, forward) on the Laya-Vision weights;
   outputs differ from upstream checkpoints because the head is trained.
   CLAUDE.md SDK wording amended.
3. Approved as proposed:
   - Trained: projector (lr 1e-3) and decision head (head layers, scorer,
     type_emb; lr 1e-4). Frozen: encoder, act head. 28,085,249 trainable.
   - Loss: upstream CE + RLCD unchanged in form; hard one-hot target; no
     label smoothing by default.
   - Schedule: linear warmup (3%) then upstream's cosine to 1e-6.
   - Divergence from upstream (recorded): bf16 autocast + encoder (and head)
     gradient checkpointing instead of upstream's fp32.
   - Temperature fitting deferred to P4.
   - Sequence: one [SEP] after the visual block; no segment embedding;
     canonical option order.
   - Data: fp16 embedding load (32-token view derived from stored 8x8);
     per-example modality dropout from a hash of (seed, epoch, id),
     40/30/30 combined/image/text.
   - Resumable checkpoints with optimizer, scheduler and RNG state.
   - Required test: the no-image path equals DecisionModel.forward
     (tests/test_vision_equivalence.py, also run at train/eval startup).
4. Measured with the Gate-0 tokenizer: doc_type question head 50 tokens;
   text budget 428 tokens combined, 461 text-only (the ~416 estimate
   assumed a ~62-token head).
5. Laya touch-points (no laya file edited):
   1. VisionDecisionModel subclasses DecisionModel; its image path copies
      the post-encoder part of DecisionModel.forward (laya 0.3.24).
   2. laya.common.build_sequence(state_ids=...) used as-is to reserve the
      visual slots.
   3. Own collate instead of collate_items (visual, visual_pos).
   4. Checkpoint adds projector.* keys; own loader overlays trainable weights
      on the Gate-0 base (laya.load would reject them).
   5. (P5) VisionAgent subclass for Agent._encode_state / _infer.
   6. (P5) FastLaya / compiled / ONNX paths cannot take inputs_embeds; the
      image path must refuse fast=True with an explicit error.
   Read-only uses, as in upstream's fine-tune script: _fix_tokenizer_config,
   proper_reward.
6. Implementation choices made while building, for review:
   - Epoch monitoring uses up to 1,000 val images OUTSIDE the 2,000-image
     comparison set, so checkpoint choice never touches the reported ids.
   - Defaults: 4 epochs (upstream), micro-batch 16 x grad-accum 4
     (effective 64, as Gate 0), seed 20261008, save every 200 updates.
   - Visual placeholder id = pad id (its embedding is always replaced).
   - eval_vision.py without --run-dir evaluates the untrained Gate-0 base,
     text view only (zero-shot Laya on OCR) as an extra reference row.

## 2026-10-08 — P3 GPU equivalence: image path rerouted through input_ids
Observed: on the L4 the train_vision.py startup check failed with max |diff|
3.20e+01 between the image path (no visual slots) and DecisionModel.forward.
tests/test_vision_equivalence.py passed exactly on the same checkpoint on CPU.
Cause found in transformers 4.49 modeling_modernbert.py (to be confirmed
with --diag-equivalence on the L4):
- laya's build_model builds the encoder with attn_implementation="sdpa", so
  flash/unpadding is not in play.
- ModernBERT's reference_compile (None in the checkpoint config) turns on
  automatically on CUDA when triton is available, and is forced off on CPU.
- With it on, input_ids go through a torch.compile'd embedding + LayerNorm
  (compiled_embeddings), while inputs_embeds always takes the eager
  LayerNorm. The two paths ran different code on GPU and the same code on
  CPU.
- Small differences there are amplified through 28 layers under bf16. 32.0
  is exactly one bf16 step at magnitudes 4096-8192.
Fix (implementation only; the design is unchanged):
- The image path now calls the encoder with input_ids, exactly as
  DecisionModel.forward does. A forward hook on encoder.embeddings replaces
  the embedding output at the visual slots with drop(norm(projector(v))).
- LayerNorm is per token, so this equals feeding projector(v) as a pre-norm
  token embedding. The projector still targets the pre-LayerNorm space.
- Text tokens, and every row of a batch without slots, now run upstream's
  exact code: compiled embeddings, any attention implementation, any amp.
- No attention implementation or compile setting is forced. The run keeps
  laya's sdpa and ModernBERT's default reference_compile, as Gate 0 did.
- The hook also handles flash-attention's unpadded layout, for P5.
Checks:
- The startup check stays a hard stop in train and eval. It now runs the
  run's own device, attention, amp and compile settings, and also requires
  that real visual slots change the output.
- train_vision.py --diag-equivalence runs the grid attention
  (eager/sdpa/flash) x amp (none/bf16) x compile (off/on). For each it
  reports the fixed path, the old inputs_embeds path, run-to-run noise and
  logit magnitudes, then the startup check under the training
  configuration. It writes diag_equivalence.json.
Touch-point 1 (2026-10-08 P3 design, item 5) now reads: the image path
injects visual tokens with a forward hook on the encoder's embeddings module
and copies the post-encoder part of DecisionModel.forward. Touch-point 6:
FastLaya, compiled and ONNX paths bypass the hook, and still must refuse
images explicitly.

## 2026-10-10 — Gate 3: PASSED — Phase 3 closed
Gate: val accuracy well above majority class, near Route A.
Phase 4 checkpoint: run02_final (final/ trainable weights of run02).

Results (accuracy):

| Model | Set | Combined | Image-only | Text-only |
|---|---|---|---|---|
| run02_final: 60k x 3 epochs, 64 tokens, lr_proj 3e-4, modality 40/30/30 | monitor | **0.872** | 0.859 | 0.304 |
| run02 after epoch 1 / 2 / 3 | monitor | 0.815 / 0.853 / 0.872 | — | — |
| run01_base | monitor | 0.849 | 0.842 | not recorded here |
| Route A: Qwen2-VL-2B describe -> Gate-0 Laya, zero-shot | comparison set (2,000) | 0.501 | — | — |
| Frozen SigLIP 1-NN, mean of tokens, 32-token view, leave-one-out | all val (5,000) | — | 0.756 | — |
| Majority class | — | 0.0645 | 0.0645 | 0.0645 |

"Monitor" is up to 1,000 val images outside the 2,000-image comparison set
(train_vision.py epoch evaluation).

Reading:
- Combined beats majority by 13.5x. It exceeds Route A (+0.37) and the frozen
  1-NN probe (+0.12). The gate is passed on both criteria.
- Not yet like-for-like with Route A. The 0.872 is on the monitor images and
  Route A's 0.501 on the comparison ids. The Gate 2 entry fixed that the
  Route A comparison uses the trained model on the comparison ids, so
  eval_vision.py on run02_final over the comparison set is the first Phase 4
  step. The final table reports that number, not the monitor figure.
- The image carries the decision: image-only 0.859 vs combined 0.872.
  Text-only at 0.304 is weak. The head is shared across views, and OCR alone
  separates RVL-CDIP classes poorly. Phase 4 reports all three views on the
  comparison set.
- run02 beats run01_base on combined (+0.023) and image-only (+0.017).

Training note, for any future run:
- In epoch 3 of run02 the pre-clip gradient norm was 100-400. Clipping at
  1.0 held it, and the loss kept falling.
- The training worked, but steps were clipped hard for a whole epoch.
- Next time, try a lower head lr (below 1e-4) and log the pre-clip norm.

## 2026-10-10 — P4 tooling and protocol (docs/phase4_protocol.md)
Built: scripts/calibrate.py, eval_vision.py --temperatures (calibrated
metrics, confusion matrix JSON + PNG), scripts/final_eval.py, scripts/plots.py.
The protocol was written before any test run.
- Checkpoint: run02_final.
- Temperatures: one per view, fitted with upstream's
  laya.calibrate.fit_one_temperature (NLL, LBFGS, clamp [0.5, 5.0]) on val
  outside the comparison set, never on comparison ids or test.
- Metrics: raw and calibrated, ECE two ways (see below).
- Route A on test: 1,000-image stratified subsample, seed 17, ids only,
  drawn by final_eval.py and committed. Results are reported as-is.
final_eval.py is the only script that reads test labels.
- One touch: a started marker is written before any label is read, a
  re-run is refused, and --force needs --reason (recorded, old files kept).
- Every label-free check runs before the touch (checkpoint/temperature
  identity, subsample hash, Route A outputs label-blind and complete).
- --dry-run runs the pipeline on the val comparison set.
Open, needs a decision: route_a.py has no label-blind test mode. Running
Route A on the test subsample needs that mode plus a recorded exception
(test images processed, labels never read), like 2026-10-07. Without it,
the Route A comparison stays on the val comparison set.

Phase 0 ECE question (ours 0.143 vs README ~0.08) resolved: not an
implementation difference.
- scripts/eval_upstream.py (Kaggle cell 14, verbatim) and vision.metrics
  call the same laya.common.ece_score (15 equal-width bins on max p).
- On typed-decisions the harness pools choice/noul/score confidences from
  laya.Agent output (checkpoint temperatures, probabilities rounded to 4 dp).
  For our single 16-way choice only the rounding differs; both are reported.
- The README's 0.081 is "mean ECE 0.466 -> 0.081" for the base `laya`
  checkpoint after per-(type, option count) refits on held-out data: a
  different checkpoint, dataset and aggregation.
- The README's like-for-like row (laya-typed-decisions, same harness) is
  ECE 0.213. Our Gate-0 0.143 is below it.

## 2026-10-10 — P4 calibration recorded
calibrate.py on run02_final, val monitor pool (3,000 val images outside the
comparison set), one temperature per view, upstream fitter:

| View | Fitted T | ECE before | ECE after | Cross-fit ECE |
|---|---|---|---|---|
| combined | 1.108 | 0.025 | 0.022 | 0.0227 |
| image | 1.068 | 0.023 | 0.011 | 0.0128 |
| text | 1.039 | 0.031 | 0.039 | 0.035 |

Reading:
- run02_final is already well calibrated raw: every view is under the 0.10
  criterion before fitting.
- The fitted temperatures are slightly above 1, so the model is mildly
  over-confident.
- For the text view, fitting raised ECE both in-sample and out-of-sample.
  The text view is weak (0.304 accuracy at Gate 3), and one scalar does not
  improve its reliability.

Decision:
- Combined and image are reported with their fitted temperatures.
- Text is reported at T = 1.0.
- Implemented as vision.UNCALIBRATED_VIEWS = ("text",). eval_vision.py and
  final_eval.py apply it and record the fitted and applied temperatures.
- docs/phase4_protocol.md §2 is updated before any test run.

Phase 0 ECE question closed: the measurement was correct.
- The two ECE implementations agree on the combined view: 0.0250 (ours)
  vs 0.0251 (upstream harness).
- The 0.143 vs ~0.08 gap was a comparison of different quantities
  (DECISIONS.md 2026-10-10, "P4 tooling and protocol").

## 2026-10-10 — P4 error analysis (val comparison set, combined view)
Source: eval_vision.py on run02_final, 2,000-image val comparison set,
combined view (eval_val_final.json). Per-class accuracy:
- Weakest: scientific report 0.672, form 0.724, letter 0.779,
  questionnaire 0.786, presentation 0.815.
- Strongest: email 0.968, resume 0.926, advertisement 0.921,
  handwritten 0.918, specification 0.897.

Finding: the errors are where earlier phases predicted, not new failure
modes.
- **Frozen embeddings:** the Gate 2 NN check's weakest classes were
  scientific report 0.578, form 0.593, presentation 0.595, memo 0.631 and
  letter 0.632. Four of those five are among the five weakest here. Training
  lifts every class but keeps the order.
- **Route A's confusion neighbourhoods:** the weak classes sit in them
  (report/publication, form/questionnaire, letter/memo/email). Within a
  neighbourhood, the strongest member (email 0.968) shows these are close
  pairs, not unlearnable classes.
- **Label noise:** form overlaps the P1 hand-audit's label-noise cluster
  (structured business documents; 2 of the 4 mislabels were "form"). Part
  of form's gap is likely label noise, not model error. That share is not
  measured.
- **Confidence pattern, to check:**
  - The weakest classes are the ones where ambiguity is in the documents,
    not just the model.
  - So low confidence on them is the expected behaviour.
  - Per-class reliability is not yet measured; it can be computed from
    eval_val_final.predictions.jsonl.
- **Neighbourhood accuracy** (the neighbourhood_accuracy block of
  eval_val_final.json): not recorded yet. The file is on Drive and its
  values were not in the request. To be added.
Test is not involved: this is the val comparison set only.

## 2026-10-10 — Recorded exception: Route A reads test labels on the pre-registered test subsample
Decision (human): route_a.py may read TEST labels, for scoring only, on the
pre-registered 1,000-image test subsample. Route A is the pre-registered
external baseline and runs once on that subsample.
- Subsample: splits/test_comparison_1000_seed17.json, drawn by
  route_a.py --make-comparison-set --records test.jsonl (stratified by
  class, seed 17).
  - Same format and sha256 guard as the val comparison set, plus
    "split": "test".
  - Committed before Route A runs; never regenerated.
- route_a.py accepts --records test.jsonl ONLY when --comparison-set is
  that file, and refuses the test subsample with val records.
- Running on test prints a notice.
- A complete test result in --out-dir refuses further runs. An
  interrupted run resumes under the same manifest; --results-only
  recomputes.
- Nothing is trained or tuned on these labels. Route A is zero-shot, with
  the same prompt (a2), VLM revision and Gate-0 Laya checkpoint as on val.
Consequences:
- CLAUDE.md test rule amended to name this exception alongside
  final_eval.py.
- docs/phase4_protocol.md §4–§5 updated before any test run.
- Supersedes the "label-blind Route A" plan in the P4 tooling entry above.
  final_eval.py no longer draws its own subsample (--make-route-a-subsample
  removed) and no longer requires label-blind Route A outputs. Instead, before
  the touch it checks that the Route A run's manifest names this subsample
  (split, seed, n, test.jsonl sha256). After loading labels it checks that
  the outputs' labels agree with test.jsonl. It scores Route A again with
  the same metric code as Laya-Vision.

## 2026-10-10 — Route A test baseline: 0.510 on the pre-registered test subsample
Route A ran once on splits/test_comparison_1000_seed17.json (1,000 test
images, stratified, seed 17), under the recorded exception above. Setup was
unchanged from val: Qwen2-VL-2B describe with prompt a2, the same VLM
revision, and the Gate-0 fine-tuned Laya, zero-shot.

| Route A | Set | Accuracy | Majority floor | VLM p50 | Laya p50 | GPU-s/image | Errors |
|---|---|---|---|---|---|---|---|
| val (2026-10-06) | val comparison set, 2,000, seed 11 | 0.501 | 0.0645 | 7.5 s | 41 ms | 7.6 | 0 |
| **test** | test subsample, 1,000, seed 17 | **0.510** | 0.064 | 7.7 s | 42 ms | 7.71 | 0 |

- Test agrees with val within sampling noise: +0.009, against a standard
  error of about 0.016 at n = 1,000. Latency and cost also match val. The
  val baseline was not a lucky or unlucky draw.
- This is the Route A number for the test-table comparison (protocol §4
  step 5). It is fixed and is not re-run.
- Next and last: final_eval.py on test, once, with --route-a-outputs
  pointing at this run's route_a_outputs.jsonl. It scores Route A again
  with the same metric code and compares on the same 1,000 ids.

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

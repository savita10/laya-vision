# Phase 4 evaluation protocol

Written 2026-10-10, before any test run. `final_eval.py` records this file's
sha256 in `final_results.json`. If this file changes after the test run, the
change shows. Code: `scripts/calibrate.py`, `scripts/eval_vision.py`,
`scripts/final_eval.py`, `scripts/plots.py`.

## 1. What is evaluated

- **Checkpoint:** `run02_final`, the `final/trainable.safetensors` of run02.
  run02 was 60k train × 3 epochs, 64 visual tokens, lr_proj 3e-4,
  modality 40/30/30 (Gate 3, DECISIONS.md 2026-10-10). It sits on the
  Gate-0 fine-tuned base (`step5/laya_finetuned`).
  - `calibration.json` records the checkpoint's sha256.
  - `eval_vision.py` and `final_eval.py` refuse temperatures fitted on any
    other checkpoint or token count.
  - No other checkpoint is evaluated on test.
- **Views:** three, with no modality dropout at evaluation.

  | View | Input | Role |
  |---|---|---|
  | combined | [visual \| OCR text] | headline |
  | image | [visual] | |
  | text | [OCR text] | exactly upstream Laya's sequence for that text |

## 2. Temperatures (calibration)

- **One scalar per view,** fitted by `calibrate.py` on the **val monitor pool**:
  - The pool is every val image outside the fixed 2,000-image comparison
    set (about 3,000 images).
  - It is the pool `train_vision.py` monitored on. The comparison ids and
    test never take part in fitting.
- **Fitter:** upstream's `laya.calibrate.fit_one_temperature`. It minimises NLL
  of `softmax(z / T)` with LBFGS on log T, clamped to [0.5, 5.0] as laya
  clamps at load. A temperature at the bound is flagged.
- **Reported in `calibration.json`:**
  - temperatures;
  - NLL, Brier and ECE before and after fitting (in-sample, so optimistic);
  - a 2-fold cross-fit estimate;
  - reliability bins;
  - PNG diagrams (15 bins, before vs after).
- **Frozen once written.** Temperatures are fixed before the test run and are
  not refitted after it.

## 3. Metrics

Every metric below is reported per view, raw (T = 1) and calibrated, except
where noted.

| Metric | Definition |
|---|---|
| accuracy | argmax over the 16 options. A scalar T does not change it, so raw = calibrated. |
| NLL | −log p(true class) |
| Brier | sum over the 16 classes of (p − onehot)² |
| **ECE (headline)** | `laya.common.ece_score(max p, correct)`, 15 equal-width bins; first bin [0, 1/15], the others (lo, hi] |
| ECE, upstream harness | as `scripts/eval_upstream.py` computes it from `laya.Agent` output: probabilities rounded to 4 decimals and renormalised, then the same `ece_score`. Reported alongside, not instead. See §6. |
| mean confidence | mean of max p |
| per-class accuracy | 16 classes |
| neighbourhood accuracy | letter/email/memo, form/questionnaire, scientific report/publication |
| confusion matrix | 16×16, rows = true class; JSON and PNG |
| reliability bins | JSON and PNG |
| cost (raw only) | GPU-seconds/image batched (batch 64); batched latency p50/p95 per batch and per image; batch-1 latency p50/p95 |

- **Cost scope:** cost covers the Laya-Vision forward only. Precompute (SigLIP
  embedding + Tesseract OCR) is reported separately from the Phase 2
  measurements. Every cost comparison says which of the two it includes.
- **Unscorable images:** images whose file failed to decode have no shard
  row and cannot be scored. They are counted. Accuracy is reported over
  scored images and also with unscorable images counted wrong.

## 4. Order of runs

1. **Fit temperatures:** `calibrate.py` on run02_final → `calibration.json`.
   Record the temperatures and the before/after numbers in DECISIONS.md.
2. **Val comparison set:** `eval_vision.py --temperatures calibration.json` on
   the 2,000-image comparison set. This is the like-for-like comparison with
   Route A (0.501 on the same ids, zero-shot). It replaces the
   monitor-set figure used at Gate 3, and adds the confusion matrix.
3. **Dry run:** `final_eval.py --dry-run` (val comparison set; test not read)
   checks the full pipeline end to end.
4. **Route A test subsample:** `final_eval.py --make-route-a-subsample` draws
   **1,000 test images, stratified by class, seed 17**.
   - It writes `splits/test_route_a_1000_seed17.json` with ids only, the
     test.jsonl sha256 and no per-class counts. The file is committed
     before Route A runs.
   - Drawing it reads test labels to stratify, which is why
     `final_eval.py` does it.
5. **Route A on the test subsample:** `route_a.py`, label-blind. Its outputs
   must carry no label fields; `final_eval.py` refuses outputs that do.
   - **Not built yet.** `route_a.py` is val-only today. A test mode needs a
     DECISIONS.md exception, like the 2026-10-07 precompute exception (test
     images processed, labels never read).
   - If it is not done, the test table reports Laya-Vision alone, and the
     Route A comparison stays on val (step 2), stated as such.
6. **The test run:** `final_eval.py` on test, **once**.
   - It scores the whole test split, three views, raw and calibrated.
   - It scores the same metrics on the Route A subsample, together with
     Route A's outputs.
   - It writes `final_results.json`, `final_predictions.jsonl`, and
     confusion and reliability PNGs.

## 5. Rules

- **Only `final_eval.py` reads test labels** (CLAUDE.md). Nothing else in
  this protocol does.
- **One touch.**
  - `final_eval.py` writes a started marker before reading any label, so a
    crash still counts as the touch.
  - It refuses to run if a result or marker exists.
  - A re-run needs `--force --reason "<why>"`. The reason and the sha256
    of the superseded files go into the new output, and the old files are
    kept (renamed).
  - Any forced re-run is recorded in DECISIONS.md, and both results are
    reported.
- **Nothing changes after the test run** on the strength of test numbers:
  checkpoint, temperatures, views, metrics and this protocol stay as they
  are.
- **Results are reported as-is,** including any negative result against the
  success criteria (CLAUDE.md):
  - combined decisively beats majority class;
  - it approaches or beats Route A at a fraction of its cost;
  - calibrated ECE < 0.10, judged on the headline ECE on test.
- **Headline numbers:**
  - Test: combined-view accuracy and calibrated ECE.
  - Route A comparison: combined vs Route A on the 1,000-image test
    subsample, or on the val comparison set if step 5 is not done.

## 6. The two ECE implementations (Phase 0 open question)

Gate 0 left a question: our fine-tuned typed-decisions ECE was 0.143 against
a README reference of ~0.08.

**Same function.** Both our harnesses call the same `laya.common.ece_score`
(15 equal-width bins on max p). There is no binning or implementation
difference:
- `scripts/eval_upstream.py` is the Kaggle notebook's cell 14, copied
  as-is.
- `vision.metrics` is ours.

**What differs between the two harnesses:**
1. **Inputs, on typed-decisions.** The upstream harness pools choice, noul
   and score questions. Confidence is max p for choice, max(p, 1 − p) for
   noul, and max of the level distribution for score. Probabilities come
   from `laya.Agent` with the checkpoint's own temperatures, rounded to 4
   decimals.
2. **For a single 16-way choice** (this project), only the 4-decimal
   rounding remains. Both are reported so the size of that effect is
   visible; it is expected to be negligible.

**Where the "~0.08" comes from.** The upstream README's "0.081" is not the
typed-decisions harness number. It is "mean ECE 0.466 → 0.081" for the
*base* `laya` checkpoint after refitting one temperature per (question type,
option count) on held-out data. That is a different checkpoint, different
data and a different aggregation.

**Like-for-like.** The README's own typed-decisions table gives the
fine-tuned checkpoint **ECE 0.213**, measured with the same harness we
ported. Our 0.143 is below that.

**Conclusion.** The gap was a comparison between two different measurements,
not an ECE implementation error.

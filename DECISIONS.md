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

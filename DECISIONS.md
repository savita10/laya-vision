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

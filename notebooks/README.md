# Notebooks

The notebooks are launchers only: they clone the repo, install pinned packages and run
scripts from `scripts/`. The real code lives in `.py` files (docs/PROJECT_SPEC.md,
engineering rule 2). Phases 1–4 ran their scripts directly; their commands are in
`DECISIONS.md` and `experiments.md`.

| File | What it is | Runtime |
|---|---|---|
| `phase0_smoke_test.ipynb` | Phase 0 smoke test and interface check (ours) | Colab, CPU is enough |
| `step5_colab_launcher.ipynb` | Phase 0 reproduction run, Gate 0 (ours) | Colab, T4 / L4 / A100 GPU |
| `laya_finetune_typed_decisions_2xT4_kaggle.ipynb` | Upstream Laya fine-tuning notebook, unmodified | Kaggle, 2x T4 |
| `laya_finetune_typed_decisions_mps.py` | Upstream single-process version of that notebook, unmodified | Apple Silicon / CPU (CUDA through our wrapper) |

## phase0_smoke_test.ipynb
- **Checks:** the pinned `laya==0.3.24` installs and answers all three typed primitives
  (choice, score, noul) in one `Router.predict` call on a sample email.
- **Interface discovery:** prints the question validator (`Agent._check_question`), the
  `Router.predict` docstring and every score/criteria rule. The published docs lagged the
  installed package, so the validator was the ground truth.
- **Cheat-sheet:** ends with the question format the rest of the project uses.
- **Setup:** Cell A mounts Drive and clones the public repo; it needs no token and no GPU.

## step5_colab_launcher.ipynb
The Phase 0 reproduction (DECISIONS.md 2026-10-02):
1. Mounts Drive and clones the repo.
2. Fine-tunes the Laya base on `LocalLLaMA/typed-decisions` with upstream's script, run
   through `scripts/run_upstream_finetune.py` (adds CUDA only). Effective batch 64, 4 epochs.
3. Evaluates the base and fine-tuned checkpoints with `scripts/eval_upstream.py` against
   the published numbers.

The fine-tuned checkpoint (`step5/laya_finetuned` on Drive) is the Gate-0 base that
Laya-Vision is built on and ships in `weights/base/`. Written while the repo was private,
it reads `GH_TOKEN` and `HF_TOKEN` from Colab secrets.

## Upstream files
`laya_finetune_typed_decisions_2xT4_kaggle.ipynb` and `laya_finetune_typed_decisions_mps.py`
are Convai Innovations' fine-tuning recipe for Laya (Apache-2.0; see `NOTICE`), kept
byte-for-byte as published:
- the reproduction runs the `.py` unmodified, through the CUDA wrapper;
- `scripts/eval_upstream.py` ports the evaluation cells of the Kaggle notebook.

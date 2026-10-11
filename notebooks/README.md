# Notebooks

The notebooks are launchers only: they clone the public repo, install pinned packages
(`transformers==4.49.0`, `laya==0.3.24`) and run scripts from `scripts/`. The real code
lives in `.py` files (docs/PROJECT_SPEC.md, engineering rule 2).
- **Cell A:** every launcher starts with the same setup cell. It mounts Drive, clones
  https://github.com/savita10/laya-vision with no token, installs, and reports the GPU.
- **Paths:** the Drive layout (`MyDrive/laya-vision/{step5,phase1,phase2,phase3}`) is the
  one recorded in `DECISIONS.md`; adjust it for your own account.

## Files and run order

| Order | File | What it runs | GPU |
|---|---|---|---|
| 1 | `phase0_smoke_test.ipynb` | Phase 0: Laya smoke test and interface check | No |
| 2 | `step5_colab_launcher.ipynb` | Phase 0: reproduce upstream's fine-tune, then Gate 0 eval. Produces the Gate-0 base `step5/laya_finetuned` | Yes (T4 / L4 / A100) |
| 3 | `phase1_colab_launcher.ipynb` | Phase 1: dataset splits, 200-image audit sample, secondary-label experiments (deferred), Route A baseline on val | P1.1–P1.2 no; P1.4–P1.5 yes (L4) |
| 4 | `phase2_colab_launcher.ipynb` | Phase 2: Tesseract OCR + SigLIP precompute into shards; nearest-neighbour check | Yes (L4 / T4) + Tesseract |
| 5 | `phase3_colab_launcher.ipynb` | Phase 3: pre-flight checks, tiny / medium / full runs, ablations, `run02_final` | Yes (L4) |
| 6 | `phase4_colab_launcher.ipynb` | Phase 4: calibration, val evaluation, Route A on the test subsample, the one-touch `final_eval.py`, release staging | Yes (L4); the CPU-latency cell runs on CPU |
| — | `laya_finetune_typed_decisions_2xT4_kaggle.ipynb` | Upstream Laya fine-tuning notebook, unmodified; reference only | Kaggle, 2x T4 |
| — | `laya_finetune_typed_decisions_mps.py` | Upstream single-process version of that notebook, unmodified; run by step 2 through `scripts/run_upstream_finetune.py` | Through step 2 |

- **Dependencies:** each phase reads the previous phase's outputs on Drive:
  - Phase 1 needs the Gate-0 base for Route A.
  - Phase 2 needs the Phase 1 splits.
  - Phase 3 needs the Phase 2 shards.
  - Phase 4 needs `run02_final`.
- **Phase 5:** the release is built on a laptop from the files Phase 4 stages
  (`scripts/build_release.py`; see the main README).
- **Test labels:** Phase 4 is the only launcher that reads them. It does so in
  `final_eval.py` (runs once) and in `route_a.py` on the pre-registered 1,000-image
  subsample, per the recorded exceptions.

## phase0_smoke_test.ipynb
- **Checks:** the pinned `laya==0.3.24` installs and answers all three typed primitives
  (choice, score, noul) in one `Router.predict` call on a sample email.
- **Interface discovery:** prints the question validator (`Agent._check_question`), the
  `Router.predict` docstring and every score/criteria rule. The validator is located with
  `inspect`, so this works on any laya version. The published docs lagged the installed
  package, so the validator was the ground truth. The findings are recorded in DECISIONS.md,
  "Backfill: Phase 0 Step 4 findings (2026-10-01, recorded late)".
- **Cheat-sheet:** ends with the question format the rest of the project uses.
- **Setup:** Cell A mounts Drive and clones the public repo; it needs no token and no GPU.

## step5_colab_launcher.ipynb
The Phase 0 reproduction (DECISIONS.md 2026-10-02):
1. Cell A, the public-repo form shared with the Phase 0 notebook:
   - mounts Drive and creates `step5/`;
   - clones the public repo over https (no token) and prints the commit to log in
     `experiments.md`;
   - installs pinned `transformers==4.49.0` and `laya==0.3.24`;
   - sets `HF_TOKEN` from Colab secrets if present;
   - reports the GPU.
2. Fine-tunes the Laya base on `LocalLLaMA/typed-decisions` with upstream's script, run
   through `scripts/run_upstream_finetune.py` (adds CUDA only). Effective batch 64, 4 epochs.
3. Evaluates the base and fine-tuned checkpoints with `scripts/eval_upstream.py` against
   the published numbers.

The fine-tuned checkpoint (`step5/laya_finetuned` on Drive) is the Gate-0 base that
Laya-Vision is built on and ships in `weights/base/`. The training cell still requires the
`HF_TOKEN` Colab secret. Cell A clones without a token, so it works once the repo is
public.

## Upstream files
`laya_finetune_typed_decisions_2xT4_kaggle.ipynb` and `laya_finetune_typed_decisions_mps.py`
are Convai Innovations' fine-tuning recipe for Laya (Apache-2.0; see `NOTICE`), kept
byte-for-byte as published:
- the reproduction runs the `.py` unmodified, through the CUDA wrapper;
- `scripts/eval_upstream.py` ports the evaluation cells of the Kaggle notebook.

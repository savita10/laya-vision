# Phase 3 design — Laya-Vision training

Approved 2026-10-08 (DECISIONS.md, "P3 design approved"). Code:
`scripts/vision.py` (shared core), `scripts/train_vision.py`,
`scripts/eval_vision.py`, `scripts/ocr_token_budget.py`,
`tests/test_vision_equivalence.py`. Pinned: laya 0.3.24, transformers 4.49.0.

## 1. Backbone

- `convaiinnovations/laya`: ModernBERT-large, d=1024, 28 layers, initialised
  from the **Gate-0 fine-tuned checkpoint** (Drive `step5/laya_finetuned`).
  This is the backbone Gate 0 reproduced and Route A used, so the Route A
  comparison shares a backbone.
- Run at **max_len 512**. The checkpoint config says 1024 because Gate 0
  was fine-tuned at that length; 512 is within the trained range.
  head_max_len stays 256, as in the checkpoint.
- Loaded through laya's own `build_model(cfg, encoder_dir=…)`, which applies
  `_apply_rope_config`. The tokenizer is loaded the way upstream's fine-tune
  script does it (`_fix_tokenizer_config`, then `AutoTokenizer`). Every base
  weight must load; only `projector.*` may be missing.

## 2. Sequence and token budget

Upstream `laya.common.build_sequence` is used unchanged:

```
[CLS] choice question: What type of document is this? [SEP]
[MASK] letter [MASK] form … [MASK] memo [SEP]
<state> [SEP]
```

The state is built by one function (`SequenceBuilder.build`), in the fixed
order **[visual | OCR text | features | user text]**:

- **visual:** 32 placeholder ids followed by one `[SEP]`. Their token
  embeddings are replaced by projected SigLIP tokens. There is no segment
  embedding.
- **text:** OCR, then derived features, then user text. Features and user
  text are empty in v1. The text is cut from the end when the budget runs
  out.

The option markers live inside the question head, before the state, so the
visual block never moves them. Positions are RoPE over `arange(seq_len)`,
and visual tokens take positions like any other token.

| Part | Tokens (doc_type, Gate-0 tokenizer) |
|---|---|
| Question head (cap head_max_len 256) | 50 |
| Visual block (32 + [SEP]), never truncated | 33 |
| Closing [SEP] | 1 |
| **Text budget, combined** | **428** |
| **Text budget, text-only** | **461** |
| Image-only sequence length | 84 |

- **Text-only:** the sequence is token-for-token upstream's for the same
  text (test 4).
- **Visual block:** if the visual block does not fit, the builder raises. It
  never silently cuts it.
- **Truncation share:** `ocr_token_budget.py` measures how much train OCR
  exceeds the budget. It flags if more than 25% of pages lose at least 50%
  of their text.

## 3. Model

`VisionDecisionModel(laya.common.DecisionModel)`, with `projector` =
Linear(768, 1024), GELU, Linear(1024, 1024).

- **No image:** `forward(…)` without `visual` is `DecisionModel.forward`,
  literally `super().forward`. This is reading (A) of the image=None
  contract: same code path, our weights.
- **With an image:**
  1. `e = encoder.embeddings.tok_embeddings(input_ids)`.
  2. Write `projector(visual)` into the visual slots, out of place.
  3. Run `encoder(inputs_embeds=e, attention_mask)`.
  4. Run the post-encoder head code copied from laya 0.3.24.

  ModernBERT applies its embedding LayerNorm to `inputs_embeds` as well, so
  the projector targets the raw, pre-norm embedding space. A mixed batch
  (some rows text-only) takes this path; rows with no slots are then
  numerically identical to upstream's forward.
- **Required equivalence test** (`tests/test_vision_equivalence.py`, and a
  startup check in both train and eval):
  - the no-image path equals `DecisionModel.forward` exactly;
  - the `inputs_embeds` path with no slots matches it;
  - gradients reach only the projector and head;
  - text-only sequences equal upstream `build_sequence`;
  - visual block and markers survive text overflow.

### Trainable parameters (d=1024)

| Module | Status | Parameters |
|---|---|---|
| SigLIP | offline (Phase 2) | 0 |
| ModernBERT-large encoder | frozen | 0 |
| act head | frozen (upstream loss weight 0) | (263,938) |
| head: 2 × TransformerEncoderLayer (16 heads, FFN 4096) | trained | 25,192,448 |
| scorer | trained | 1,052,673 |
| type_emb | trained | 3,072 |
| projector 768 → 1024 → 1024 | trained | 1,837,056 |
| **Total trainable** | | **28,085,249** |

Gradients must flow back through all 28 frozen encoder layers to reach the
projector. So the backward pass costs nearly as much as full fine-tuning,
minus the weight gradients.

## 4. Training (`train_vision.py`)

- **Loss:** upstream's, unchanged in form. Cross-entropy plus RLCD:
  - 4 noisy logit samples, sigma 0.4 → 0.1 linearly over the epochs;
  - `proper_reward(w_sph=0.75, w_rps=1.0)` (RPS is inactive for choice);
  - normalised advantage;
  - act-head term weighted 0.
  - Target: a hard one-hot over 16 options in canonical order. No label
    smoothing.
- **Optimizer:** AdamW, weight decay 0.01, grad clip 1.0.
  - Head group lr 1e-4 (upstream).
  - Projector group lr 1e-3 (new).
  - Upstream's encoder group is removed (encoder frozen).
- **Schedule:** linear warmup over 3% of updates, then upstream's cosine to
  eta_min 1e-6.
- **Precision:** bf16 autocast, plus gradient checkpointing on the encoder
  (non-reentrant) and the head. This diverges from upstream's fp32.
- **Defaults:** 4 epochs, micro-batch 16 × accumulation 4 (effective 64, as
  in Gate 0), 32 visual tokens, seed 20261008.
- **Modality dropout:** drawn per example from
  `sha256(seed:epoch:id)` → 40% combined / 30% image-only / 30% text-only.
  Deterministic, and survives a resume.
- **Epoch order:** `Random(seed + epoch)` shuffle, as upstream does. Then
  length grouping within pools of 50 batches, then the batch order is
  shuffled.
- **No temperature fitting:** that is Phase 4.

## 5. Data (`vision.load_split_data`)

- **Visual:** `shards.load_split(tokens=32, dtype=float16)` derives the
  8x4 view from the stored 8x8 exactly and keeps it fp16: 60k × 32 × 768 is
  2.9 GB.
- **OCR:** `shards.load_ocr`. It is tokenised once with upstream's state
  tokenisation (mask token blanked, no specials), capped at 512, stored as
  int32, and cached on disk. The cache key covers the ids hash, tokenizer
  file hashes and the OCR pass spec.
- **Labels:** from `<split>.jsonl`. Split loading accepts only train and
  val; `shards.py` refuses test.
- **`--limit N`:** the first N train examples (dry runs and the tiny run).

## 6. Evaluation (`eval_vision.py`)

- **Data:** the fixed 2,000-image comparison set. Its val.jsonl sha256 is
  checked as in route_a.py.
- **Views:** combined, image-only and text-only, with no dropout.
- **Per view:**
  - accuracy, NLL, Brier, 15-bin ECE (`laya.common.ece_score`), mean
    confidence;
  - per-class accuracy and per-neighbourhood accuracy (letter/email/memo,
    form/questionnaire, scientific report/publication);
  - text truncation count;
  - forward cost: GPU-s/image batched, and batch-1 latency p50/p95.
- **Outputs:** per-image predictions written as JSONL. Raw probabilities
  (no temperature).
- **Untrained baseline:** `--run-dir` omitted gives the Gate-0 base,
  text view only. That is zero-shot Laya on OCR text.
- **What is reported against Route A:** Route A's numbers come from
  `docs/route_a_results.json`. Laya-Vision cost excludes precompute
  (SigLIP + Tesseract, Phase 2), and the report says so.

## 7. Monitoring, checkpoints, logging

- **Monitor set:** epoch evaluation uses up to 1,000 val images **outside**
  the comparison set, so checkpoint choice never touches the reported ids.
- **`latest/`:** replaced atomically every 200 updates. It holds trainable
  weights, optimizer, scheduler, RNG states and the position (epoch,
  micro-step). `epoch_N/` and `final/` hold trainable weights plus monitor
  metrics.
- **Resume:**
  - Restarting with the same arguments resumes. A changed configuration is
    refused (`run_config.json` identity).
  - Checked on CPU: two mid-epoch kills plus resumes gave final weights
    bit-identical to an uninterrupted run.
- **Logging:** `train_log.jsonl` records loss, CE/RL split, train accuracy,
  grad norm, both learning rates, sigma, examples/s, tokens/example,
  modality counts, text truncations and peak GPU memory. Every run also
  gets a line in `experiments.md`; the script prints a draft line.

## 8. Laya touch-points (no laya file is edited)

1. `VisionDecisionModel` subclasses `DecisionModel`. The image path copies
   the post-encoder part of `DecisionModel.forward` (0.3.24), guarded by the
   equivalence test.
2. `laya.common.build_sequence(state_ids=…)` is used as-is to reserve the
   visual slots. `build_head` and `encode_text` are used to measure and
   tokenise.
3. Our own collate replaces `collate_items`, adding `visual` and
   `visual_pos`.
4. The checkpoint format adds `projector.*`. Our loader overlays trainable
   weights on the Gate-0 base; `laya.load` would reject the extra keys.
5. (P5) `Agent._encode_state` and `_infer` are private and call the model
   positionally, so they need a `VisionAgent` subclass.
6. (P5) FastLaya, compiled and ONNX paths do not take `inputs_embeds`. The
   image path must refuse `fast=True` with an explicit error.

Also used read-only, as upstream's own fine-tune script does:
`laya.agent._fix_tokenizer_config` and `laya.common.proper_reward`.

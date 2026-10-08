"""Laya-Vision core: model, sequence building, data, loss and metrics (Phase 3).

Shared by train_vision.py, eval_vision.py, ocr_token_budget.py and the
equivalence test, so training and evaluation build every input the same way.
Design: docs/phase3_design.md; decisions: DECISIONS.md 2026-10-08.

Backbone: the Gate-0 fine-tuned convaiinnovations/laya checkpoint
(ModernBERT-large, d=1024), run at max_len 512. Frozen: the encoder and the
act head. Trained: the projector (768 -> 1024 -> 1024 MLP) and the decision
head (head layers, type embedding, scorer).

Sequence (upstream laya.common.build_sequence, unchanged):
    [CLS] choice question: <ins> [SEP] [MASK] opt0 ... [MASK] opt15 [SEP] <state> [SEP]
The state is built in the fixed order [visual | OCR text | features | user text]:
    visual  32 placeholder ids followed by one [SEP]; at the placeholders the
            embedding output is replaced by norm(projected SigLIP token), via a
            forward hook on encoder.embeddings (the encoder still gets input_ids)
    text    OCR, then derived features, then user text (features and user text
            are empty in v1); truncated from the end when the budget runs out
The option markers sit inside the question head, before the state, so the
visual block never moves them. Visual tokens are never truncated: a budget
too small for them raises. A text-only example has no visual block, so its
sequence is exactly upstream's for the same text.

Laya touch-points (DECISIONS.md 2026-10-08): VisionDecisionModel subclasses
laya.common.DecisionModel; its no-image path IS DecisionModel.forward, and its
image path calls the encoder with input_ids, injects visual tokens with a
forward hook on encoder.embeddings, and duplicates the post-encoder part of
DecisionModel.forward as of laya 0.3.24 (guarded by
tests/test_vision_equivalence.py and a startup check).
"""

import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import rvlcdip

REPO = Path(__file__).resolve().parent.parent
LAYA_VERSION = "0.3.24"
QUESTION_ID = "doc_type"
MAX_LEN = 512
N_VISUAL = 32
VISUAL_DIM = 768
MODALITIES = ("combined", "image", "text")
DEFAULT_MODALITY_PROBS = (0.4, 0.3, 0.3)
TRAINABLE_PREFIXES = ("projector.", "head.", "type_emb.", "scorer.")
# RVL-CDIP confusion neighbourhoods seen in Route A (DECISIONS.md 2026-10-06).
NEIGHBOURHOODS = {
    "letter/email/memo": ["letter", "email", "memo"],
    "form/questionnaire": ["form", "questionnaire"],
    "scientific report/publication": ["scientific report", "scientific publication"],
}


def check_laya_version():
    from importlib.metadata import version

    installed = version("laya")
    if installed != LAYA_VERSION:
        raise SystemExit(f"laya {installed} is installed; VisionDecisionModel duplicates DecisionModel.forward "
                         f"from laya {LAYA_VERSION} (pip install laya=={LAYA_VERSION}).")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_git_hash():
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# --- model ------------------------------------------------------------------------

class Projector(nn.Module):
    """2-layer MLP: pooled SigLIP token (768) -> Laya token-embedding space (d_model).

    Its output replaces raw (pre-LayerNorm) token embeddings: ModernBERT applies
    its embedding LayerNorm to inputs_embeds as well.
    """

    def __init__(self, visual_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(visual_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x):
        return self.net(x)


def _inject_visual(module, args, output):
    """Forward hook on encoder.embeddings: put projected visual tokens into the visual slots.

    Set per call by VisionDecisionModel.forward through `module._laya_visual` =
    (rows, positions, projected tokens [n, d], attention_mask), else None (no-op). The tokens
    get the same LayerNorm + dropout as text embeddings. Handles ModernBERT's padded output
    [B, L, d] and its flash-attention unpadded output [nnz, d].
    """
    inject = module.__dict__.get("_laya_visual")
    if inject is None:
        return None
    rows, pos, tokens, attention_mask = inject
    v = module.drop(module.norm(tokens.to(module.tok_embeddings.weight.dtype))).to(output.dtype)
    if output.dim() == 3:
        return output.index_put((rows, pos), v)
    # Unpadded: row-major position among attended tokens (visual slots are always attended).
    flat = rows * attention_mask.size(1) + pos
    idx = attention_mask.flatten().long().cumsum(0)[flat] - 1
    return output.index_put((idx,), v)


def _decision_model_class():
    from laya.common import DecisionModel

    return DecisionModel


def make_vision_model_class():
    DecisionModel = _decision_model_class()

    class VisionDecisionModel(DecisionModel):
        """DecisionModel + projector; projected visual tokens enter through a hook on encoder.embeddings."""

        def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1, no_init=False, visual_dim=VISUAL_DIM):
            super().__init__(encoder, head_layers, n_act, dropout, no_init=no_init)
            # Always initialised (never on meta): the base checkpoint has no projector.
            self.projector = Projector(visual_dim, encoder.config.hidden_size)
            encoder.embeddings.register_forward_hook(_inject_visual)

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False,
                    visual=None, visual_pos=None):
            if visual is None:
                # No image: exactly upstream's forward (reading A, DECISIONS.md 2026-10-08).
                return super().forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)
            # The encoder is always called with input_ids, so every text token takes upstream's
            # path (compiled embeddings on GPU, any attention implementation). The hook then
            # swaps the embedding output at the visual slots for norm(projector(v)); LayerNorm is
            # per token, so this equals feeding projector(v) as a pre-norm token embedding.
            emb = self.encoder.embeddings
            slots = visual_pos >= 0
            inject = None
            if slots.any():
                rows = torch.arange(input_ids.size(0), device=input_ids.device)[:, None].expand_as(visual_pos)
                inject = (rows[slots], visual_pos[slots], self.projector(visual)[slots], attention_mask)
            object.__setattr__(emb, "_laya_visual", inject)
            try:
                h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            finally:
                object.__setattr__(emb, "_laya_visual", None)
            return self._decide(h, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)

        def forward_inputs_embeds(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                                  visual, visual_pos, detach_encoder=False):
            """The pre-fix image path (encoder(inputs_embeds=...)), kept for --diag-equivalence only.

            On GPU it differs from upstream's input_ids path: ModernBERT compiles its input_ids
            embedding (reference_compile) but runs inputs_embeds eagerly (DECISIONS.md 2026-10-08,
            GPU equivalence)."""
            e = self.encoder.embeddings.tok_embeddings(input_ids)
            slots = visual_pos >= 0
            if slots.any():
                proj = self.projector(visual).to(e.dtype)
                rows = torch.arange(input_ids.size(0), device=input_ids.device)[:, None].expand_as(visual_pos)
                e = e.index_put((rows[slots], visual_pos[slots]), proj[slots])
            h = self.encoder(inputs_embeds=e, attention_mask=attention_mask).last_hidden_state
            return self._decide(h, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)

        def _decide(self, h, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
            # Copied from laya 0.3.24 DecisionModel.forward, after the encoder call. Keep in step
            # with the pinned version; tests/test_vision_equivalence.py checks they agree.
            from torch.utils.checkpoint import checkpoint

            if detach_encoder:
                h = h.detach()
            h = h + self.type_emb(qtype)[:, None, :]
            if self.head is not None:
                pad = ~attention_mask.bool()
                for layer in self.head.layers:
                    if self.head_checkpointing and self.training and torch.is_grad_enabled():
                        h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                    else:
                        h = layer(h, src_key_padding_mask=pad)
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx)
            logits = self.scorer(m).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)

            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            if p.size(-1) >= 2:
                top2 = p.topk(2, -1).values
            else:
                top1 = p.topk(1, -1).values
                top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
            pooled = h[:, 0].float()
            act_logits = self.act_head(torch.cat([pooled, feats], -1))
            return logits, act_logits

    return VisionDecisionModel


def load_base(base_dir, max_len=MAX_LEN, projector_seed=0):
    """Gate-0 checkpoint -> (VisionDecisionModel fp32 on CPU, tokenizer, cfg, base sha256).

    Every base weight must load; the only missing keys allowed are the projector's.
    """
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    base_dir = Path(base_dir)
    with open(base_dir / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["max_len"] = max_len  # DECISIONS.md 2026-10-08: run at 512
    _fix_tokenizer_config(str(base_dir))
    tok = AutoTokenizer.from_pretrained(base_dir / "tokenizer")
    encoder = build_model(cfg, encoder_dir=str(base_dir / "encoder")).encoder
    VisionDecisionModel = make_vision_model_class()
    torch.manual_seed(projector_seed)
    model = VisionDecisionModel(encoder, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1, no_init=True)
    weights_path = base_dir / "model.safetensors"
    missing, unexpected = model.load_state_dict(load_file(str(weights_path)), strict=False)
    projector_keys = {k for k in model.state_dict() if k.startswith("projector.")}
    if unexpected or set(missing) != projector_keys:
        raise SystemExit(f"{weights_path}: unexpected keys {unexpected[:5]}, "
                         f"missing non-projector keys {sorted(set(missing) - projector_keys)[:5]}")
    model.float()
    return model, tok, cfg, sha256_file(weights_path)


def freeze_for_training(model):
    """Train projector + decision head; freeze the encoder and the act head."""
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE_PREFIXES))


def parameter_report(model):
    groups = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        g = groups.setdefault(top, {"params": 0, "trainable": 0})
        g["params"] += p.numel()
        g["trainable"] += p.numel() if p.requires_grad else 0
    return {"modules": groups, "trainable_total": sum(g["trainable"] for g in groups.values()),
            "total": sum(g["params"] for g in groups.values())}


def trainable_state(model):
    return {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()
            if k.startswith(TRAINABLE_PREFIXES)}


def load_trainable(model, path):
    from safetensors.torch import load_file

    state = load_file(str(path))
    bad = [k for k in state if not k.startswith(TRAINABLE_PREFIXES)]
    if bad:
        raise SystemExit(f"{path}: non-trainable keys {bad[:5]}")
    expected = {k for k in model.state_dict() if k.startswith(TRAINABLE_PREFIXES)}
    if set(state) != expected:
        raise SystemExit(f"{path}: trainable keys differ from the model ({len(state)} vs {len(expected)})")
    model.load_state_dict(state, strict=False)


# --- question and sequences -------------------------------------------------------

def doc_type_question(records):
    """The internal question dict (as the upstream training script builds it), shared by all records."""
    first = None
    for r in records:
        q = r["questions"][QUESTION_ID]
        q = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
        if first is None:
            first = q
        elif q != first:
            raise SystemExit(f"{r['id']}: doc_type question differs from the first record's.")
    if list(first["crit"]) != rvlcdip.CLASSES:
        raise SystemExit("doc_type options are not in canonical rvlcdip.CLASSES order.")
    return first


class SequenceBuilder:
    """Builds every Laya-Vision sequence; the only place the state order is defined."""

    def __init__(self, tok, question, max_len=MAX_LEN, head_max_len=256, n_visual=N_VISUAL):
        from laya.common import build_head

        self.tok, self.q = tok, question
        self.max_len, self.head_max_len, self.n_visual = max_len, head_max_len, n_visual
        self.placeholder = tok.pad_token_id  # embedding replaced; any id would do
        head_ids, self.markers, stats = build_head(tok, question, head_max_len)
        if stats["options_distinct"] != stats["options"]:
            raise SystemExit(f"question head cannot tell options apart: {stats}")
        self.head_len = len(head_ids)
        self.room = max(0, max_len - self.head_len - 1)  # -1: the closing [SEP]
        self.visual_block = n_visual + 1                 # placeholders + one [SEP]
        if self.room < self.visual_block:
            raise SystemExit(f"max_len {max_len} leaves {self.room} state tokens; the visual block needs "
                             f"{self.visual_block}. Visual tokens are never truncated.")

    def text_budget(self, visual):
        return self.room - (self.visual_block if visual else 0)

    def encode_text(self, texts):
        """Text -> ids exactly as upstream tokenises a state (mask token blanked, no specials)."""
        from laya.common import encode_text

        if isinstance(texts, str):
            texts = [texts]
        clean = [t.replace(self.tok.mask_token, " ") for t in texts]
        return encode_text(self.tok, clean, add_special_tokens=False)["input_ids"]

    def build(self, visual, ocr_ids=(), feature_ids=(), user_ids=()):
        """-> dict(ids, markers, visual_pos, text_tokens, text_tokens_used)."""
        from laya.common import build_sequence

        prefix = ([self.placeholder] * self.n_visual + [self.tok.sep_token_id]) if visual else []
        text = list(ocr_ids) + list(feature_ids) + list(user_ids)
        ids, markers, stats = build_sequence(self.tok, "", self.q, self.max_len, self.head_max_len,
                                             state_ids=prefix + text, return_truncation_stats=True)
        if markers != self.markers:
            raise RuntimeError("option markers moved; the question head is not what the builder measured.")
        visual_pos = list(range(self.head_len, self.head_len + self.n_visual)) if visual else []
        if visual and ids[self.head_len:self.head_len + self.visual_block] != prefix:
            raise RuntimeError("visual block truncated; this must never happen.")
        return {"ids": ids, "markers": markers, "visual_pos": visual_pos,
                "text_tokens": len(text), "text_tokens_used": stats["state_tokens_used"] - len(prefix)}


# --- data -------------------------------------------------------------------------

def read_records(path, expect_split):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    splits = {r["split"] for r in rows}
    if splits != {expect_split}:
        raise SystemExit(f"{path}: expected split {expect_split!r}, found {sorted(splits)}.")
    return rows


def tokenize_ocr(builder, texts, cache_path=None, cache_key=None, cap=MAX_LEN, chunk=512):
    """OCR texts (list, aligned with ids) -> (int32 token arrays capped at `cap`, full lengths).

    Cached on disk under `cache_key` when cache_path is given; a different key recomputes.
    """
    import pickle

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached["key"] == cache_key:
            return cached["ids"], cached["lengths"]
    ids, lengths = [], []
    for start in range(0, len(texts), chunk):
        for row in builder.encode_text(texts[start:start + chunk]):
            lengths.append(len(row))
            ids.append(np.asarray(row[:cap], dtype=np.int32))
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(cache_path) + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"key": cache_key, "ids": ids, "lengths": lengths}, f)
        os.replace(tmp, cache_path)
    return ids, lengths


class VisionData:
    """One split held in memory: fp16 visual tokens, OCR token ids, labels."""

    def __init__(self, ids, emb, ocr_ids, ocr_lengths, labels):
        assert len(ids) == len(emb) == len(ocr_ids) == len(labels)
        self.ids, self.emb, self.ocr_ids, self.ocr_lengths = ids, emb, ocr_ids, ocr_lengths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.index = {i: n for n, i in enumerate(ids)}

    def __len__(self):
        return len(self.ids)

    def subset(self, rows):
        rows = list(rows)
        return VisionData([self.ids[r] for r in rows], self.emb[rows], [self.ocr_ids[r] for r in rows],
                          [self.ocr_lengths[r] for r in rows], self.labels[rows])

    def item(self, builder, row, modality):
        visual = modality in ("combined", "image")
        text = modality in ("combined", "text")
        seq = builder.build(visual, self.ocr_ids[row].tolist() if text else ())
        label = int(self.labels[row])
        target = [0.0] * len(seq["markers"])
        target[label] = 1.0
        return {**seq, "row": row, "modality": modality, "label": label, "target": target,
                "truncated": seq["text_tokens_used"] < (self.ocr_lengths[row] if text else 0)}


def load_split_data(shards_root, records_path, split, builder, cache_dir=None, tokens=N_VISUAL, limit=None):
    """Shards + labels for train or val (never test) -> (VisionData, provenance dict)."""
    import shards

    if split not in ("train", "val"):
        raise SystemExit("Phase 3 code loads train and val only; test is final_eval.py's.")
    records = read_records(records_path, split)
    label_of = {r["id"]: rvlcdip.CLASSES.index(r["label_name"]) for r in records}
    split_dir = Path(shards_root) / split
    emb_cfg, ids, emb = shards.load_split(split_dir, tokens=tokens, dtype=np.float16)
    ocr_cfg, texts = shards.load_ocr(split_dir)
    if limit is not None:
        ids, emb = ids[:limit], emb[:limit]
    missing = [i for i in ids if i not in label_of or i not in texts]
    if missing:
        raise SystemExit(f"{len(missing)} {split} shard ids lack a label or OCR row: {missing[:5]}")
    tok_files = sorted((Path(builder.tok.name_or_path)).glob("*.json"))
    key = {
        "split": split, "n": len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "tokenizer_files_sha256": {p.name: sha256_file(p) for p in tok_files},
        "ocr_pass": ocr_cfg["passes"]["ocr"], "cap": builder.max_len,
    }
    cache = Path(cache_dir) / f"ocr_tokens_{split}_{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]}.pkl" \
        if cache_dir else None
    t0 = time.perf_counter()
    ocr_ids, lengths = tokenize_ocr(builder, [texts[i] for i in ids], cache, key, cap=builder.max_len)
    print(f"{split}: {len(ids)} examples, visual {emb.shape[1:]} fp16, OCR tokens ready in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    provenance = {"split": split, "n": len(ids), "records_sha256": sha256_file(records_path),
                  "shards_identity": emb_cfg["identity"], "embed_pass": emb_cfg["passes"]["embed"],
                  "ocr_pass": ocr_cfg["passes"]["ocr"], "tokens": tokens}
    return VisionData(ids, emb, ocr_ids, lengths, [label_of[i] for i in ids]), provenance


def modality_for(seed, epoch, example_id, probs=DEFAULT_MODALITY_PROBS):
    """Per-example modality from a hash of (seed, epoch, id): deterministic and resume-safe."""
    h = hashlib.sha256(f"{seed}:{epoch}:{example_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    edge = 0.0
    for name, p in zip(MODALITIES, probs):
        edge += p
        if u < edge:
            return name
    return MODALITIES[-1]


def collate(items, data, pad_id, n_visual=N_VISUAL):
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    k = max(len(it["markers"]) for it in items)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attention = torch.zeros((n, L), dtype=torch.long)
    marker_pos = torch.zeros((n, k), dtype=torch.long)
    marker_mask = torch.zeros((n, k), dtype=torch.bool)
    target = torch.zeros((n, k), dtype=torch.float32)
    visual_pos = torch.full((n, n_visual), -1, dtype=torch.long)
    for i, it in enumerate(items):
        input_ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        attention[i, :len(it["ids"])] = 1
        marker_pos[i, :len(it["markers"])] = torch.tensor(it["markers"])
        marker_mask[i, :len(it["markers"])] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"])
        if it["visual_pos"]:
            visual_pos[i] = torch.tensor(it["visual_pos"])
    visual = torch.from_numpy(data.emb[[it["row"] for it in items]].astype(np.float32))
    return {
        "input_ids": input_ids, "attention_mask": attention, "marker_pos": marker_pos, "marker_mask": marker_mask,
        "qtype": torch.zeros(n, dtype=torch.long),  # QTYPES["choice"]
        "target": target, "label": torch.tensor([it["label"] for it in items]),
        "visual": visual, "visual_pos": visual_pos,
    }


def forward(model, batch, device):
    """Image path whenever any row has visual slots; otherwise upstream's forward."""
    args = [batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    if (batch["visual_pos"] >= 0).any():
        return model(*args, visual=batch["visual"].to(device), visual_pos=batch["visual_pos"].to(device))
    return model(*args)


def length_batches(lengths, order, batch_size, rng, pool=50):
    """Group a shuffled order into batches of similar length (pools of `pool` batches), then shuffle batches."""
    batches = []
    span = batch_size * pool
    for start in range(0, len(order), span):
        chunk = sorted(order[start:start + span], key=lambda r: lengths[r])
        batches.extend(chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size))
    rng.shuffle(batches)
    return batches


# --- loss -------------------------------------------------------------------------

def rlcd_loss(logits, mask, target, qtype, sigma, samples=4):
    """Upstream fine-tune loss (notebooks/laya_finetune_typed_decisions_mps.py): CE + RLCD.

    Returns (loss, ce, rl). The act head is untrained (upstream weights it 0).
    """
    from laya.common import proper_reward

    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    noisy = logits.detach().unsqueeze(0) + eps
    probs = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(probs, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    rl = -(advantage * logp).mean()
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return rl + ce, ce.detach(), rl.detach()


def sigma_for_epoch(epoch, epochs):
    """Upstream's exploration noise schedule: 0.4 -> 0.1 linearly over epochs."""
    return 0.4 + (0.1 - 0.4) * epoch / max(1, epochs - 1)


def lr_lambda(total_steps, warmup_steps, base_lr, eta_min=1e-6):
    """Linear warmup, then cosine to eta_min (upstream: CosineAnnealingLR, eta_min 1e-6, no warmup)."""
    def f(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        floor = eta_min / base_lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


# --- evaluation -------------------------------------------------------------------

@torch.no_grad()
def predict(model, data, builder, modality, device, batch_size=64, amp=None, rows=None):
    """-> (probs [n, 16] float32, rows, seconds, truncated count). No temperature (P4)."""
    model.eval()
    rows = list(range(len(data))) if rows is None else list(rows)
    out = np.zeros((len(rows), len(rvlcdip.CLASSES)), dtype=np.float32)
    truncated, seconds = 0, 0.0
    for start in range(0, len(rows), batch_size):
        items = [data.item(builder, r, modality) for r in rows[start:start + batch_size]]
        truncated += sum(it["truncated"] for it in items)
        batch = collate(items, data, builder.tok.pad_token_id)
        sync(device)
        t0 = time.perf_counter()
        with autocast(device, amp):
            logits, _ = forward(model, batch, device)
        sync(device)
        seconds += time.perf_counter() - t0
        out[start:start + len(items)] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out, rows, seconds, truncated


def metrics(probs, labels, bins=15):
    from laya.common import ece_score

    labels = np.asarray(labels)
    pred = probs.argmax(1)
    correct = pred == labels
    conf = probs.max(1)
    onehot = np.eye(probs.shape[1])[labels]
    res = {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "ece_15bin": float(ece_score(conf, correct.astype(float), bins)),
        "mean_confidence": float(conf.mean()),
        "per_class_accuracy": {},
        "neighbourhood_accuracy": {},
    }
    for c, name in enumerate(rvlcdip.CLASSES):
        m = labels == c
        if m.any():
            res["per_class_accuracy"][name] = float(correct[m].mean())
    for hood, names in NEIGHBOURHOODS.items():
        m = np.isin(labels, [rvlcdip.CLASSES.index(n) for n in names])
        if m.any():
            res["neighbourhood_accuracy"][hood] = float(correct[m].mean())
    return res


def autocast(device, amp):
    if amp in (None, "none") or torch.device(device).type != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[amp])


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_batches(data, builder, device, n):
    """The first n rows as a text-only batch and as a combined batch (real visual slots)."""
    rows = list(range(min(n, len(data))))
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    out = {}
    for view in ("text", "combined"):
        batch = collate([data.item(builder, r, view) for r in rows], data, builder.tok.pad_token_id)
        out[view] = ([batch[k].to(device) for k in keys], batch["visual"].to(device), batch["visual_pos"].to(device))
    return out


def _max_diff(a, b):
    """(max |diff| of choice logits, max |diff| of act logits)."""
    return (a[0] - b[0]).abs().max().item(), (a[1] - b[1]).abs().max().item()


def equivalence_check(model, data, builder, device, amp, n=8, tol=1e-3):
    """Required check (DECISIONS.md 2026-10-08), run under the run's own device, attention,
    amp and compile settings. With no visual slots the image path (encoder on input_ids +
    visual hook + copied head) must match DecisionModel.forward; the no-image path must BE
    DecisionModel.forward. With real slots the output must change. Returns max |diff|."""
    was_training = model.training
    model.eval()
    b = _check_batches(data, builder, device, n)
    args, visual, visual_pos = b["text"]
    cargs, cvisual, cvisual_pos = b["combined"]
    with torch.no_grad(), autocast(device, amp):
        ref = _decision_model_class().forward(model, *args)
        none = model(*args)
        emb = model(*args, visual=visual, visual_pos=torch.full_like(visual_pos, -1))
        with_img = model(*cargs, visual=cvisual, visual_pos=cvisual_pos)
        without_img = model(*cargs, visual=cvisual, visual_pos=torch.full_like(cvisual_pos, -1))
    model.train(was_training)
    if not (torch.equal(ref[0], none[0]) and torch.equal(ref[1], none[1])):
        raise SystemExit("equivalence check failed: the no-image path is not DecisionModel.forward.")
    diff = max(_max_diff(ref, emb))
    if diff > tol:
        raise SystemExit(f"equivalence check failed: the image path with no visual slots differs from "
                         f"DecisionModel.forward by {diff:.2e} (> {tol}) on {device} (amp {amp}, attention "
                         f"{model.encoder.config._attn_implementation}, compile "
                         f"{model.encoder.config.reference_compile}). Run train_vision.py --diag-equivalence.")
    if torch.equal(with_img[0], without_img[0]):
        raise SystemExit("equivalence check failed: visual slots did not change the output (hook not applied).")
    return diff


def _with_attention(model, impl):
    """A copy of `model` whose encoder is rebuilt with attention implementation `impl`
    (ModernBERT picks its rotary module at construction, so flash needs a rebuild)."""
    import copy

    from laya.common import _no_init_weights
    from transformers import AutoModel

    cfg = copy.deepcopy(model.encoder.config)
    with _no_init_weights():
        enc = AutoModel.from_config(cfg, attn_implementation=impl)
    enc.load_state_dict(model.encoder.state_dict())
    enc.embeddings.register_forward_hook(_inject_visual)
    encoder = model.encoder
    del model.encoder
    try:
        clone = copy.deepcopy(model)
    finally:
        model.encoder = encoder
    clone.encoder = enc
    return clone


def diagnose_equivalence(model, data, builder, device, n=8):
    """--diag-equivalence: where does the image path diverge from DecisionModel.forward on this device?

    Starts from fp32 / compile off / eager attention, then runs every combination of
    attention (eager, sdpa, flash_attention_2 if installed) x amp (none, bf16) x
    ModernBERT reference_compile (off, on). For each it prints max |diff| against
    DecisionModel.forward (choice logits / act logits) for:
      hook      the image path as fixed (encoder on input_ids, visual hook), no slots
      embeds    the pre-fix path (encoder(inputs_embeds=...)), no slots
      repeat    DecisionModel.forward run twice (run-to-run noise floor)
    plus hook-vs-embeds with real visual slots, and the largest |logit| / |act logit|
    (one bf16 step at magnitude m is 2**(floor(log2 m) - 7)). Returns the rows.
    """
    from transformers.utils import is_flash_attn_2_available

    D = _decision_model_class()
    impls = ["eager", "sdpa"]
    if torch.device(device).type == "cuda" and is_flash_attn_2_available():
        impls.append("flash_attention_2")
    else:
        print("flash_attention_2: flash_attn not installed (or not on CUDA), skipped", flush=True)
    model = model.float().cpu().eval()
    baseline = ("eager", "none", False)
    rows = []
    print(f"{'attention':18s} {'amp':5s} {'compile':7s} | {'hook logit/act':17s} | {'embeds logit/act':17s} | "
          f"{'repeat':8s} | {'hook~embeds slots':17s} | max|logit| max|act|", flush=True)
    for impl in impls:
        m = _with_attention(model, impl).to(device).eval()
        b = _check_batches(data, builder, device, n)
        args, visual, visual_pos = b["text"]
        cargs, cvisual, cvisual_pos = b["combined"]
        no_slots = torch.full_like(visual_pos, -1)
        for amp in ("none", "bf16"):
            for comp in (False, True):
                if torch.device(device).type != "cuda" and (comp or amp != "none"):
                    continue
                m.encoder.config.reference_compile = comp
                try:
                    with torch.no_grad(), autocast(device, None if amp == "none" else amp):
                        ref = D.forward(m, *args)
                        ref2 = D.forward(m, *args)
                        hook = m(*args, visual=visual, visual_pos=no_slots)
                        embeds = m.forward_inputs_embeds(*args, visual, no_slots)
                        hook_s = m(*cargs, visual=cvisual, visual_pos=cvisual_pos)
                        embeds_s = m.forward_inputs_embeds(*cargs, cvisual, cvisual_pos)
                except Exception as e:  # report and keep going: one combination failing is a finding
                    print(f"{impl:18s} {amp:5s} {str(comp):7s} | ERROR {type(e).__name__}: {e}", flush=True)
                    rows.append({"attention": impl, "amp": amp, "compile": comp,
                                 "error": f"{type(e).__name__}: {e}"})
                    continue
                masked = ref[0].masked_fill(~args[3], 0)
                key = (impl, amp, comp)
                row = {"attention": impl, "amp": amp, "compile": comp, "baseline": key == baseline,
                       "one_factor_from_baseline": sum(x != y for x, y in zip(key, baseline)) == 1,
                       "hook": _max_diff(ref, hook), "embeds": _max_diff(ref, embeds),
                       "repeat": max(_max_diff(ref, ref2)), "hook_vs_embeds_slots": _max_diff(hook_s, embeds_s),
                       "max_abs_logit": masked.abs().max().item(), "max_abs_act": ref[1].abs().max().item()}
                rows.append(row)
                tag = " <- baseline" if row["baseline"] else (" <- one factor" if row["one_factor_from_baseline"]
                                                               else "")
                print(f"{impl:18s} {amp:5s} {str(comp):7s} | {row['hook'][0]:.1e}/{row['hook'][1]:.1e} | "
                      f"{row['embeds'][0]:.1e}/{row['embeds'][1]:.1e} | {row['repeat']:.1e} | "
                      f"{row['hook_vs_embeds_slots'][0]:.1e}/{row['hook_vs_embeds_slots'][1]:.1e} | "
                      f"{row['max_abs_logit']:.1f} {row['max_abs_act']:.1f}{tag}", flush=True)
        del m
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return rows

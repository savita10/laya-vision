"""Laya-Vision model: laya's DecisionModel + a projector for SigLIP visual tokens.

Backbone: the Gate-0 fine-tuned convaiinnovations/laya checkpoint (ModernBERT-large,
d=1024), run at max_len 512. Frozen: the encoder and the act head. Trained: the
projector (768 -> 1024 -> 1024 MLP) and the decision head (head layers, type
embedding, scorer).

Laya touch-points (DECISIONS.md 2026-10-08): VisionDecisionModel subclasses
laya.common.DecisionModel; its no-image path IS DecisionModel.forward, and its
image path calls the encoder with input_ids, injects visual tokens with a
forward hook on encoder.embeddings, and duplicates the post-encoder part of
DecisionModel.forward as of laya 0.3.24 (guarded by
tests/test_vision_equivalence.py and a startup check).
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

LAYA_VERSION = "0.3.24"
MAX_LEN = 512
VISUAL_DIM = 768
TRAINABLE_PREFIXES = ("projector.", "head.", "type_emb.", "scorer.")


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


def _tokenizer_needs_fix(cfg_file):
    """The two cases laya.agent._fix_tokenizer_config (0.3.24) rewrites."""
    if not cfg_file.exists():
        return False
    with open(cfg_file, encoding="utf-8") as f:
        tcfg = json.load(f)
    return tcfg.get("tokenizer_class") in (None, "TokenizersBackend") or isinstance(
        tcfg.get("extra_special_tokens"), list)


def load_tokenizer(base_dir):
    """The checkpoint's tokenizer, with laya's tokenizer_config fix applied to a temporary copy.

    laya.agent._fix_tokenizer_config rewrites base_dir/tokenizer/tokenizer_config.json in place;
    a release folder is never modified, so the fix runs on a copy when it is needed at all.
    """
    import shutil
    import tempfile

    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    tok_dir = Path(base_dir) / "tokenizer"
    if not _tokenizer_needs_fix(tok_dir / "tokenizer_config.json"):
        return AutoTokenizer.from_pretrained(tok_dir)
    with tempfile.TemporaryDirectory(prefix="laya_vision_tok_") as tmp:
        shutil.copytree(tok_dir, Path(tmp) / "tokenizer")
        _fix_tokenizer_config(tmp)
        return AutoTokenizer.from_pretrained(Path(tmp) / "tokenizer")


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

    Every base weight must load; the only missing keys allowed are the projector's. No file in
    base_dir is written (see load_tokenizer).
    """
    from laya.common import build_model
    from safetensors.torch import load_file

    base_dir = Path(base_dir)
    with open(base_dir / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["max_len"] = max_len  # DECISIONS.md 2026-10-08: run at 512
    tok = load_tokenizer(base_dir)
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

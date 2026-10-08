"""Required test (DECISIONS.md 2026-10-08): Laya-Vision's no-image path is upstream's.

Model checks run on CPU with a small random ModernBERT (no download):
  1. VisionDecisionModel without an image returns exactly DecisionModel.forward.
  2. The image path with no visual slots equals DecisionModel.forward exactly,
     so the copied post-encoder head code is in step with laya 0.3.24.
  3. With visual slots the output changes, gradients reach the projector and
     the head, and none reach the frozen encoder or act head.
  3b. With visual slots, the hook path (encoder on input_ids, embedding output
     replaced at the slots) equals the inputs_embeds formulation (projector
     output as a pre-norm token embedding), and the hook's unpadded
     (flash-attention) branch writes the same tokens as its padded branch.
     The GPU-only factors (compiled embeddings, bf16, flash) are covered by
     train_vision.py --diag-equivalence and the startup check.
Sequence checks need the Gate-0 tokenizer (--base-ckpt, or the pinned
convaiinnovations/laya revision from the Hub; skipped if neither loads):
  4. A text-only sequence is token-for-token upstream build_sequence's.
  5. A combined sequence keeps the markers and the whole visual block when
     the text overflows; only text is truncated.

    python tests/test_vision_equivalence.py [--base-ckpt <dir>]
    pytest tests/test_vision_equivalence.py
"""

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import vision  # noqa: E402

BASE_REPO = "convaiinnovations/laya"
BASE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"  # laya.revisions pin in 0.3.24


def tiny_models(seed=0):
    from laya.common import DecisionModel
    from transformers import AutoModel, ModernBertConfig

    torch.manual_seed(seed)
    cfg = ModernBertConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                           num_attention_heads=4, global_attn_every_n_layers=3, local_attention=8,
                           max_position_embeddings=256, pad_token_id=0, reference_compile=False)
    enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
    base = DecisionModel(enc, head_layers=2, n_act=2).eval()
    VisionDecisionModel = vision.make_vision_model_class()
    vis = VisionDecisionModel(copy.deepcopy(enc), head_layers=2, n_act=2, visual_dim=24)
    missing, unexpected = vis.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("projector.") for k in missing)
    return base, vis.eval()


def tiny_batch(seed=1, n=3, length=40, k=5):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, 128, (n, length), generator=g)
    att = torch.ones(n, length, dtype=torch.long)
    for i, used in enumerate((length, length - 7, length - 15)[:n]):
        ids[i, used:] = 0
        att[i, used:] = 0
    marker_pos = torch.stack([torch.arange(2, 2 + 2 * k, 2) for _ in range(n)])
    marker_mask = torch.ones(n, k, dtype=torch.bool)
    qtype = torch.zeros(n, dtype=torch.long)
    return ids, att, marker_pos, marker_mask, qtype


def test_no_image_is_upstream():
    base, vis = tiny_models()
    args = tiny_batch()
    with torch.no_grad():
        a, b = base(*args), vis(*args)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_embeds_path_matches_upstream():
    base, vis = tiny_models()
    args = tiny_batch()
    visual = torch.randn(args[0].size(0), 6, 24)
    no_slots = torch.full((args[0].size(0), 6), -1, dtype=torch.long)
    with torch.no_grad():
        a = base(*args)
        b = vis(*args, visual=visual, visual_pos=no_slots)
    # Same encoder call (input_ids) and the copied head: equal, not just close.
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_visual_slots_train_only_projector_and_head():
    _, vis = tiny_models()
    vision.freeze_for_training(vis)
    vis.train()
    args = tiny_batch()
    n = args[0].size(0)
    visual = torch.randn(n, 6, 24)
    slots = torch.arange(12, 18).repeat(n, 1)
    slots[1] = -1  # a text-only row in a mixed batch
    with torch.no_grad():
        vis.eval()
        plain = vis(*args)[0]
        with_img = vis(*args, visual=visual, visual_pos=slots)[0]
        vis.train()
    assert not torch.allclose(plain[0], with_img[0])
    logits, _ = vis(*args, visual=visual, visual_pos=slots)
    logits.logsumexp(-1).sum().backward()
    for name, p in vis.named_parameters():
        if name.startswith(("projector.", "head.", "scorer.")):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        if name.startswith(("encoder.", "act_head.")):
            assert p.grad is None, name


def test_hook_matches_inputs_embeds_with_slots():
    _, vis = tiny_models()
    args = tiny_batch()
    n = args[0].size(0)
    visual = torch.randn(n, 6, 24)
    slots = torch.arange(12, 18).repeat(n, 1)
    slots[1] = -1
    with torch.no_grad():
        a = vis(*args, visual=visual, visual_pos=slots)
        b = vis.forward_inputs_embeds(*args, visual, slots)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)
    assert vis.encoder.embeddings.__dict__.get("_laya_visual") is None


def test_hook_unpadded_branch():
    _, vis = tiny_models()
    emb = vis.encoder.embeddings.eval()
    ids, att, *_ = tiny_batch()
    slots = torch.arange(3, 9).repeat(ids.size(0), 1)
    slots[1, 4:] = -1
    keep = slots >= 0
    rows = torch.arange(ids.size(0))[:, None].expand_as(slots)
    tokens = torch.randn(ids.size(0), 6, emb.tok_embeddings.weight.size(1))[keep]
    with torch.no_grad():
        padded = emb(input_ids=ids)
        unpadded = padded[att.bool()]
        inject = (rows[keep], slots[keep], tokens, att)
        object.__setattr__(emb, "_laya_visual", inject)
        try:
            out_p = vision._inject_visual(emb, (), padded)
            out_u = vision._inject_visual(emb, (), unpadded)
        finally:
            object.__setattr__(emb, "_laya_visual", None)
    assert not torch.equal(out_p, padded)
    assert torch.equal(out_p[att.bool()], out_u)


def load_tokenizer(base_ckpt=None):
    from transformers import AutoTokenizer

    if base_ckpt:
        from laya.agent import _fix_tokenizer_config

        _fix_tokenizer_config(str(base_ckpt))
        return AutoTokenizer.from_pretrained(Path(base_ckpt) / "tokenizer"), 256
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(BASE_REPO, revision=BASE_REVISION, allow_patterns=["tokenizer/*"])
        from laya.agent import _fix_tokenizer_config

        _fix_tokenizer_config(path)
        return AutoTokenizer.from_pretrained(Path(path) / "tokenizer"), 256
    except Exception as e:  # offline: sequence checks are skipped, model checks still ran
        print(f"SKIP sequence checks: tokenizer not available ({type(e).__name__}: {e})")
        return None, None


def doc_question():
    import rvlcdip

    return {"t": "choice", "ins": "What type of document is this?", "crit": {c: "" for c in rvlcdip.CLASSES}}


def check_sequences(tok, head_max_len):
    from laya.common import build_sequence

    q = doc_question()
    b = vision.SequenceBuilder(tok, q, 512, head_max_len, 32)
    text = "MEMORANDUM\nTo: All staff\nFrom: R. J. Reynolds [MASK] research\nSubject: quarterly budget review"
    ours = b.build(False, b.encode_text(text)[0])
    ids, markers = build_sequence(tok, text, q, 512, head_max_len)
    assert ours["ids"] == ids and ours["markers"] == markers, "text-only sequence differs from upstream"

    long_text = " ".join(["invoice total amount due"] * 400)
    combined = b.build(True, b.encode_text(long_text)[0])
    assert len(combined["ids"]) == 512 and combined["markers"] == b.markers
    vp = combined["visual_pos"]
    assert vp == list(range(b.head_len, b.head_len + 32))
    assert all(combined["ids"][p] == b.placeholder for p in vp) and combined["ids"][vp[-1] + 1] == tok.sep_token_id
    assert combined["text_tokens_used"] == b.text_budget(True)
    image_only = b.build(True)
    assert len(image_only["ids"]) == b.head_len + 33 + 1
    return b


def test_sequences():
    tok, head_max_len = load_tokenizer()
    if tok is not None:
        check_sequences(tok, head_max_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", default=None)
    args = p.parse_args()
    vision.check_laya_version()
    test_no_image_is_upstream()
    print("ok 1  no-image path == DecisionModel.forward (exact)")
    test_embeds_path_matches_upstream()
    print("ok 2  image path, no slots == DecisionModel.forward (exact)")
    test_visual_slots_train_only_projector_and_head()
    print("ok 3  visual slots change output; grads reach projector + head only")
    test_hook_matches_inputs_embeds_with_slots()
    test_hook_unpadded_branch()
    print("ok 3b visual hook == inputs_embeds formulation; unpadded branch == padded branch")
    tok, head_max_len = load_tokenizer(args.base_ckpt)
    if tok is not None:
        b = check_sequences(tok, head_max_len)
        print(f"ok 4  text-only sequence == upstream build_sequence")
        print(f"ok 5  visual block and markers survive overflow (head {b.head_len} tokens, text budget "
              f"combined {b.text_budget(True)}, text-only {b.text_budget(False)})")


if __name__ == "__main__":
    main()

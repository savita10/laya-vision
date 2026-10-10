"""The SDK (laya_vision.VisionAgent, scripts/demo.py) reproduces the training and evaluation inputs.

Without a release (CPU, no download):
  1. laya_vision.image.letterbox is pixel-identical to precompute.letterbox (p2v1).
  2. laya_vision.image.layout_text is precompute.TesseractOCR.one's text layout.
With --release <dir> (downloads the pinned SigLIP on first run):
  3. SiglipEncoder equals precompute.Embedder (fp32, 8x8) followed by shards.derive_view.
  4. VisionAgent's image path, image-only and with text, builds the same sequence as
     vision.SequenceBuilder and returns softmax(logits / T_view) of the scripts' forward.
  5. image=None returns upstream laya.Agent's answer on the same weights: the scripts' text-view
     logits divided by the base's rl_agent_config temperature.
  6. A corrupt image raises ImageError.
  7. With an image, a question other than the trained doc_type gets T = 1.0 and
     "calibrated": false; the trained question (under any id) keeps its fitted temperature and
     "calibrated": true; reordered options count as another question.

    python tests/test_sdk_equivalence.py [--release release]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from laya_vision import image as lv_image  # noqa: E402


def sample_images():
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    out = []
    for size, mode in (((754, 1000), "L"), ((1000, 754), "RGB"), ((383, 500), "L"), ((1700, 2200), "1"),
                       ((600, 600), "RGBA")):
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(arr).convert(mode)
        d = ImageDraw.Draw(img)
        d.rectangle([10, 10, size[0] // 2, size[1] // 3], fill=0 if mode in ("L", "1") else (0, 0, 0)[:len(mode)])
        out.append(img)
    return out


def document_image(lines):
    from PIL import Image, ImageDraw

    img = Image.new("L", (754, 1000), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((60, 60 + 22 * i), line, fill=0)
    return img


def test_letterbox():
    import precompute

    for img in sample_images():
        a, b = lv_image.letterbox(img), precompute.letterbox(img)
        assert a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b)), img.size
    print("ok  1 letterbox == precompute.letterbox (5 sizes / modes)")


def test_layout_text():
    import precompute

    data = {"text": ["Dear", "Sir,", "", "Thank", "you", "for", "", "Yours", " "],
            "block_num": [1, 1, 1, 1, 1, 1, 2, 2, 2], "par_num": [1, 1, 1, 2, 2, 2, 1, 1, 1],
            "line_num": [1, 1, 1, 1, 1, 2, 1, 1, 1], "left": [0] * 9, "top": [0] * 9, "width": [1] * 9,
            "height": [1] * 9, "conf": [90] * 9}

    class FakePT:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(img, lang, config, output_type):
            return data

    ocr = precompute.TesseractOCR.__new__(precompute.TesseractOCR)
    ocr.pytesseract = FakePT
    expected = ocr.one(None)["text"]
    assert lv_image.layout_text(data) == expected, (lv_image.layout_text(data), expected)
    print("ok  2 layout_text == precompute.TesseractOCR.one text")


def test_release(release_dir):
    import torch
    import precompute
    import shards
    import vision
    from laya_vision import ImageError, VisionAgent

    agent = VisionAgent(release_dir=release_dir, device="cpu")
    cfg = agent.config
    imgs = [document_image(["MEMORANDUM", "TO: All staff", "FROM: Research", "RE: Budget review"]),
            sample_images()[1]]

    # 3. SigLIP encoder vs precompute.Embedder + shards.derive_view
    emb = precompute.Embedder("cpu", "fp32", tuple(cfg["siglip"]["pooling"]["grid_rows_cols"]))
    canv = [lv_image.letterbox(im) for im in imgs]
    ref = shards.derive_view(emb(canv), tuple(cfg["siglip"]["pooling"]["grid_rows_cols"]), cfg["visual_tokens"])
    got = np.stack([agent._visual(c) for c in canv])
    assert np.array_equal(got, ref), np.abs(got - ref).max()
    print(f"ok  3 SiglipEncoder == precompute.Embedder + derive_view ({cfg['visual_tokens']} tokens)")

    # 4. image path vs the scripts' SequenceBuilder + forward
    q = agent.doc_type_question()
    internal = {"t": "choice", "ins": q["doc_type"]["instructions"], "crit": {c: "" for c in cfg["doc_type"]["classes"]}}
    builder = vision.SequenceBuilder(agent.tok, internal, max_len=cfg["max_len"], head_max_len=cfg["head_max_len"],
                                     n_visual=cfg["visual_tokens"])
    assert builder.head_len == cfg["head_len_doc_type"]
    model = agent.release.model
    for img, user_text in ((imgs[0], None), (imgs[0], "Memo about the budget review. " * 120)):
        res = agent.predict(user_text, q, image=img)
        visual = torch.from_numpy(agent._visual(lv_image.letterbox(img)))[None]
        built = builder.build(True, user_ids=builder.encode_text(user_text)[0] if user_text else ())
        ids = torch.tensor([built["ids"]])
        with torch.no_grad():
            logits, _ = model(ids, torch.ones_like(ids), torch.tensor([built["markers"]]),
                              torch.ones(1, len(built["markers"]), dtype=torch.bool), torch.zeros(1, dtype=torch.long),
                              visual=visual, visual_pos=torch.tensor([built["visual_pos"]]))
        view = "combined" if user_text else "image"
        assert res["usage"]["view"] == view and res["answers"]["doc_type"]["calibrated"] is True
        p = vision.softmax_t(logits.numpy(), agent.release.temperatures[view])[0]
        got = np.array([res["answers"]["doc_type"]["probabilities"][c] for c in cfg["doc_type"]["classes"]])
        assert np.abs(got - p).max() <= 5e-5 + 1e-12, np.abs(got - p).max()  # upstream rounds to 4 places
        assert res["answers"]["doc_type"]["choice"] == cfg["doc_type"]["classes"][int(p.argmax())]
        assert res["usage"]["truncated"] == (built["text_tokens_used"] < built["text_tokens"])
        print(f"ok  4 image path ({view}{', text truncated' if res['usage']['truncated'] else ''}) == scripts "
              f"forward at T={agent.release.temperatures[view]:.4f}: {res['answers']['doc_type']['choice']}")

    # 7. untrained questions with an image: T = 1.0, calibrated false
    classes = cfg["doc_type"]["classes"]
    qs = {"kind": dict(q["doc_type"]),
          "reordered": dict(q["doc_type"], criteria=classes[::-1]),
          "is_memo": {"type": "noul", "instructions": "Is this a memo?"}}
    res = agent.predict(None, qs, image=imgs[0])
    assert res["answers"]["kind"]["calibrated"] is True and res["usage"]["temperatures"]["kind"] != 1.0
    for qid in ("reordered", "is_memo"):
        assert res["answers"][qid]["calibrated"] is False and res["usage"]["temperatures"][qid] == 1.0
    one = agent.predict(None, {"r": qs["reordered"]}, image=imgs[0])
    visual = torch.from_numpy(agent._visual(lv_image.letterbox(imgs[0])))[None]
    b2 = vision.SequenceBuilder(agent.tok, {"t": "choice", "ins": q["doc_type"]["instructions"],
                                            "crit": {c: "" for c in classes[::-1]}},
                                max_len=cfg["max_len"], head_max_len=cfg["head_max_len"], n_visual=cfg["visual_tokens"])
    built = b2.build(True)
    ids = torch.tensor([built["ids"]])
    with torch.no_grad():
        logits, _ = model(ids, torch.ones_like(ids), torch.tensor([built["markers"]]),
                          torch.ones(1, len(built["markers"]), dtype=torch.bool), torch.zeros(1, dtype=torch.long),
                          visual=visual, visual_pos=torch.tensor([built["visual_pos"]]))
    p = vision.softmax_t(logits.numpy(), 1.0)[0]
    got = np.array([one["answers"]["r"]["probabilities"][c] for c in classes[::-1]])
    assert np.abs(got - p).max() <= 5e-5 + 1e-12
    print("ok  7 untrained questions with an image: T = 1.0, calibrated false; trained question under "
          "another id: fitted T, calibrated true")

    # 5. image=None: upstream Agent on the same weights
    text = "MEMORANDUM\nTO: All staff\nFROM: Research\nRE: Budget review"
    res = agent.predict(text, q)
    built = builder.build(False, ocr_ids=builder.encode_text(text)[0])
    ids = torch.tensor([built["ids"]])
    with torch.no_grad():
        logits, _ = model(ids, torch.ones_like(ids), torch.tensor([built["markers"]]),
                          torch.ones(1, len(built["markers"]), dtype=torch.bool), torch.zeros(1, dtype=torch.long))
    t_up = agent.release.upstream_temperature("choice", len(cfg["doc_type"]["classes"]))
    p = vision.softmax_t(logits.numpy(), t_up)[0]
    got = np.array([res["answers"]["doc_type"]["probabilities"][c] for c in cfg["doc_type"]["classes"]])
    assert res["model"] == "laya-rl-agent" and np.abs(got - p).max() <= 5e-5 + 1e-12
    print(f"ok  5 image=None == upstream path at the base's temperature {t_up:.4f}")

    # 6. corrupt image
    try:
        agent.predict(None, q, image=b"not an image")
    except ImageError as e:
        print(f"ok  6 corrupt image refused: {str(e)[:60]}")
    else:
        raise AssertionError("corrupt image accepted")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=None)
    a = ap.parse_args()
    test_letterbox()
    test_layout_text()
    if a.release:
        test_release(a.release)

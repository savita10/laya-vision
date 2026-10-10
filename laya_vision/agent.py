"""VisionAgent: the Laya-Vision SDK. predict(state, questions, image=None).

- image=None: upstream laya's Agent, unchanged, on Laya-Vision's weights (same tokenisation,
  sequence, forward, rl_agent_config.json temperatures, response). Its max_len is the base's
  rl_agent_config.json value, as upstream.
- image given: [CLS] head [SEP] [visual tokens | SEP | OCR text | user text] [SEP] at the release's
  max_len, options and visual tokens never truncated, text truncated from the end. The response
  has upstream's shape plus "calibrated" on each answer (DECISIONS.md 2026-10-10, "SDK
  temperatures for questions asked with an image"):
  - the trained doc_type question (same type, instructions and 16 options in training order,
    whatever its id) gets the release's fitted temperature for the view ("combined" when the
    sequence holds any text, else "image"): "calibrated": true;
  - any other question gets T = 1.0: "calibrated": false. Its accuracy is not measured.
"""

import gc
import json

import numpy as np

from .image import SiglipEncoder, TesseractOCR, letterbox, open_image
from .release import BASE_DIR, CONFIG_NAME, _resolve, load_release

DOC_TYPE_ID = "doc_type"
DOC_TYPE_INSTRUCTIONS = "What type of document is this?"  # scripts/build_dataset.py, the trained question


class VisionAgent:
    """Load a release folder (or Hub repo id) once; then predict(state, questions, image=None)."""

    def __init__(self, release_dir="release", device=None, revision=None):
        from laya import Agent

        root = _resolve(release_dir, revision)
        with open(root / CONFIG_NAME, encoding="utf-8") as f:
            config = json.load(f)
        prefix = BASE_DIR + "/"
        base_hashes = {rel[len(prefix):]: sha for rel, sha in config["files"].items() if rel.startswith(prefix)}
        # Upstream's runtime: device, autocast policy, tokenizer, temperatures and the image=None path.
        # Its weights are checked against the release before they are parsed, then replaced by the
        # Laya-Vision model (whose no-image forward is upstream's).
        self._agent = Agent(str(root / BASE_DIR), device=device, expected_sha256=base_hashes)
        self.device = self._agent.device
        self._agent.model = None
        gc.collect()
        self.release = load_release(root, device=str(self.device))
        self._agent.model = self.release.model
        self.config = self.release.config
        self.tok = self._agent.tok
        self.n_visual = self.config["visual_tokens"]
        self._encoder = None
        self._ocr = None

    # --- public -------------------------------------------------------------------

    def doc_type_question(self):
        """The trained question: 16 RVL-CDIP classes, in training order."""
        return {DOC_TYPE_ID: {"type": "choice", "instructions": DOC_TYPE_INSTRUCTIONS,
                              "criteria": list(self.config["doc_type"]["classes"])}}

    def _is_trained(self, q):
        """Internal question q is the trained doc_type question: the head the temperatures were fitted on."""
        crit = q["crit"]
        return (q["t"] == "choice" and q["ins"] == DOC_TYPE_INSTRUCTIONS and isinstance(crit, dict)
                and list(crit) == list(self.config["doc_type"]["classes"])
                and all(v in (None, "") for v in crit.values())
                and "option_order" not in q and "labels" not in q)

    def predict(self, state=None, questions=None, image=None, ocr=False):
        """Typed answers in upstream laya's response shape.

        state: text, dict or list (upstream's state), or None with an image.
        image: path, bytes or PIL image. None runs upstream laya exactly.
        ocr: with an image, run Tesseract on the original image and put its text after the visual
             tokens (the "combined" view). Needs pytesseract and the tesseract binary.
        """
        if questions is None:
            raise TypeError("questions is required")
        if image is None:
            if ocr:
                raise ValueError("ocr=True needs an image.")
            return self._agent.predict(state, questions)
        return self._predict_image(state, questions, image, ocr)

    # --- image path ---------------------------------------------------------------

    def _visual(self, canvas):
        if self._encoder is None:
            self._encoder = SiglipEncoder(self.config["siglip"], self.n_visual, device=self.device)
        return self._encoder([canvas])[0]

    def _text_ids(self, state, ocr_text):
        from laya.common import encode_text, serialize_state

        texts = []
        if ocr_text:
            texts.append(ocr_text)
        if state is not None:
            texts.append(serialize_state(state))
        ids = []
        for t in texts:  # state order: OCR text, then user text (no derived features in v1)
            ids += encode_text(self.tok, t.replace(self.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        return ids

    def _items(self, ids, internal, text_ids):
        from laya.common import QTYPES, build_head, build_sequence, render_options

        max_len, head_max_len = self.config["max_len"], self.config["head_max_len"]
        prefix = [self.tok.pad_token_id] * self.n_visual + [self.tok.sep_token_id]
        items = []
        for qid in ids:
            q = internal[qid]
            head, _, _ = build_head(self.tok, q, head_max_len, option_order=q.get("option_order"))
            if max_len - len(head) - 1 < len(prefix):
                raise ValueError(f"question {qid!r}: its head ({len(head)} tokens) leaves no room for the "
                                 f"{self.n_visual} visual tokens in max_len {max_len}.")
            seq, markers, stats, st = build_sequence(
                self.tok, "", q, max_len, head_max_len, option_order=q.get("option_order"),
                state_ids=prefix + text_ids, return_stats=True, return_truncation_stats=True)
            if len(markers) != len(render_options(q)):
                raise ValueError(f"question {qid!r}: options do not fit in max_len {max_len}.")
            if seq[len(head):len(head) + len(prefix)] != prefix:
                raise RuntimeError("visual block truncated; this must never happen.")
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "options": stats,
                          "visual_pos": list(range(len(head), len(head) + self.n_visual)),
                          "text_dropped": st["state_tokens_dropped"]})
        return items

    def _predict_image(self, state, questions, image, ocr):
        import torch
        from laya.agent import _amp_context
        from laya.common import collate_items, temp_bucket

        if not isinstance(questions, dict) or not questions:
            raise TypeError("questions must be a non-empty dict of question id -> definition")
        img = open_image(image)
        visual = self._visual(letterbox(img))
        ocr_text = None
        if ocr:
            if self._ocr is None:
                self._ocr = TesseractOCR(self.config["ocr"])
            ocr_text = self._ocr(img)  # the original image, full resolution
        text_ids = self._text_ids(state, ocr_text)
        view = "combined" if text_ids else "image"

        agent = self._agent
        ids = list(questions)
        for qid in ids:
            agent._check_question(qid, questions[qid])
        internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
        items = self._items(ids, internal, text_ids)
        calibrated = {qid: self._is_trained(internal[qid]) for qid in ids}
        temperatures = {qid: self.release.temperatures[view] if calibrated[qid] else 1.0 for qid in ids}
        b = collate_items([items], self.tok.pad_token_id)
        n = len(items)
        vis = torch.from_numpy(np.ascontiguousarray(visual)).float()[None].expand(n, -1, -1)
        vis_pos = torch.tensor([it["visual_pos"] for it in items])
        d = self.device
        with torch.no_grad(), _amp_context(d, agent.dtype, agent._amp_enabled_for(n)):
            logits, act = self.release.model(
                b["input_ids"].to(d), b["attention_mask"].to(d), b["marker_pos"].to(d), b["marker_mask"].to(d),
                b["qtype"].to(d), visual=vis.to(d), visual_pos=vis_pos.to(d))
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        # upstream's decoder divides each row by its own temperature; pre-scale so it divides by ours.
        for j, (qid, it) in enumerate(zip(ids, items)):
            k = len(it["markers"])
            t_up = agent.temperature_by_options.get(temp_bucket(it["qtype"], k), agent.temperature[it["qtype"]])
            logits[j, :k] *= t_up / temperatures[qid]
        answers = agent._decode_answers(logits, act, items, ids, internal, 0)
        for qid in ids:
            answers[qid]["calibrated"] = calibrated[qid]

        dropped = max(it["text_dropped"] for it in items)
        return {
            "model": "laya-vision",
            "answers": answers,
            "usage": {
                "input_tokens": int(b["attention_mask"].sum()),
                "output_tokens": 0,
                "visual_tokens": self.n_visual,
                "state_tokens": len(text_ids),
                "state_tokens_dropped": dropped,
                "truncated": dropped > 0,
                "truncated_questions": [qid for qid, it in zip(ids, items) if it["text_dropped"]],
                "view": view,
                "temperatures": temperatures,
                "ocr": ocr,
            },
        }

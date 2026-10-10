"""Image inputs for inference: decode, preprocessing p2v1, SigLIP visual tokens, Tesseract OCR.

These reproduce what scripts/precompute.py did to every training image (the release config's
"preprocess", "siglip" and "ocr" entries); tests/test_sdk_equivalence.py checks they agree with
precompute.py. Never change one without the other: a new preprocessing version is a new model.
"""

import io
from pathlib import Path

import numpy as np

PREPROCESS_VERSION = "p2v1"
IMAGE_SIZE = 384
CONTRAST_CUTOFF_PERCENT = 1
PAD_RGB = (255, 255, 255)


class ImageError(ValueError):
    """The image could not be decoded."""


def open_image(image):
    """Path, bytes or PIL image -> decoded PIL image, as stored (no resize, no mode change).

    Multi-page files give their first page. An unreadable image raises ImageError; there is no
    silent text-only fallback (CLAUDE.md, failure rules).
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(image, Image.Image):
        return image
    try:
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            raise ImageError(f"image must be a path, bytes or a PIL image, got {type(image).__name__}.")
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageError(f"could not decode image: {e}") from e
    return img


def letterbox(img):
    """Preprocessing p2v1 on a decoded original image -> 384x384 RGB (as precompute.letterbox)."""
    from PIL import Image, ImageOps

    rgb = img.convert("RGB")
    rgb = ImageOps.autocontrast(rgb, cutoff=CONTRAST_CUTOFF_PERCENT, preserve_tone=True)
    w, h = rgb.size
    scale = IMAGE_SIZE / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    rgb = rgb.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), PAD_RGB)
    canvas.paste(rgb, ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2))
    return canvas


def derive_view(emb, stored_grid, view_grid):
    """[n, R*C, d] row-major on stored_grid -> [n, r*c, d] float32 block means (as shards.derive_view)."""
    R, C = stored_grid
    r, c = view_grid
    n, t, d = emb.shape
    if t != R * C or R % r or C % c:
        raise ValueError(f"cannot derive a {r}x{c} view from {t} tokens on a {R}x{C} grid")
    x = emb.astype(np.float32).reshape(n, r, R // r, c, C // c, d)
    return x.mean(axis=(2, 4)).reshape(n, r * c, d)


# Token count -> grid of the derived view (scripts/shards.py TOKEN_VIEWS).
TOKEN_VIEWS = {64: (8, 8), 32: (8, 4), 16: (4, 4), 8: (4, 2)}


class SiglipEncoder:
    """Frozen SigLIP vision tower -> patch grid -> fixed average pooling -> fp16 (as stored) ->
    the release's token view in float32. Built from the release config's "siglip" entry."""

    def __init__(self, spec, tokens, device="cpu"):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.torch, self.device = torch, torch.device(device)
        self.dtype = getattr(torch, spec["compute_dtype"])
        proc = AutoImageProcessor.from_pretrained(spec["model_id"], revision=spec["revision"])
        if (list(proc.image_mean) != [0.5] * 3 or list(proc.image_std) != [0.5] * 3
                or abs(proc.rescale_factor - 1 / 255) > 1e-9
                or (proc.size["height"], proc.size["width"]) != (IMAGE_SIZE, IMAGE_SIZE)):
            raise ValueError(f"{spec['model_id']}@{spec['revision']}: processor config differs from the "
                             "assumed 384px, mean 0.5, std 0.5, 1/255 rescale.")
        self.model = SiglipVisionModel.from_pretrained(spec["model_id"], revision=spec["revision"],
                                                       torch_dtype=self.dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.grid = cfg.image_size // cfg.patch_size
        self.hidden = cfg.hidden_size
        if [self.grid, self.grid] != list(spec["patch_grid"]) or self.hidden != spec["hidden_size"]:
            raise ValueError(f"SigLIP patch grid {self.grid} / hidden {self.hidden} differ from the release config.")
        self.stored_grid = tuple(spec["pooling"]["grid_rows_cols"])
        if tokens not in TOKEN_VIEWS:
            raise ValueError(f"no token view for {tokens} visual tokens")
        self.view_grid = TOKEN_VIEWS[tokens]

    def __call__(self, canvases):
        """384x384 RGB PIL images -> float32 numpy [n, tokens, hidden]."""
        torch = self.torch
        arr = np.stack([np.asarray(im, dtype=np.uint8) for im in canvases])
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float()
        x = ((x / 255.0) - 0.5) / 0.5
        with torch.no_grad():
            h = self.model(pixel_values=x.to(self.dtype)).last_hidden_state
            n = h.shape[0]
            h = h.float().reshape(n, self.grid, self.grid, self.hidden).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.adaptive_avg_pool2d(h, self.stored_grid)
            pooled = pooled.flatten(2).transpose(1, 2)
        stored = pooled.to(torch.float16).cpu().numpy()  # training saw fp16-stored embeddings
        return derive_view(stored, self.stored_grid, self.view_grid)


def layout_text(data):
    """pytesseract image_to_data dict -> text: words joined by spaces per line, lines by \\n,
    paragraphs/blocks by a blank line (as precompute.TesseractOCR.one)."""
    lines, current, last_line, last_par = [], [], None, None
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        par = (data["block_num"][i], data["par_num"][i])
        line = par + (data["line_num"][i],)
        if line != last_line and current:
            lines.append(" ".join(current))
            current = []
        if par != last_par and last_par is not None:
            lines.append("")
        current.append(text)
        last_line, last_par = line, par
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


class TesseractOCR:
    """Tesseract on the original image with the release config's language and options."""

    def __init__(self, spec):
        import warnings

        try:
            import pytesseract
        except ImportError as e:
            raise ImportError("OCR needs pytesseract and the tesseract binary: "
                              "pip install pytesseract; apt-get install tesseract-ocr") from e
        self.pytesseract, self.lang, self.config = pytesseract, spec["lang"], spec["config"]
        try:
            self.version = str(pytesseract.get_tesseract_version())
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError("tesseract binary not found; install tesseract-ocr") from e
        if self.version != spec["version"]:
            warnings.warn(f"tesseract {self.version}; the model was trained on OCR from tesseract "
                          f"{spec['version']}. Text may differ slightly.", RuntimeWarning, stacklevel=2)

    def __call__(self, img):
        pt = self.pytesseract
        return layout_text(pt.image_to_data(img, lang=self.lang, config=self.config, output_type=pt.Output.DICT))

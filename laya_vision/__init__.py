"""Laya-Vision: convaiinnovations/laya with an image input channel (SigLIP visual tokens).

    from laya_vision import VisionAgent
    agent = VisionAgent(release_dir="release")
    agent.predict(image="page.png", questions=agent.doc_type_question())

    from laya_vision import load_release
    release = load_release("release/")          # or a Hub repo id with revision=...
    release.model, release.tokenizer, release.temperatures
"""

from .model import LAYA_VERSION, make_vision_model_class
from .release import CONFIG_NAME, Release, ReleaseError, load_release
from .agent import VisionAgent
from .image import ImageError

__version__ = "0.1.0"
__all__ = ["VisionAgent", "ImageError", "CONFIG_NAME", "LAYA_VERSION", "Release", "ReleaseError", "load_release", "make_vision_model_class"]

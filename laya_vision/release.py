"""Load a Laya-Vision release folder (DECISIONS.md 2026-10-10, "P5 release decisions").

Layout (built by scripts/build_release.py):

    laya_vision_config.json        this file's CONFIG_NAME; sha256 of every file below
    calibration.json               calibrate.py output, unchanged
    weights/trainable.safetensors  projector + decision head (fp32)
    weights/base/                  the Gate-0 fine-tuned Laya checkpoint, unchanged:
        rl_agent_config.json, model.safetensors, encoder/config.json, tokenizer/

Nothing in the folder is written while loading (the tokenizer fix laya applies in
place runs on a temporary copy; see model.load_tokenizer).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .model import LAYA_VERSION, check_laya_version, load_base, load_trainable, sha256_file

CONFIG_NAME = "laya_vision_config.json"
FORMAT_VERSION = 1
BASE_DIR = "weights/base"
TRAINABLE_FILE = "weights/trainable.safetensors"
CALIBRATION_FILE = "calibration.json"
VIEWS = ("combined", "image", "text")


class ReleaseError(ValueError):
    """The folder is not a consistent Laya-Vision release."""


@dataclass
class Release:
    """A loaded release: the model and everything needed to build its inputs and read its outputs."""

    model: object                 # VisionDecisionModel, eval mode, fp32
    tokenizer: object
    config: dict                  # laya_vision_config.json
    base_cfg: dict                # the base's rl_agent_config.json (max_len set to config["max_len"])
    temperatures: dict            # applied per view: {"combined": T, "image": T, "text": 1.0}
    calibration: dict             # calibration.json
    path: Path
    fitted_temperatures: dict = field(default_factory=dict)

    def upstream_temperature(self, qtype="choice", n_options=16):
        """The temperature laya.Agent applies to this question from the base's rl_agent_config.json
        (laya 0.3.24 Agent._decode_answers, no language override). The SDK's image=None path uses
        it; the published text-view numbers use T = 1.0 (protocol §2)."""
        from laya.common import QTYPES, clamp_temperature, temp_bucket

        qt = QTYPES[qtype]
        by_options = {k: clamp_temperature(v) for k, v in self.base_cfg.get("temperature_by_options", {}).items()}
        per_type = [clamp_temperature(t) for t in self.base_cfg.get("temperature", [1.0, 1.0, 1.0])]
        return by_options.get(temp_bucket(qt, n_options), per_type[qt])


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve(path_or_repo, revision):
    path = Path(path_or_repo)
    if path.is_dir():
        if revision is not None:
            raise ReleaseError(f"{path} is a local folder; revision applies to Hub repos only.")
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision))


def load_release(path_or_repo, device="cpu", revision=None, verify=True):
    """Release folder (or Hub repo id at a revision) -> Release.

    Checks, before returning:
      - format version and the pinned laya version;
      - sha256 of every file listed in the config (verify=False skips the files that are not
        hashed anyway: the base weights and the trainable weights are always hashed);
      - the base and trainable weights are the ones the config and calibration.json name;
      - the model's projector matches the config's visual_dim and d_model;
      - the applied temperatures follow from calibration.json and the config's uncalibrated views.
    """
    root = _resolve(path_or_repo, revision)
    config_path = root / CONFIG_NAME
    if not config_path.exists():
        raise ReleaseError(f"{config_path} missing: not a Laya-Vision release folder.")
    config = _read_json(config_path)
    if config.get("format_version") != FORMAT_VERSION:
        raise ReleaseError(f"{config_path}: format_version {config.get('format_version')}, "
                           f"this loader reads {FORMAT_VERSION}.")
    if config["laya_version"] != LAYA_VERSION:
        raise ReleaseError(f"release built for laya {config['laya_version']}, this package pins {LAYA_VERSION}.")
    check_laya_version()

    files = config["files"]
    weights = (f"{BASE_DIR}/model.safetensors", TRAINABLE_FILE)  # hashed while loading, below
    for rel in weights + (CALIBRATION_FILE,):
        if rel not in files:
            raise ReleaseError(f"{config_path}: no sha256 recorded for {rel}.")
    for rel, expected in files.items():
        p = root / rel
        if not p.exists():
            raise ReleaseError(f"{p} missing.")
        if rel not in weights and (verify or rel == CALIBRATION_FILE) and sha256_file(p) != expected:
            raise ReleaseError(f"{p}: sha256 differs from {CONFIG_NAME}.")

    model, tok, base_cfg, base_sha = load_base(root / BASE_DIR, max_len=config["max_len"])
    if base_sha != files[f"{BASE_DIR}/model.safetensors"] or base_sha != config["base_model_sha256"]:
        raise ReleaseError(f"{root / BASE_DIR / 'model.safetensors'}: sha256 differs from {CONFIG_NAME}.")
    trainable_path = root / TRAINABLE_FILE
    trainable_sha = sha256_file(trainable_path)
    if trainable_sha != files[TRAINABLE_FILE]:
        raise ReleaseError(f"{trainable_path}: sha256 differs from {CONFIG_NAME}.")
    load_trainable(model, trainable_path)

    proj = model.projector.net
    if (proj[0].in_features, proj[-1].out_features) != (config["visual_dim"], config["d_model"]):
        raise ReleaseError(f"projector is {proj[0].in_features} -> {proj[-1].out_features}, config says "
                           f"{config['visual_dim']} -> {config['d_model']}.")

    calibration = _read_json(root / CALIBRATION_FILE)
    ckpt = calibration["checkpoint"]
    if ckpt["trainable_sha256"] != trainable_sha or ckpt["visual_tokens"] != config["visual_tokens"]:
        raise ReleaseError(f"{CALIBRATION_FILE} was fitted on another checkpoint or token count.")
    fitted = {v: float(t) for v, t in calibration["temperatures"].items()}
    uncalibrated = set(config["uncalibrated_views"])
    applied = {v: 1.0 if v in uncalibrated else fitted[v] for v in VIEWS}
    if applied != {v: float(t) for v, t in config["temperatures_applied"].items()}:
        raise ReleaseError(f"{CONFIG_NAME} temperatures_applied {config['temperatures_applied']} do not follow "
                           f"from {CALIBRATION_FILE} {fitted} with uncalibrated views {sorted(uncalibrated)}.")

    model.to(device).eval()
    return Release(model=model, tokenizer=tok, config=config, base_cfg=base_cfg, temperatures=applied,
                   calibration=calibration, path=root, fitted_temperatures=fitted)

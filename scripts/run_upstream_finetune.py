"""CUDA-enabling wrapper around upstream's single-process fine-tuning script.

notebooks/laya_finetune_typed_decisions_mps.py restricts --device to
auto/mps/cpu, and "auto" falls back to CPU off Apple Silicon. That file is
kept byte-identical to upstream; this wrapper loads it as a module, adds
"cuda" to the --device choices, and calls its main(). All other flags and
behaviour are upstream's.

    python scripts/run_upstream_finetune.py --device cuda --epochs 4 \
        --micro-batch 8 --grad-accum 8 --model-dir ... --items ... --output-dir ...
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

UPSTREAM = Path(__file__).resolve().parent.parent / "notebooks" / "laya_finetune_typed_decisions_mps.py"


class _CudaArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if "--device" in args and "cuda" not in kwargs.get("choices", ["cuda"]):
            kwargs["choices"] = list(kwargs["choices"]) + ["cuda"]
        return super().add_argument(*args, **kwargs)


def load_upstream():
    spec = importlib.util.spec_from_file_location("laya_finetune_upstream", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    upstream = load_upstream()

    # The upstream parser is built inside main(), so swap the argparse it sees.
    shim = types.SimpleNamespace(**vars(argparse))
    shim.ArgumentParser = _CudaArgumentParser
    upstream.argparse = shim

    upstream_choose_device = upstream.choose_device

    def choose_device(requested):
        if requested == "cuda" and not torch.cuda.is_available():
            sys.exit("--device cuda requested but no CUDA device is available.")
        return upstream_choose_device(requested)

    upstream.choose_device = choose_device
    upstream.main()


if __name__ == "__main__":
    main()

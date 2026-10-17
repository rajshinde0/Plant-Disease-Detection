"""Report the training environment and recommend settings for this machine.

Run this first on any new machine - especially the CUDA laptop. The most
common and most silent failure when moving this project is installing the
default PyTorch wheel, which is CPU-only: training then runs 20x slower with
no error message at all. This script catches exactly that.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CUDA_INSTALL_HINT = (
    "pip install torch torchvision --index-url "
    "https://download.pytorch.org/whl/cu124"
)


def main() -> int:
    print("=" * 68)
    print("Environment report")
    print("=" * 68)
    print("  platform      : %s %s" % (platform.system(), platform.release()))
    print("  python        : %s" % platform.python_version())
    print("  cpu cores     : %s logical" % (os.cpu_count() or "?"))

    try:
        import torch
    except ImportError:
        print("  torch         : NOT INSTALLED")
        print()
        print("  Install it with:  pip install -r requirements.txt")
        return 2

    print("  torch         : %s" % torch.__version__)
    print("  torchvision   : ", end="")
    try:
        import torchvision
        print(torchvision.__version__)
    except ImportError:
        print("NOT INSTALLED")

    cuda = torch.cuda.is_available()
    cpu_only_wheel = "+cpu" in torch.__version__
    print("  CUDA available: %s" % cuda)

    vram_gb = None
    if cuda:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print("  GPU           : %s" % props.name)
        print("  VRAM          : %.1f GB" % vram_gb)
        print("  device count  : %d" % torch.cuda.device_count())

    print()
    print("-" * 68)
    print("RECOMMENDATION")
    print("-" * 68)

    if cuda:
        if vram_gb and vram_gb < 5:
            batch = 32
        elif vram_gb and vram_gb < 9:
            batch = 64
        else:
            batch = 128
        print("  Train at the paper's resolution with mixed precision:")
        print()
        print("    python -m src.train --img-size 256 --epochs 5 "
              "--batch-size %d --amp" % batch)
        print()
        print("  Expect roughly 30 minutes and ~99%% validation accuracy.")
        print("  If you hit a CUDA out-of-memory error, halve --batch-size.")
    else:
        if cpu_only_wheel:
            print("  torch is the CPU-ONLY wheel (version string ends in +cpu).")
            print("  If this machine has an NVIDIA GPU, that GPU is being ignored.")
            print("  Install the CUDA build instead:")
            print()
            print("    " + CUDA_INSTALL_HINT)
            print()
            print("  (Check your driver's CUDA version first with: nvidia-smi)")
            print()
        print("  No CUDA device, so training runs on the CPU. Measured recipes")
        print("  for a 4-core/8-thread laptop:")
        print()
        print("    %-9s %-11s  %s" % ("time", "accuracy", "command"))
        print("    " + "-" * 62)
        for flags, dur, acc in (
            ("--img-size 64 --subset 0.1 --epochs 1", "~15 min", "smoke test"),
            ("--img-size 64 --epochs 4", "~2 h", "~95-97%"),
            ("--img-size 128 --subset 0.4 --epochs 3", "~2.5 h", "~97-98.5%"),
            ("--img-size 128 --epochs 3", "~5.5 h", "~98-99%"),
        ):
            print("    %-9s %-11s  python -m src.train %s" % (dur, acc, flags))
        print()
        print("  Avoid --img-size 256 on CPU: ~7 h per epoch.")
        print("  Note: integrated Intel graphics (Iris/UHD) cannot be used for")
        print("  training - intel-extension-for-pytorch targets Arc and newer,")
        print("  and torch-directml does not support current torch/python.")

    print()
    print("-" * 68)
    print("DATASET")
    print("-" * 68)
    for label, path, expected in (
        ("train", config.TRAIN_DIR, config.EXPECTED_TRAIN),
        ("valid", config.VALID_DIR, config.EXPECTED_VALID),
        ("test", config.TEST_DIR, config.EXPECTED_TEST),
    ):
        if Path(path).is_dir():
            n = sum(1 for _ in Path(path).rglob("*") if _.is_file())
            flag = "OK" if n == expected else "expected %d" % expected
            print("  %-6s %6d images  (%s)" % (label, n, flag))
        else:
            print("  %-6s MISSING -> run: python scripts/prepare_dataset.py" % label)

    print()
    print("MODEL")
    print("  %s" % ("found: " + str(config.MODEL_FILE)
                     if config.MODEL_FILE.is_file() else
                     "not trained yet -> run src.train"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

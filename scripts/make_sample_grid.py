"""Produce the report's dataset figures.

    reports/figures/dataset_samples.png      one leaf per class, labelled
    reports/figures/class_distribution.png   images per class, train vs valid

Run after scripts/prepare_dataset.py.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.remedies import display_name  # noqa: E402
from src.viz import (SERIES, TEXT_SECONDARY, apply_style, save,  # noqa: E402
                     subtitle)


def sample_grid(train_dir: Path, cols: int = 8, seed: int = config.SEED):
    import random

    import matplotlib.pyplot as plt
    from PIL import Image

    classes = sorted(p for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        print("ERROR: no class directories in %s" % train_dir, file=sys.stderr)
        return None

    rng = random.Random(seed)
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.85, rows * 2.15))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, cls_dir in zip(axes, classes):
        files = [f for f in cls_dir.iterdir() if f.is_file()]
        ax.imshow(Image.open(rng.choice(files)).convert("RGB"))
        label = display_name(cls_dir.name)
        # Wrap the label so long disease names stay readable under the tile.
        if len(label) > 26:
            parts = label.split(" - ")
            label = parts[0] + "\n" + " - ".join(parts[1:]) if len(parts) > 1 else label
        ax.set_title(label, fontsize=6.6, color=TEXT_SECONDARY, pad=3)
        ax.axis("off")

    for ax in axes[len(classes):]:
        ax.axis("off")

    fig.suptitle("One sample per class - 38 plant-disease classes across 14 crops",
                 x=0.5, y=0.998, fontsize=12.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return save(fig, "dataset_samples.png")


def class_distribution(train_dir: Path, valid_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    def counts(d: Path):
        c = collections.Counter()
        if d.is_dir():
            for cls in sorted(p for p in d.iterdir() if p.is_dir()):
                c[cls.name] = sum(1 for f in cls.iterdir() if f.is_file())
        return c

    tr, va = counts(train_dir), counts(valid_dir)
    if not tr:
        return None

    names = sorted(tr, key=lambda k: -tr[k])
    labels = [display_name(n) for n in names]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 11.5))
    # Two series (train, valid), so a legend is present; slots 1 and 2 in fixed
    # order, validated as a CVD-safe pair.
    ax.barh(y - 0.2, [tr[n] for n in names], height=0.38,
            color=SERIES[0], label="Train")
    ax.barh(y + 0.2, [va.get(n, 0) for n in names], height=0.38,
            color=SERIES[1], label="Validation")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_title("Images per class", loc="left")
    subtitle(ax, "%s train / %s validation - classes are near-balanced, "
                 "so accuracy is not inflated by a dominant class"
                 % (format(sum(tr.values()), ","), format(sum(va.values()), ",")))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return save(fig, "class_distribution.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate dataset figures.")
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args(argv)

    if not config.TRAIN_DIR.is_dir():
        print("ERROR: dataset not found. Run: python scripts/prepare_dataset.py",
              file=sys.stderr)
        return 2

    apply_style()
    print("Generating dataset figures...")
    sample_grid(config.TRAIN_DIR, cols=args.cols)
    class_distribution(config.TRAIN_DIR, config.VALID_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Regenerate the paper's ResNet6-vs-ResNet9 comparison from actual runs.

The paper tabulates accuracy, inference time, training time and FLOPs for the
two architectures. This script measures those under an equal training budget so
the comparison is fair, and writes both a JSON table and a figure.

Defaults are sized for a CPU laptop. On a GPU, raise --img-size and --subset.

    python -m src.compare_models
    python -m src.compare_models --img-size 128 --subset 0.3 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import data as datamod
from src.engine import evaluate, fit_one_cycle
from src.model import build_model, count_parameters
from src.viz import SERIES, TEXT_SECONDARY, apply_style, save


@torch.no_grad()
def measure_inference_ms(model, img_size, device, runs: int = 30) -> float:
    """Mean single-image latency - what matters for the web app."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(5):            # warm up
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / runs


def plot_comparison(rows):
    import matplotlib.pyplot as plt

    names = [r["model"].upper() for r in rows]
    panels = [
        ("Validation accuracy", [r["val_acc"] * 100 for r in rows], "%", "%.2f"),
        ("Training time", [r["train_seconds"] / 60 for r in rows], " min", "%.1f"),
        ("Inference latency", [r["inference_ms"] for r in rows], " ms", "%.1f"),
        ("Parameters", [r["parameters"] / 1e6 for r in rows], "M", "%.2f"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.4))
    for ax, (title, vals, unit, fmt) in zip(axes, panels):
        # One series across two categories, so a single hue and no legend; the
        # x-axis labels carry the identity.
        ax.bar(names, vals, color=SERIES[0], width=0.55)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.grid(axis="x", visible=False)
        headroom = max(vals) * 1.18 if max(vals) > 0 else 1
        ax.set_ylim(0, headroom)
        # Direct labels: only 2 bars, so every one is signal rather than noise.
        for i, v in enumerate(vals):
            ax.text(i, v + headroom * 0.03, (fmt % v) + unit, ha="center",
                    fontsize=10, color=TEXT_SECONDARY, fontweight="semibold")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    fig.suptitle("ResNet6 vs ResNet9 under an equal training budget",
                 x=0.005, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, "model_comparison.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare ResNet6 and ResNet9.")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--subset", type=float, default=0.1)
    ap.add_argument("--max-lr", type=float, default=config.MAX_LR)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args(argv)

    device = datamod.get_default_device()
    torch.manual_seed(config.SEED)

    print("=" * 68)
    print("ResNet6 vs ResNet9")
    print("=" * 68)
    print("  equal budget: %dpx, %d epochs, subset %g, batch %d, device %s"
          % (args.img_size, args.epochs, args.subset, args.batch_size, device.type))
    print()

    # Build the loaders once so both models see identical data.
    train_dl, valid_dl, class_names = datamod.build_loaders(
        img_size=args.img_size, batch_size=args.batch_size, subset=args.subset,
        num_workers=args.num_workers, device=device)
    print("  train %d images / valid %d images / %d classes"
          % (len(train_dl.dl.dataset), len(valid_dl.dl.dataset), len(class_names)))

    rows = []
    for name in ("resnet6", "resnet9"):
        print()
        print("-" * 68)
        print("Training %s" % name.upper())
        print("-" * 68)
        torch.manual_seed(config.SEED)
        model = datamod.to_device(build_model(name, len(class_names)), device)

        t0 = time.time()
        history = fit_one_cycle(args.epochs, model, train_dl, valid_dl,
                                max_lr=args.max_lr,
                                weight_decay=config.WEIGHT_DECAY,
                                grad_clip=config.GRAD_CLIP,
                                use_amp=args.amp, device=device)
        train_secs = time.time() - t0

        final = evaluate(model, valid_dl)
        latency = measure_inference_ms(model, args.img_size, device)

        rows.append({
            "model": name,
            "parameters": count_parameters(model),
            "val_acc": final["val_acc"],
            "val_loss": final["val_loss"],
            "train_seconds": train_secs,
            "inference_ms": latency,
            "epochs": args.epochs,
            "img_size": args.img_size,
            "subset": args.subset,
            "history": history,
        })

    print()
    print("=" * 68)
    print("RESULTS")
    print("=" * 68)
    print("  %-10s %10s %10s %12s %12s" %
          ("model", "params", "val_acc", "train(min)", "infer(ms)"))
    print("  " + "-" * 60)
    for r in rows:
        print("  %-10s %9.2fM %9.2f%% %11.1f %11.1f"
              % (r["model"].upper(), r["parameters"] / 1e6, r["val_acc"] * 100,
                 r["train_seconds"] / 60, r["inference_ms"]))

    best = max(rows, key=lambda r: r["val_acc"])
    gap = (best["val_acc"] - min(r["val_acc"] for r in rows)) * 100
    print()
    print("  %s wins by %.2f percentage points of accuracy."
          % (best["model"].upper(), gap))
    print("  Note: this is a short, equal-budget comparison at %dpx on %g of the"
          % (args.img_size, args.subset))
    print("  data, so absolute numbers are lower than a full training run.")

    out = config.REPORTS_DIR / "model_comparison.json"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"device": device.type, "args": vars(args), "results": rows},
                  fh, indent=2, default=str)
    print()
    print("  wrote %s" % out.relative_to(config.ROOT))

    apply_style()
    plot_comparison(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

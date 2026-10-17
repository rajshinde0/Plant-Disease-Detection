"""Learning-rate sweep - tests the paper's claim that lr 0.1 is optimal.

The paper states "the highest accuracy is achieved at a learning rate of 0.1",
while the canonical ResNet9 implementation this architecture comes from uses
max_lr 0.01. Those cannot both be right, so this script settles it by running
the same short budget at each candidate rate and plotting the result.

    python -m src.lr_sweep
    python -m src.lr_sweep --rates 0.001,0.01,0.1 --epochs 2
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
from src.model import build_model
from src.viz import GOOD, SERIES, apply_style, save, subtitle

DEFAULT_RATES = "0.001,0.005,0.01,0.05,0.1"


def plot_sweep(rows, epochs, img_size, subset):
    import matplotlib.pyplot as plt

    rates = [r["max_lr"] for r in rows]
    accs = [r["val_acc"] * 100 for r in rows]
    best_i = max(range(len(rows)), key=lambda i: accs[i])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    # Single series, so no legend - the title names it.
    ax.plot(rates, accs, color=SERIES[0], marker="o", zorder=3)
    # The winning point gets a reserved status colour plus a label, so the
    # highlight never depends on colour alone.
    ax.scatter([rates[best_i]], [accs[best_i]], s=180, color=GOOD, zorder=4,
               label="best")
    ax.annotate("best: lr=%g\n%.2f%%" % (rates[best_i], accs[best_i]),
                (rates[best_i], accs[best_i]), textcoords="offset points",
                xytext=(0, -40), ha="center", fontsize=10, color=GOOD,
                fontweight="semibold")

    ax.set_xscale("log")
    ax.set_xticks(rates)
    ax.set_xticklabels([("%g" % r) for r in rates])
    ax.set_xlabel("One-Cycle max learning rate (log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy vs max learning rate", loc="left")
    subtitle(ax, "%d epochs at %dpx on %g of the training data"
                 % (epochs, img_size, subset))

    span = max(accs) - min(accs)
    ax.set_ylim(min(accs) - max(span * 0.35, 2), max(accs) + max(span * 0.25, 2))
    return save(fig, "lr_sweep.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sweep the One-Cycle max learning rate.")
    ap.add_argument("--rates", default=DEFAULT_RATES,
                    help="comma-separated learning rates (default: %s)" % DEFAULT_RATES)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--subset", type=float, default=0.08)
    ap.add_argument("--model", default="resnet9", choices=["resnet9", "resnet6"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args(argv)

    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    device = datamod.get_default_device()

    print("=" * 68)
    print("Learning-rate sweep")
    print("=" * 68)
    print("  rates : %s" % ", ".join("%g" % r for r in rates))
    print("  budget: %dpx, %d epoch(s), subset %g, device %s"
          % (args.img_size, args.epochs, args.subset, device.type))
    print()

    train_dl, valid_dl, class_names = datamod.build_loaders(
        img_size=args.img_size, batch_size=args.batch_size, subset=args.subset,
        num_workers=args.num_workers, device=device)
    print("  train %d images / valid %d images"
          % (len(train_dl.dl.dataset), len(valid_dl.dl.dataset)))

    rows = []
    for lr in rates:
        print()
        print("-" * 68)
        print("max_lr = %g" % lr)
        print("-" * 68)
        torch.manual_seed(config.SEED)      # same init for every rate
        model = datamod.to_device(build_model(args.model, len(class_names)), device)
        t0 = time.time()
        history = fit_one_cycle(args.epochs, model, train_dl, valid_dl, max_lr=lr,
                                weight_decay=config.WEIGHT_DECAY,
                                grad_clip=config.GRAD_CLIP,
                                use_amp=args.amp, device=device)
        final = evaluate(model, valid_dl)
        rows.append({
            "max_lr": lr,
            "val_acc": final["val_acc"],
            "val_loss": final["val_loss"],
            "seconds": time.time() - t0,
            "history": history,
        })

    rows.sort(key=lambda r: r["max_lr"])
    best = max(rows, key=lambda r: r["val_acc"])

    print()
    print("=" * 68)
    print("RESULTS")
    print("=" * 68)
    print("  %10s %12s %12s" % ("max_lr", "val_acc", "val_loss"))
    print("  " + "-" * 36)
    for r in rows:
        mark = "  <- best" if r is best else ""
        print("  %10g %11.2f%% %12.4f%s"
              % (r["max_lr"], r["val_acc"] * 100, r["val_loss"], mark))

    print()
    print("  Best rate measured: %g" % best["max_lr"])
    paper_claim = 0.1
    if abs(best["max_lr"] - paper_claim) < 1e-9:
        print("  This supports the paper's claim that 0.1 is optimal.")
    else:
        in_sweep = [r for r in rows if abs(r["max_lr"] - paper_claim) < 1e-9]
        if in_sweep:
            print("  The paper claims 0.1 is optimal, but 0.1 scored %.2f%% here"
                  % (in_sweep[0]["val_acc"] * 100))
            print("  versus %.2f%% at %g - the paper's claim is not reproduced."
                  % (best["val_acc"] * 100, best["max_lr"]))
    print("  (Short-budget sweep; confirm with a full run before citing.)")

    out = config.REPORTS_DIR / "lr_sweep.json"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"device": device.type, "args": vars(args),
                   "best_max_lr": best["max_lr"], "results": rows}, fh, indent=2, default=str)
    print()
    print("  wrote %s" % out.relative_to(config.ROOT))

    apply_style()
    plot_sweep(rows, args.epochs, args.img_size, args.subset)
    return 0


if __name__ == "__main__":
    sys.exit(main())

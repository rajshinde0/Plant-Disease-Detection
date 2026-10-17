"""Train ResNet9 on the PlantVillage dataset.

Measured throughput on an Intel i7-1065G7 (4 cores / 8 threads, 15 W), so you
can pick a recipe that fits the time you have:

    recipe                                        time      expected accuracy
    --img-size 64  --subset 0.1 --epochs 1        ~15 min   pipeline proof
    --img-size 64               --epochs 4        ~2 h      ~95-97%
    --img-size 128 --subset 0.4 --epochs 3        ~2.5 h    ~97-98.5%
    --img-size 128              --epochs 3        ~5.5 h    ~98-99%

On a CUDA machine, use the paper's resolution and add mixed precision:

    --img-size 256 --epochs 5 --batch-size 64 --amp          ~30 min, ~99%

ResNet9 converges by epoch 2-3 on this dataset, so --epochs 5 is ample; the
paper's stated 20 is only needed to reproduce its exact protocol.
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
from src.engine import fit_one_cycle
from src.model import build_model, count_parameters


def save_checkpoint(path: Path, model, model_name, class_names, img_size,
                    val_acc, history, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model_name,
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "img_size": img_size,
        "val_acc": val_acc,
        "num_classes": len(class_names),
        "history": history,
        "args": vars(args) if args is not None else {},
    }, path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Train ResNet9 for plant disease detection.")
    ap.add_argument("--model", default="resnet9", choices=["resnet9", "resnet6"])
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--max-lr", type=float, default=config.MAX_LR)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--grad-clip", type=float, default=config.GRAD_CLIP)
    ap.add_argument("--subset", type=float, default=1.0,
                    help="fraction of each class to use (stratified). Default 1.0")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable training-time augmentation")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (CUDA only; ignored on CPU)")
    ap.add_argument("--out", type=Path, default=config.MODEL_FILE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = datamod.get_default_device()
    workers = (args.num_workers if args.num_workers is not None
               else datamod.default_workers(device))

    print("=" * 68)
    print("Training %s for plant disease detection" % args.model)
    print("=" * 68)
    print("  device     : %s%s" % (device.type,
          ("  (" + torch.cuda.get_device_name(0) + ")") if device.type == "cuda" else ""))
    print("  image size : %d" % args.img_size)
    print("  batch size : %d" % args.batch_size)
    print("  epochs     : %d" % args.epochs)
    print("  max_lr     : %g   weight_decay: %g   grad_clip: %g"
          % (args.max_lr, args.weight_decay, args.grad_clip))
    print("  subset     : %g" % args.subset)
    print("  augment    : %s" % (not args.no_augment))
    print("  workers    : %d" % workers)
    if args.amp and device.type != "cuda":
        print("  NOTE: --amp ignored, mixed precision needs CUDA")
    print()

    train_dl, valid_dl, class_names = datamod.build_loaders(
        img_size=args.img_size, batch_size=args.batch_size, subset=args.subset,
        augment=not args.no_augment, num_workers=workers, device=device)

    n_train = len(train_dl.dl.dataset)
    n_valid = len(valid_dl.dl.dataset)
    print("  classes    : %d" % len(class_names))
    print("  train      : %d images (%d batches)" % (n_train, len(train_dl)))
    print("  valid      : %d images (%d batches)" % (n_valid, len(valid_dl)))

    model = datamod.to_device(build_model(args.model, len(class_names)), device)
    print("  parameters : %.2fM" % (count_parameters(model) / 1e6))
    print()

    datamod.save_class_names(class_names)

    best = {"val_acc": -1.0}
    running = []

    def write_history(history, elapsed=None):
        with open(config.HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump({"history": history, "train_seconds": elapsed,
                       "args": vars(args) | {"out": str(args.out)}}, fh,
                      indent=2, default=str)

    def on_epoch_end(_epoch, result, mdl):
        # Persist after EVERY epoch, not just at the end. A long run that is
        # interrupted then still leaves a usable model and real curves behind
        # rather than nothing.
        running.append(result)
        if result["val_acc"] > best["val_acc"]:
            best["val_acc"] = result["val_acc"]
            save_checkpoint(args.out, mdl, args.model, class_names,
                            args.img_size, result["val_acc"], list(running), args)
            print("    -> new best (val_acc=%.4f), checkpoint saved" % result["val_acc"])
        write_history(list(running))

    t0 = time.time()
    history = fit_one_cycle(
        args.epochs, model, train_dl, valid_dl,
        max_lr=args.max_lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, use_amp=args.amp, device=device,
        on_epoch_end=on_epoch_end)
    elapsed = time.time() - t0

    # Rewrite the checkpoint so it carries the full history too.
    ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    ckpt["history"] = history
    ckpt["train_seconds"] = elapsed
    torch.save(ckpt, args.out)

    write_history(history, elapsed)

    print()
    print("=" * 68)
    print("  best val_acc : %.4f  (%.2f%%)" % (best["val_acc"], best["val_acc"] * 100))
    print("  total time   : %.1f min" % (elapsed / 60))
    print("  model        : %s" % args.out)
    print("  classes      : %s" % config.CLASS_NAMES_FILE)
    print("  history      : %s" % config.HISTORY_FILE)
    print("=" * 68)
    print()
    print("Next: python -m src.evaluate")
    return 0


if __name__ == "__main__":
    # Required on Windows: DataLoader workers use spawn, which re-imports this
    # module in each child process.
    sys.exit(main())

"""Evaluate the trained model and regenerate the report's tables and figures.

Every number here is measured, never copied from the write-up.

A note on the paper's precision 0.94 / recall 0.96 / F1 0.92: those are NOT
self-contradictory, despite looking it. Macro-F1 is the mean of the per-class
F1 scores - it is not the harmonic mean of macro-precision and macro-recall -
so it can legitimately sit below both. A class with precision 1.00 and recall
0.02 scores F1 0.04 and drags the macro average down hard. This script's own
output demonstrates the effect. What the paper does omit is WHICH averaging it
used, so this script reports macro and weighted side by side and leaves the
reader in no doubt.

Outputs:
    reports/metrics.json
    reports/figures/confusion_matrix.png
    reports/figures/per_class_f1.png
    reports/figures/accuracy_vs_epochs.png
    reports/figures/loss_vs_epochs.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import data as datamod
from src.engine import predict_logits
from src.inference import (ModelNotTrained, _json_safe, load_image, load_model,
                           predict_image)
from src.remedies import display_name
from src.viz import (BLUE_SEQUENTIAL, SERIES, TEXT_MUTED, TEXT_SECONDARY,
                     apply_style, save)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, class_names, normalize: bool = True):
    import matplotlib.pyplot as plt

    mat = cm.astype(float)
    if normalize:
        rows = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    # One hue, light -> dark: this encodes magnitude, which has order.
    im = ax.imshow(mat, cmap=BLUE_SEQUENTIAL, vmin=0, vmax=1 if normalize else None,
                   interpolation="nearest", aspect="equal")

    labels = [display_name(c) for c in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix" + (" (row-normalised recall)" if normalize else ""),
                 loc="left")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    # A 2px surface gap between cells, so adjacent fills stay distinguishable.
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, color=TEXT_SECONDARY)
    cb.set_label("Share of true class" if normalize else "Images",
                 fontsize=9.5, color=TEXT_SECONDARY)
    return save(fig, "confusion_matrix.png")


def plot_per_class_f1(class_names, f1_scores):
    import matplotlib.pyplot as plt

    order = np.argsort(f1_scores)
    names = [display_name(class_names[i]) for i in order]
    vals = np.asarray(f1_scores)[order]

    fig, ax = plt.subplots(figsize=(10, 11))
    # Single series: no legend needed, the title names it.
    ax.barh(range(len(vals)), vals, color=SERIES[0], height=0.74)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-class F1 score", loc="left")
    ax.grid(axis="y", visible=False)

    # Selective direct labels: only the weakest classes, which are the ones a
    # reader needs the exact number for.
    for i, v in enumerate(vals):
        if i < 6:
            ax.text(v + 0.012, i, "%.3f" % v, va="center", fontsize=7.5,
                    color=TEXT_SECONDARY)
    return save(fig, "per_class_f1.png")


def plot_history(history):
    import matplotlib.pyplot as plt

    if not history:
        return []
    epochs = [h["epoch"] for h in history]
    written = []

    # Accuracy: one series, so no legend box.
    fig, ax = plt.subplots(figsize=(8, 5))
    accs = [h["val_acc"] * 100 for h in history]
    ax.plot(epochs, accs, color=SERIES[0], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy vs epochs", loc="left")
    ax.set_xticks(epochs)
    # Direct-label the final value only - a number on every point is noise.
    ax.annotate("%.2f%%" % accs[-1], (epochs[-1], accs[-1]),
                textcoords="offset points", xytext=(-8, 10),
                fontsize=10, color=TEXT_SECONDARY, ha="right")
    written.append(save(fig, "accuracy_vs_epochs.png"))

    # Loss: two series, so a legend is present.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], color=SERIES[0],
            marker="o", label="Train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], color=SERIES[1],
            marker="s", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss vs epochs", loc="left")
    ax.set_xticks(epochs)
    ax.legend(loc="upper right")
    written.append(save(fig, "loss_vs_epochs.png"))
    return written


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_test_images(model, class_names, img_size, test_dir: Path):
    """Predict the 33 held-out evaluation images.

    They are unlabelled, but their filenames name the expected disease (e.g.
    AppleCedarRust1.JPG), so we report a loose filename-based agreement check
    alongside each prediction rather than inventing an accuracy figure.
    """
    if not test_dir.is_dir():
        return []
    rows = []
    for path in sorted(p for p in test_dir.iterdir() if p.is_file()):
        res = predict_image(model, load_image(path), class_names, img_size, topk=3)
        stem = path.stem.lower().replace("_", "").replace("-", "")
        crop = res["class_name"].split("___")[0].split(",")[0].lower()
        rows.append({
            "file": path.name,
            "predicted": res["class_name"],
            "display": res["display"],
            "confidence": round(res["confidence"], 4),
            "crop_matches_filename": crop[:5] in stem,
            "low_confidence": res["low_confidence"],
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the trained model.")
    ap.add_argument("--model", type=Path, default=config.MODEL_FILE)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit", type=float, default=1.0,
                    help="evaluate on this fraction of the validation set")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)

    try:
        model, class_names, img_size = load_model(args.model)
    except ModelNotTrained as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    device = datamod.get_default_device()
    print("=" * 68)
    print("Evaluating %s" % args.model.name)
    print("=" * 68)
    print("  device   : %s" % device.type)
    print("  img size : %d (from checkpoint)" % img_size)
    print("  classes  : %d" % len(class_names))

    valid_dl, valid_ds = datamod.build_valid_loader(
        img_size=img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device)

    if args.limit < 1.0:
        from torch.utils.data import DataLoader
        sub = datamod.stratified_subset(valid_ds, args.limit)
        dl = DataLoader(sub, args.batch_size * 2, shuffle=False, num_workers=0)
        valid_dl = datamod.DeviceDataLoader(dl, device)
        print("  NOTE: limited to %.0f%% of the validation set" % (args.limit * 100))

    n_images = len(valid_dl.dl.dataset)
    print("  images   : %d" % n_images)
    print()

    t0 = time.time()
    logits, labels = predict_logits(model, valid_dl)
    infer_secs = time.time() - t0

    probs = F.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    y_true = labels.numpy()
    y_pred = preds.numpy()

    accuracy = float((y_pred == y_true).mean())
    loss = float(F.cross_entropy(logits, labels).item())

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)),
        zero_division=0)

    # Top-3 accuracy: useful because visually similar diseases are the known
    # weak point, and a shortlist is still actionable for a farmer.
    top3 = float((torch.topk(probs, min(3, len(class_names)), dim=1).indices
                  == labels.unsqueeze(1)).any(dim=1).float().mean())

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Most-confused ordered pairs, off the diagonal.
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                    "share_of_true_class": round(float(cm[i, j] / max(1, cm[i].sum())), 4),
                })
    confusions.sort(key=lambda d: -d["count"])

    print("-" * 68)
    print("OVERALL")
    print("-" * 68)
    print("  accuracy            : %.4f  (%.2f%%)" % (accuracy, accuracy * 100))
    print("  top-3 accuracy      : %.4f  (%.2f%%)" % (top3, top3 * 100))
    print("  cross-entropy loss  : %.4f" % loss)
    print("  precision (macro)   : %.4f     (weighted: %.4f)" % (p_macro, p_w))
    print("  recall    (macro)   : %.4f     (weighted: %.4f)" % (r_macro, r_w))
    print("  F1        (macro)   : %.4f     (weighted: %.4f)" % (f_macro, f_w))
    print("  inference           : %.1f s total, %.1f ms/image"
          % (infer_secs, 1000 * infer_secs / max(1, n_images)))

    print()
    print("-" * 68)
    print("WEAKEST CLASSES (by F1)")
    print("-" * 68)
    worst = np.argsort(f_cls)[:8]
    print("  %-50s %6s %6s %6s %6s" % ("class", "prec", "rec", "F1", "n"))
    for i in worst:
        print("  %-50s %6.3f %6.3f %6.3f %6d"
              % (display_name(class_names[i])[:50], p_cls[i], r_cls[i],
                 f_cls[i], support[i]))

    print()
    print("-" * 68)
    print("MOST CONFUSED PAIRS")
    print("-" * 68)
    if confusions:
        for c in confusions[:10]:
            print("  %-36s -> %-36s %4d (%.1f%%)"
                  % (display_name(c["true"])[:36], display_name(c["predicted"])[:36],
                     c["count"], c["share_of_true_class"] * 100))
    else:
        print("  none - no misclassifications at all")

    # The paper claims mildew/downy-mildew confusion specifically; check it.
    mildew = [c for c in confusions
              if "mildew" in c["true"].lower() and "mildew" in c["predicted"].lower()]
    print()
    print("  mildew-vs-mildew confusions (the paper's claim): %s"
          % ("%d pair(s), %d images" % (len(mildew), sum(c["count"] for c in mildew))
             if mildew else "none observed"))

    # The 33 held-out evaluation images.
    test_rows = evaluate_test_images(model, class_names, img_size, config.TEST_DIR)
    if test_rows:
        agree = sum(1 for r in test_rows if r["crop_matches_filename"])
        print()
        print("-" * 68)
        print("HELD-OUT TEST IMAGES (%d)" % len(test_rows))
        print("-" * 68)
        print("  crop matches filename: %d/%d" % (agree, len(test_rows)))
        print("  mean confidence      : %.2f%%"
              % (100 * sum(r["confidence"] for r in test_rows) / len(test_rows)))

    # --- persist ---------------------------------------------------------
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    history = ckpt.get("history", []) or []

    metrics = {
        "model": ckpt.get("model", "resnet9"),
        "img_size": img_size,
        "num_classes": len(class_names),
        "validation_images": int(n_images),
        "accuracy": round(accuracy, 6),
        "top3_accuracy": round(top3, 6),
        "loss": round(loss, 6),
        "precision_macro": round(float(p_macro), 6),
        "recall_macro": round(float(r_macro), 6),
        "f1_macro": round(float(f_macro), 6),
        "precision_weighted": round(float(p_w), 6),
        "recall_weighted": round(float(r_w), 6),
        "f1_weighted": round(float(f_w), 6),
        "inference_ms_per_image": round(1000 * infer_secs / max(1, n_images), 3),
        "train_seconds": ckpt.get("train_seconds"),
        "epochs_trained": len(history),
        "training_args": _json_safe(ckpt.get("args", {})),
        "per_class": [
            {
                "class_name": class_names[i],
                "display": display_name(class_names[i]),
                "precision": round(float(p_cls[i]), 6),
                "recall": round(float(r_cls[i]), 6),
                "f1": round(float(f_cls[i]), 6),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        ],
        "most_confused_pairs": confusions[:30],
        "mildew_confusions": mildew,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "held_out_test": test_rows,
        "history": history,
        "note": ("All figures measured on the validation set by src/evaluate.py. "
                 "Macro-F1 is the mean of per-class F1 scores, not the harmonic "
                 "mean of macro-precision and macro-recall, so it can sit below "
                 "both - as it does here. The source paper does not state which "
                 "averaging it used; macro and weighted are both reported above."),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    text_report = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=[display_name(c) for c in class_names], zero_division=0, digits=4)
    with open(config.REPORTS_DIR / "classification_report.txt", "w",
              encoding="utf-8") as fh:
        fh.write(text_report)

    print()
    print("-" * 68)
    print("WRITTEN")
    print("-" * 68)
    print("  %s" % config.METRICS_FILE.relative_to(config.ROOT))
    print("  %s" % (config.REPORTS_DIR / "classification_report.txt").relative_to(config.ROOT))

    if not args.no_figures:
        apply_style()
        plot_confusion(cm, class_names)
        plot_per_class_f1(class_names, f_cls)
        plot_history(history)

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

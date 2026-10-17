"""Shared model loading and single-image prediction.

Used by src/predict.py (CLI), src/evaluate.py and app/app.py so that all three
load weights and preprocess images identically. The checkpoint records the
img_size it was trained at, so inference configures itself and cannot drift
out of sync with training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.data import build_transforms, get_default_device  # noqa: E402
from src.model import build_model  # noqa: E402
from src.remedies import lookup  # noqa: E402


class ModelNotTrained(FileNotFoundError):
    """Raised when no checkpoint exists yet, so callers can show guidance."""


def load_model(path: Path | None = None, device: torch.device | None = None):
    """Load a trained checkpoint.

    Returns (model, class_names, img_size).
    """
    path = Path(path) if path else config.MODEL_FILE
    if not path.is_file():
        raise ModelNotTrained(
            "no trained model at %s\n"
            "Train one first, e.g.:\n"
            "  python -m src.train --img-size 64 --subset 0.3 --epochs 4" % path)

    device = device or get_default_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)

    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", config.IMG_SIZE))
    model = build_model(ckpt.get("model", "resnet9"), len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, img_size


def _json_safe(value):
    """Coerce a value to something json/jsonify can emit.

    Training args carry a Path for --out, which would make /health raise
    "Object of type WindowsPath is not JSON serializable".
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def checkpoint_info(path: Path | None = None) -> dict:
    """Metadata about a checkpoint without building the model."""
    path = Path(path) if path else config.MODEL_FILE
    if not path.is_file():
        return {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "model": ckpt.get("model"),
        "img_size": ckpt.get("img_size"),
        "val_acc": ckpt.get("val_acc"),
        "num_classes": ckpt.get("num_classes", len(ckpt.get("class_names", []))),
        "epochs_trained": len(ckpt.get("history", []) or []),
        "train_seconds": ckpt.get("train_seconds"),
        "args": _json_safe(ckpt.get("args", {})),
    }


@torch.no_grad()
def predict_image(model, image: Image.Image, class_names, img_size: int,
                  topk: int = 3, device: torch.device | None = None):
    """Classify one PIL image.

    Returns a dict with the top-1 class, its confidence, the top-k list, and
    the matching remedy record. Softmax is applied here rather than in the
    model, because training needs raw logits for cross-entropy.
    """
    device = device or next(model.parameters()).device
    tf = build_transforms(img_size, train=False)
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].cpu()

    k = max(1, min(topk, len(class_names)))
    top_probs, top_idx = torch.topk(probs, k)

    predictions = [
        {
            "class_name": class_names[int(i)],
            "display": lookup(class_names[int(i)])["display"],
            "confidence": float(p),
        }
        for p, i in zip(top_probs, top_idx)
    ]

    best = predictions[0]
    remedy = lookup(best["class_name"])
    confidence = best["confidence"]

    return {
        "class_name": best["class_name"],
        "display": best["display"],
        "confidence": confidence,
        "healthy": bool(remedy["healthy"]),
        "low_confidence": confidence < config.LOW_CONFIDENCE_THRESHOLD,
        "predictions": predictions,
        "remedy": remedy,
    }


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
